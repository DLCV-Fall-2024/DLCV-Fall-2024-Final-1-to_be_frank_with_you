import typer

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    name: str = typer.Argument(..., help="Name of the experiment"),
    local_rank: int = typer.Option(0, "--local_rank", help="Local rank"),
):
    from src.arguments.deepspeed import Config
    from src.utils.experiment import load_config

    assets = load_config(
        name,
        Config,
        auto_create=local_rank == 0,
        external_defaults=[(["deepspeed", "config"], "deepspeed_default")],
    )
    config, timestamp, output_dir, checkpoint_dir, log_dir = assets
    if config is None:
        print("Configuration created")
        return

    import os
    from pathlib import Path
    import json
    from omegaconf import OmegaConf
    from tqdm import tqdm

    import torch
    import deepspeed
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader
    from torch.utils.data import DistributedSampler

    # from src.models.llava import LlavaPEFT
    from src.models.llava_ds_test import LlavaPEFT
    from src.utils.experiment import dump_additional_config
    from src.utils import default
    from src.utils.dataset import DiscDataset
    from src.utils.log import (
        PerformanceMonitor,
        Timer,
        Profiler,
        init_logger,
        init_wandb,
    )

    from torch.profiler import profile, record_function, ProfilerActivity

    addition_config = {}
    mp = config.model
    dp = config.dataset
    dsp = config.deepspeed

    seed_everything(config.seed)

    world_size = torch.cuda.device_count()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.liger_kernel:
        apply_liger_kernel_to_llama()

    model = LlavaPEFT(
        model_params=mp,
        gradient_checkpointing=False,
        lora_config=mp.lora_config,
    )
    addition_config["model_struct"] = model.get_model_struct()

    transform = model.transform
    processor = model.processor

    # trainable_params, all_param = model.llava.get_nb_trainable_parameters()
    # print(
    #     f"Trainable: {trainable_params/1e6:.4f}M | All: {all_param/1e6:.4f}M | Ratio: {trainable_params/all_param * 100:.3f}%"
    # )
    # WARNING: Test
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()
    print(f"Total trainable parameters: {total / 1e6:.4f}M")

    # for name, param in model.named_parameters():
    #     if not (param.requires_grad == ("lora" in name)):
    #         print(f"Warning: {name}.required_grad= {param.requires_grad}.")

    dump_additional_config(addition_config, output_dir)
    del addition_config

    ### We don't transform the inputs in the dataset since we don't know the prompt size in advance (fix-sized padding introduces overhead)
    ### Instead, we will transform the inputs in the inference loop.
    dataset_dir = Path(dp.dataset_path)
    train_set = dataset_dir / "train"
    val_set = dataset_dir / "val"
    assert (
        train_set.exists() and val_set.exists()
    ), f"Dataset not found. {dataset_dir} should contain 'train' and 'val' folders."

    train_dataset = DiscDataset(train_set, train=True)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=dsp.batch_size,
        prefetch_factor=dp.prefetch_factor,
        num_workers=dp.num_workers,
        sampler=train_sampler,
    )

    val_dataset = DiscDataset(val_set, train=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=dsp.batch_size,
        num_workers=dp.num_workers,
    )

    ######################### Logging ########################

    if config.wandb:
        project_name = default(config.project_name, "DLCV-FINAL-Traffic-LLaVA")
        run_name = default(
            config.run_name, f"{name}-{mp.model_id.split('/')[-1]}-{timestamp}"
        )

        # DeepSpeed doesn't provide run_name setting, so manually init is required
        print(f"Logging to {run_name}")
        init_wandb(
            project_name,
            run_name,
            config=OmegaConf.to_container(config, resolve=True),
            log_dir=log_dir,
            local_rank=local_rank,
        )

        dsp.config["wandb"] = dict(enabled=True)

    logger = init_logger(local_rank=local_rank)
    print(type(logger))

    timer = Timer(10 * 60)  # 10 minutes
    DEBUG = PerformanceMonitor(config.debug)
    global_step = 0

    ###################### Optimization ######################

    # Training settings
    dsp.config["train_micro_batch_size_per_gpu"] = dsp.batch_size
    dsp.config["gradient_accumulation_steps"] = dsp.accumulation_steps

    # Optimizer settings
    dsp.config["optimizer"] = dict(
        type="AdamW",
        params=dict(lr=dsp.learning_rate),
    )

    # Scheduler settings
    # Definition of `step`:
    #   scheduler step: number of data point per gradient update
    #   wandb step: one data point
    # Note: gradient is updated when all ranks have finished accumulating gradients
    total_steps = len(train_dataset) * dsp.epochs
    effective_batch_size = dsp.batch_size * world_size
    grad_acc_steps = dsp.accumulation_steps
    total_update_steps = total_steps // (effective_batch_size * grad_acc_steps)

    first_step_size = total_update_steps * 0.3
    second_step_size = total_update_steps * 0.7
    dsp.config["scheduler"] = dict(
        type="OneCycle",
        params=dict(
            cycle_first_step_size=first_step_size,
            cycle_first_stair_count=first_step_size * 0.5,
            cycle_second_step_size=second_step_size,
            cycle_second_stair_count=second_step_size * 0.5,
            decay_step_size=0,
            cycle_max_lr=dsp.learning_rate,
            cycle_min_lr=dsp.learning_rate / 25,
            decay_lr_rate=0.001,
            cycle_min_mom=0.85,
            cycle_max_mom=0.99,
            decay_mom_rate=0.0,
        ),
    )

    ###################### Training ######################

    # DeepSpeed read config from file
    model_engine, _, _, _ = deepspeed.initialize(
        config=OmegaConf.to_container(dsp.config, resolve=True),
        model=model,
    )

    with Profiler(profile=config.profile) as PROF:
        for epoch in range(dsp.epochs):
            train_bar = tqdm(train_loader)
            train_bar.set_description(f"[Train {epoch}/{dsp.epochs}]")
            for ids, batch in train_bar:
                with DEBUG:
                    inputs = transform(
                        batch["image"],
                        prompt=batch["prompt"],
                    ).to(device=local_rank)
                    inputs["labels"] = torch.zeros(
                        dsp.batch_size,
                        dtype=torch.long,
                        device=local_rank,
                    )

                    # for k, v in inputs.items():
                    #     if torch.is_tensor(v) and v.dtype in [
                    #         torch.float32,
                    #         torch.float64,
                    #         torch.float16,
                    #         torch.bfloat16,
                    #     ]:
                    #         v.requires_grad = True
                    # labels = inputs["input_ids"].clone()

                    # DEBUG.stamp()
                    # DEBUG.set_params(**{"labels": labels})

                    outputs = model_engine.forward(inputs)
                    loss = outputs.loss
                    model_engine.backward(loss)

                    if config.debug:
                        for name, param in model.named_parameters():
                            if param.requires_grad and param.grad is None:
                                print(f"Warning: {name}.grad is {param.grad}.")

                    # weight update
                    model_engine.step()
                    global_step += 1
                    # logger({"train/loss": loss, "global_step": global_step})
                    # logger(
                    #     {
                    #         "train/lr": scheduler.get_last_lr(),
                    #         "global_step": global_step,
                    #     }
                    # )

                    # DEBUG.stamp()
                    # DEBUG.stamp()
                    # DEBUG.log_performance(log_per=20)
                    del outputs, loss

                    if timer.timesup():
                        model_engine.save_checkpoint(checkpoint_dir)
                        timer.reset()

    if local_rank == 0:
        PROF.export(output_dir / "trace.json")


if __name__ == "__main__":
    app()
