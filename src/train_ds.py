from typing import Any, Tuple
import os
import sys
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # Load configuration from preset and CLI arguments
    from src.utils.experiment import load_config
    from src.arguments.deepspeed import Config

    # Argument parsing
    parser = argparse.ArgumentParser(description="DeepSpeed Training Script")
    parser.add_argument("--name", type=str, help="Name of the config presets")
    parser.add_argument("--local_rank", type=int, required=True, help="Local rank")

    Config().load(parser)
    args = parser.parse_args()

    # Get DeepSpeed metadata
    import deepspeed
    from torch import distributed as dist

    name = args.name
    local_rank = args.local_rank

    dist.init_process_group()
    world_size = dist.get_world_size()

    accelerator = deepspeed.get_accelerator()
    device = f"{accelerator.device_name()}:{local_rank}"

    config, timestamp, *assets = load_config(
        Config,
        name=name,
        cli_args=Config().extract(args),
        external_defaults=[(["deepspeed", "config"], "deepspeed_default")],
        sync_fn=lambda: dist.barrier(),
        auto_create=local_rank == 0,
    )
    if config is None:
        print("Configuration created")
        sys.exit()

    output_dir, checkpoint_dir, log_dir = assets

    import warnings
    from pathlib import Path
    from omegaconf import OmegaConf
    from tqdm import tqdm

    import torch
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader
    from torch.utils.data import DistributedSampler

    from src.models.llava import LlavaPEFT, collate_fn
    from src.utils.experiment import dump_config
    from src.utils import container_to, default
    from src.utils.dataset import DiscDataset
    from src.utils.log import (
        PerformanceMonitor,
        Timer,
        Profiler,
        # init_logger,
        # init_wandb,
        print_once,
    )

    addition_config = {}
    mp = config.model
    dp = config.dataset
    dsp = config.deepspeed

    seed_everything(config.seed)

    if config.liger_kernel:
        apply_liger_kernel_to_llama()

    model = LlavaPEFT(
        model_params=mp,
        gradient_checkpointing=True,
        lora_config=mp.lora_config,
        # device=device,
        torch_dtype=torch.bfloat16,
    )

    addition_config["model_struct"] = model.get_model_struct()

    transform = model.transform
    processor = model.processor

    trainable_params, all_param = model.llava.get_nb_trainable_parameters()
    print(
        f"Trainable: {trainable_params/1e6:.4f}M | All: {all_param/1e6:.4f}M | Ratio: {trainable_params/all_param * 100:.3f}%"
    )

    dump_config(config, output_dir / "config.yaml")
    dump_config(addition_config, output_dir / "model_config.yaml")
    del addition_config

    ### We don't transform the inputs in the dataset since we don't know the prompt size in advance (fix-sized padding introduces overhead)
    ### Instead, we will transform the inputs in the inference loop.
    dataset_dir = Path(dp.dataset_path)
    train_set = dataset_dir / "train"
    val_set = dataset_dir / "val"
    assert (
        train_set.exists() and val_set.exists()
    ), f"Dataset not found. {dataset_dir} should contain 'train' and 'val' folders."

    num_workers = dp.num_workers
    prefetch_factor = dp.prefetch_factor if num_workers > 0 else None

    train_dataset = DiscDataset(
        train_set,
        transform=transform,
        train=True,
        use_processed=mp.use_processed,
        depth_model_id=mp.depth_model_id,
        segmentation_model_id=mp.segmentation_model_id,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=dsp.batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    val_dataset = DiscDataset(
        val_set,
        transform=transform,
        train=True,
        use_processed=mp.use_processed,
        depth_model_id=mp.depth_model_id,
        segmentation_model_id=mp.segmentation_model_id,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dsp.batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )

    ######################### Logging ########################

    if config.wandb:
        team = "DLCV-Final"
        project = default(config.project_name, "DLCV-FINAL-Traffic-LLaVA")
        group = default(config.run_name, f"{name}-{mp.model_id.split('/')[-1]}-{timestamp}")
        dsp.config["wandb"] = dict(
            enabled=True,
            team=team,
            project=project,
            group=group,
        )

    # logger = init_logger(local_rank=local_rank)
    # print(type(logger))

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


    def initialize_model(model) -> Tuple[deepspeed.DeepSpeedEngine, Any]:
        model_engine, _, _, scheduler = deepspeed.initialize(
            config=OmegaConf.to_container(dsp.config, resolve=True),
            model=model,
        )
        model_engine: deepspeed.DeepSpeedEngine
        return model_engine, scheduler


    model.finetune_language(config.finetune_language)
    model_engine, scheduler = initialize_model(model)

    if config.resume:
        print(f"Loading checkpoint from {config.resume}")
        model_engine.load_checkpoint(
            config.resume,
            tag=config.resume_tag,
            load_module_strict=False,
            load_module_only=True,
        )

    with Profiler(profile=config.profile) as PROFILER:
        for epoch in range(dsp.epochs):
            model_engine.train()

            train_bar = tqdm(train_loader)
            train_bar.set_description(f"[Train {epoch}/{dsp.epochs}]")
            for ids, batch in train_bar:
                with DEBUG:
                    # `batch` is a nested dict with keys: `pixel_values`, `aux_inputs`, `input_ids`, `attention_mask`
                    # `aux_inputs` is a list of nested dict
                    target_dtypes = [torch.float16, torch.float32, torch.float64]
                    inputs = container_to(
                        batch, target_dtypes, device=device, dtype=torch.bfloat16
                    )
                    # `input_ids` and `attention_mask` should be long tensors
                    inputs["input_ids"] = inputs["input_ids"].to(torch.long)
                    inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
                    labels = inputs["input_ids"].clone()

                    DEBUG.stamp()

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="None of the inputs have requires_grad=True. Gradients will be None",
                        )
                        outputs = model_engine.forward(
                            **inputs,
                            labels=labels,
                            vision_feature_select_strategy=mp.vision_feature_select_strategy,
                            use_cache=False,
                            # other_params={"ids": ids, **batch},
                        )
                    DEBUG.stamp()

                    DEBUG.set_params(**{"loss": outputs.loss})
                    loss = outputs.loss
                    model_engine.backward(loss)

                    if config.debug:
                        for name, param in model.named_parameters():
                            if param.requires_grad and (
                                param.grad is None or torch.isnan(param.grad).any()
                            ):
                                print_once(f"Warning: {name}.grad is {param.grad}.")

                    # weight update
                    model_engine.step()
                    global_step += 1

                    # NOTE: DeepSpeed provides its own logging
                    # logger({"train/loss": loss, "global_step": global_step})
                    # if global_step % dsp.accumulation_steps == 0:
                    #     logger(
                    #         {
                    #             "train/lr": scheduler.get_last_lr(),
                    #             "global_step": global_step,
                    #         }
                    #     )

                    DEBUG.stamp()
                    DEBUG.stamp()
                    DEBUG.log_performance(log_per=20)
                    del outputs, loss, labels

                    dist.barrier()
                    if timer.timesup():
                        model_engine.save_checkpoint(
                            checkpoint_dir, exclude_frozen_parameters=True
                        )
                        timer.reset()

            model_engine.eval()

            val_bar = tqdm(val_loader)
            val_bar.set_description(f"[Val {epoch}/{dsp.epochs}]")
            # NOTE: Disabled since it requires re-initializing the model_engine
            # model.merge_adapter()
            with torch.no_grad():
                for ids, batch in val_bar:
                    with DEBUG:
                        # `batch` is a nested dict with keys: `pixel_values`, `aux_inputs`, `input_ids`, `attention_mask`
                        # `aux_inputs` is a list of nested dict
                        target_dtypes = [torch.float16, torch.float32, torch.float64]
                        inputs = container_to(
                            batch, target_dtypes, device=device, dtype=torch.bfloat16
                        )
                        # `input_ids` and `attention_mask` should be long tensors
                        inputs["input_ids"] = inputs["input_ids"].to(torch.long)
                        inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
                        labels = inputs["input_ids"].clone()

                        DEBUG.stamp()
                        DEBUG.set_params(**{"labels": labels})

                        outputs = model_engine.forward(
                            **inputs,
                            labels=labels,
                            vision_feature_select_strategy=mp.vision_feature_select_strategy,
                        )
                        loss = outputs.loss

                        # logger({"val/loss": loss.item()})

                        DEBUG.stamp()
                        DEBUG.log_performance(log_per=20)
                        del outputs, loss, labels

            dist.barrier()
            model_engine.save_checkpoint(
                checkpoint_dir, tag=f"{epoch}", exclude_frozen_parameters=True
            )

    if local_rank == 0:
        PROFILER.export(output_dir / "trace.json")


if __name__ == "__main__":
    main()
