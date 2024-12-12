import typer

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(name: str = typer.Argument(..., help="Name of the experiment")):

    from tqdm import tqdm
    from pathlib import Path
    from omegaconf import OmegaConf

    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from pytorch_lightning import seed_everything

    import transformers
    from transformers import DepthAnythingForDepthEstimation

    from src.arguments.dataclass import Config
    from src.utils.experiment import load_config, dump_additional_config
    from src.models.llava import LlavaPEFT

    from utils import default
    from utils.dataset import DiscDataset
    from utils.log import (
        PerformanceMonitor,
        Timer,
        init_logger,
        init_wandb,
        pretty_print,
    )

    config, timestamp, output_dir, checkpoint_dir, log_dir = load_config(name, Config)
    if config is None:
        print("Configuration created")
        return

    addition_config = {}
    mp = config.model
    dp = config.dataset
    pp = config.pipeline
    op = config.optimization

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### DeepSpeed Compatibility ###
    __USE_DEEPSPEED__ = False
    local_rank = -1
    use_cache = not (mp.gradient_checkpointing and not __USE_DEEPSPEED__)

    model = LlavaPEFT(
        model_params=mp,
        gradient_checkpointing=not use_cache,
        lora_config=mp.lora_config,
    )
    addition_config["model_struct"] = model.get_model_struct()

    transform = model.transform
    processor = model.processor

    trainable_params, all_param = model.llava.get_nb_trainable_parameters()
    print(
        f"Trainable: {trainable_params/1e6:.4f}M | All: {all_param/1e6:.4f}M | Ratio: {trainable_params/all_param * 100:.3f}%"
    )

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Trainable: {name}")

    # model.to(device)

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=op.batch_size,
        prefetch_factor=dp.prefetch_factor,
        num_workers=dp.num_workers,
        shuffle=True,
    )

    val_dataset = DiscDataset(val_set, train=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=op.batch_size,
        num_workers=dp.num_workers,
    )
    ###################### Optimization ######################

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR, StepLR

    training_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(training_params, lr=op.lr)
    scheduler = OneCycleLR(
        optimizer, max_lr=op.lr, steps_per_epoch=len(train_loader), epochs=op.epochs
    )
    optimizer.zero_grad()

    accum_steps = getattr(op, "accumulation_steps", 1)
    print(f"Effective batch size: {op.batch_size * accum_steps}")
    print(f"Using {device} device")

    ##########################################################

    project_name = default(config.project_name, "DLCV-FINAL-Traffic-LLaVA")
    run_name = default(
        config.run_name, f"{name}-{mp.model_id.split('/')[-1]}-{timestamp}"
    )

    if pp.wandb:
        init_wandb(
            project_name,
            run_name,
            config=OmegaConf.to_container(config, resolve=True),
            log_dir=log_dir,
            local_rank=local_rank if __USE_DEEPSPEED__ else None,
        )
    logger = init_logger(local_rank=local_rank if __USE_DEEPSPEED__ else None)

    epochs = op.epochs
    timer = Timer(10)  # 10 minutes
    DEBUG = PerformanceMonitor(pp.debug)
    global_step = 0
    accum_loss = 0
    for name, param in model.named_parameters():
        if not (param.requires_grad == ("lora" in name)):
            print(f"Warning: {name}.required_grad= {param.requires_grad}.")

    model.to(device)

    for epoch in range(epochs):
        model.train()

        train_bar = tqdm(train_loader)
        train_bar.set_description(f"[Train {epoch}/{epochs}]")
        for ids, batch in train_bar:
            with DEBUG:
                inputs = transform(batch["image"], prompt=batch["prompt"]).to(
                    device, torch.bfloat16
                )
                for k, v in inputs.items():
                    if torch.is_tensor(v) and v.dtype in [
                        torch.float32,
                        torch.float64,
                        torch.float16,
                        torch.bfloat16,
                    ]:
                        v.requires_grad = True
                labels = inputs["input_ids"].clone()

                DEBUG.stamp()
                DEBUG.set_params(**{"labels": labels})
                out = model.forward(
                    **inputs,
                    labels=labels,
                    vision_feature_select_strategy=mp.vision_feature_select_strategy,
                    use_cache=use_cache,
                )
                loss = out.loss  # / accum_steps
                loss.backward()
                if pp.debug:
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is None:
                            print(f"Warning: {name}.grad is {param.grad}.")
                global_step += 1
                accum_loss += loss.item()
                # if global_step % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                logger({"train/loss": accum_loss, "global_step": global_step})
                logger(
                    {
                        "train/lr": scheduler.get_last_lr()[0],
                        "global_step": global_step,
                    }
                )
                accum_loss = 0

                DEBUG.stamp()
                DEBUG.stamp()
                DEBUG.log_performance(log_per=20)
                del out, loss, labels

                if timer.timesup():
                    ## save model for a certain interval
                    ckpt_path = checkpoint_dir / f"latest.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    timer.reset()

        model.eval()
        val_bar = tqdm(val_loader)
        val_bar.set_description(f"[Val {epoch}/{epochs}]")
        with torch.no_grad():
            for ids, batch in val_bar:
                with DEBUG:
                    inputs = transform(batch["image"], prompt=batch["prompt"]).to(
                        device
                    )
                    labels = processor.tokenizer(
                        batch["labels"], return_tensors="pt", padding=True
                    ).input_ids.to(device)
                    DEBUG.stamp()
                    DEBUG.set_params(**{"labels": labels})

                    out = model.forward(
                        **inputs,
                        labels=labels,
                        vision_feature_select_strategy=mp.vision_feature_select_strategy,
                    )
                    loss = out.loss

                    logger({"val/loss": loss.item()})

                    DEBUG.stamp()
                    DEBUG.log_performance(log_per=20)

        ckpt_path = checkpoint_dir / f"model-{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    app()
