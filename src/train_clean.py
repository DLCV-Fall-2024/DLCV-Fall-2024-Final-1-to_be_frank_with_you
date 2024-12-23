import argparse
import os
import sys


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    # Load configuration from preset and CLI arguments
    from src.arguments.dataclass import Config
    from src.utils.experiment import load_config

    # Argument parsing
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--name", type=str, help="Name of the config presets")

    Config().load(parser)
    args = parser.parse_args()

    name = args.name
    config, timestamp, *assets = load_config(
        Config,
        name=name,
        cli_args=args,
        auto_create=True,
    )
    if config is None:
        print("Configuration created")
        sys.exit()

    output_dir, checkpoint_dir, log_dir = assets

    import warnings
    from pathlib import Path

    import torch
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    from omegaconf import OmegaConf
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers.utils import logging

    logging.set_verbosity_error()

    from src.models.llava import LlavaPEFT, collate_fn
    from src.utils import container_to, default, dataclass_to_dict
    from src.utils.dataset import DiscDataset
    from src.utils.experiment import dump_config
    from src.utils.log import (
        PerformanceMonitor,
        Timer,
        init_logger,
        init_wandb,
        print_once,
    )

    addition_config = {}
    mp = config.model
    dp = config.dataset
    pp = config.pipeline
    op = config.optimization

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.liger_kernel:
        apply_liger_kernel_to_llama()
    ### DeepSpeed Compatibility ###
    __USE_DEEPSPEED__ = False
    local_rank = -1
    use_cache = not (mp.gradient_checkpointing and not __USE_DEEPSPEED__)

    model = LlavaPEFT(
        model_params=mp,
        gradient_checkpointing=not use_cache,
        lora_config=mp.lora_config,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    if resume := config.resume:
        model.load_state_dict(
            torch.load(
                resume,
                weights_only=True,
            )
        )
    # ensure_all_on_device(model, device)
    # ensure_all_same_dtype(model, torch.bfloat16)
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

    dump_config(config, output_dir / "config.yaml")
    dump_config(addition_config, output_dir / "model_config.yaml")
    del addition_config

    ###################### Dataset ######################

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=op.batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        shuffle=True,
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=op.batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        collate_fn=collate_fn,
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
    gradient_clip_val = getattr(op, "gradient_clip_val", 1.0)
    ##########################################################

    project_name = default(config.project_name, "DLCV-FINAL-Traffic-LLaVA")
    run_name = default(
        config.run_name, f"{name}-{mp.model_id.split('/')[-1]}-{timestamp}"
    )

    if pp.wandb:
        init_wandb(
            project_name,
            run_name,
            config=dataclass_to_dict(config),
            log_dir=log_dir,
            local_rank=local_rank if __USE_DEEPSPEED__ else None,
            entity="DLCV-Final",
        )
    logger = init_logger(local_rank=local_rank if __USE_DEEPSPEED__ else None)
    print(type(logger))
    epochs = op.epochs
    timer = Timer(10 * 60)  # 10 minutes
    DEBUG = PerformanceMonitor(pp.debug)
    global_step = 0
    accum_loss = 0
    for name, param in model.named_parameters():
        if not (param.requires_grad == ("lora" in name)):
            print(f"Warning: {name}.required_grad= {param.requires_grad}.")

    model.to(device)
    model.finetune_language(False)  # turn off language finetuning
    for epoch in range(epochs):
        model.train()

        train_bar = tqdm(train_loader)
        train_bar.set_description(f"[Train {epoch}/{epochs}]")
        if epoch == op.train_language_start_epoch:
            model.finetune_language(True)
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
                    out = model.forward(
                        **inputs,
                        labels=labels,
                        vision_feature_select_strategy=mp.vision_feature_select_strategy,
                        use_cache=use_cache,
                    )
                DEBUG.stamp()

                DEBUG.set_params(**{"loss": out.loss})
                loss = out.loss  # / accum_steps
                loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                if pp.debug:
                    for name, param in model.named_parameters():
                        if param.requires_grad and (
                            param.grad is None or torch.isnan(param.grad).any()
                        ):
                            print_once(f"Warning: {name}.grad is {param.grad}.")
                global_step += op.batch_size
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
                    torch.save(model.enssetial_state_dict(), ckpt_path)
                    timer.reset()

        model.eval()
        val_bar = tqdm(val_loader)
        val_bar.set_description(f"[Val {epoch}/{epochs}]")
        model.merge_adapter()
        model.to(device, torch.bfloat16)
        with torch.no_grad():
            for ids, batch in val_bar:
                with DEBUG:
                    inputs = transform(batch["image"], prompt=batch["prompt"]).to(
                        device, torch.bfloat16
                    )
                    labels = inputs["input_ids"].clone()
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
                del out, loss, labels

        ckpt_path = checkpoint_dir / f"{epoch}.pt"
        model.unmerge_adapter()
        torch.save(model.enssetial_state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
