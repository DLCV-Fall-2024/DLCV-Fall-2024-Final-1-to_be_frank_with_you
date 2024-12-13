import typer

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(name: str = typer.Argument(..., help="Name of the experiment")):
    from src.arguments.deepspeed import Config
    from src.utils.experiment import load_config

    config, timestamp, output_dir, checkpoint_dir, log_dir = load_config(name, Config)
    if config is None:
        print("Configuration created")
        return

    from pathlib import Path

    import torch
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    import deepspeed
    from omegaconf import OmegaConf
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import json

    from src.models.llava import LlavaPEFT
    from src.utils.experiment import dump_additional_config
    from utils import default
    from utils.dataset import DiscDataset
    from utils.log import PerformanceMonitor, Timer, init_logger, init_wandb

    addition_config = {}
    mp = config.model
    dp = config.dataset
    pp = config.pipeline
    op = config.optimization
    dsp = config.deepspeed

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    trainable_params, all_param = model.llava.get_nb_trainable_parameters()
    print(
        f"Trainable: {trainable_params/1e6:.4f}M | All: {all_param/1e6:.4f}M | Ratio: {trainable_params/all_param * 100:.3f}%"
    )

    for name, param in model.named_parameters():
        if not (param.requires_grad == ("lora" in name)):
            print(f"Warning: {name}.required_grad= {param.requires_grad}.")

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
            local_rank=config.local_rank,
        )
    logger = init_logger(local_rank=config.local_rank)
    print(type(logger))

    timer = Timer(10 * 60)  # 10 minutes
    DEBUG = PerformanceMonitor(pp.debug)

    ###################### Optimization ######################

    # DeepSpeed read config from file
    deepspeed_config_path = output_dir / "deepspeed_config.json"
    with open(deepspeed_config_path, "w") as f:
        json.dump(OmegaConf.to_container(dsp.config, resolve=True), f)

    deepspeed_args = lambda: None  # "object" type cannot assign attribute
    deepspeed_args.deepspeed_config = deepspeed_config_path
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=deepspeed_args,
        model=model,
    )

    for epoch in range(op.epochs):
        train_bar = tqdm(train_loader)
        train_bar.set_description(f"[Train {epoch}/{op.epochs}]")
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


if __name__ == "__main__":
    app()
