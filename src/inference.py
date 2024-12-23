import os
import sys
import argparse


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# NOTE: (Tom) So I guess config_file is the config we used to train the model we're loading for inference
def main(
    # model_path: str = typer.Argument(
    #     ...,
    #     help="Path to the model checkpoint.",
    # ),
    # ## TODO: We should improve the mechanism to auto find and load the model configuration file
    # config_file: str = typer.Argument(..., help="Path to the configuration file"),
    # output_dir: str = typer.Option(
    #     "results", help="Directory to save the results", show_default=True
    # ),
    # infer_config_path: str = typer.Option(
    #     "configs/inference/default.yaml", help="Path to the generation config file."
    # ),
):
    from pathlib import Path

    from omegaconf import OmegaConf

    from src.arguments.dataclass import GenerateParams, Config
    from src.utils.experiment import load_config, dump_config
    from src.utils import container_to, default

    # Argument parsing
    parser = argparse.ArgumentParser(description="DeepSpeed Training Script")
    parser.add_argument(
        "--name", required=True, type=str, help="Name of the config presets"
    )
    # NOTE: (Tom) Keep for now, in case I'm wrong
    # parser.add_argument(
    #     "--model_config_path",
    #     required=False,
    #     type=str,
    #     help="Path to the model config file",
    # )
    parser.add_argument(
        "--training_dir",
        required=True,
        type=str,
        help="Path to the train output directory",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        type=str,
        help="Path to the checkpoint file (relative to training_dir/checkpoint)",
    )

    GenerateParams().load(parser)
    args = parser.parse_args()

    # config_path = Path(config_file)
    # config_name = config_path.stem
    # config, timestamp, _, _, _ = load_config(
    #     Config, name=config_name, auto_create=False
    # )
    # if config is None:
    #     raise FileNotFoundError("Configuration not found. Please create one.")

    name = args.name
    infer_config, _, *assets = load_config(
        GenerateParams,
        name=name,
        cli_args=GenerateParams().extract(args),
        auto_create=True,
        prefix="inference",
    )
    if infer_config is None:
        print("Configuration created")
        sys.exit()

    output_dir, checkpoint_dir, log_dir = assets
    print(f"Output dir: {output_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Log dir: {log_dir}")

    ## Load train config and merge with inference config
    training_dir = Path(args.training_dir)
    assert training_dir.exists(), f"Train output dir not found at {training_dir}"

    train_config_path = training_dir / "config.yaml"
    assert train_config_path.exists(), f"Train config not found at {train_config_path}"

    # Check if checkpoint exists
    model_path = training_dir / "checkpoint" / args.ckpt_path
    assert model_path.exists(), f"Checkpoint not found at {model_path}"
    infer_config.model_path = str(model_path)

    # train_config_dict = OmegaConf.load(train_config_path)
    # train_config_dict = OmegaConf.to_container(train_config_dict)

    # config_dict = OmegaConf.to_container(infer_config)
    # config_dict["model_config"] = train_config_dict

    # WARNING: (Tom) If current Config is different to the one used to train the model, this will cause issues
    train_config = OmegaConf.load(train_config_path)
    train_config = OmegaConf.structured(Config, train_config)
    train_config: Config = OmegaConf.to_object(train_config)

    infer_config: GenerateParams = OmegaConf.to_object(infer_config)
    infer_config.model_config = train_config
    dump_config(infer_config, output_dir / "config.yaml")

    ic = infer_config

    # NOTE: (Tom) Keep for now, in case I'm wrong
    # model_config_path = getattr(args, "model_config_path", None)
    # if model_config_path is not None:
    #     model_config_path = Path(model_config_path)
    #     assert (
    #         model_config_path.exists()
    #     ), f"Model config not found at {model_config_path}"

    #     model_config_dict = OmegaConf.load(model_config_path)
    #     model_config_dict = OmegaConf.to_container(model_config_dict)

    #     config_dict = OmegaConf.to_container(config)
    #     config_dict["model_config"] = model_config_dict
    #     dump_config(config_dict, output_dir / "config.yaml")

    # model_path: Path = Path(model_path)
    # assert model_path.exists(), f"Model checkpoint not found at {model_path}"

    # output_dir: Path = Path(output_dir)
    # infer_config_path: Path = Path(infer_config_path)

    # infer_config_path.parent.mkdir(parents=True, exist_ok=True)

    # output_dir = output_dir / config_name / timestamp
    # output_dir.mkdir(parents=True, exist_ok=True)

    # infer_config: GenerateParams = OmegaConf.structured(GenerateParams)

    # if not infer_config_path.exists():
    #     OmegaConf.save(infer_config, infer_config_path)
    # else:
    #     infer_config: GenerateParams = OmegaConf.load(infer_config_path)
    # infer_config.config_path = str(config_path)
    # infer_config.output_dir = str(output_dir)
    # infer_config.model_path = str(model_path)
    # infer_config.model_config = config
    # OmegaConf.save(infer_config, str(output_dir / "config.yaml"))

    # ic = infer_config

    import json
    import re

    import torch
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import GenerationConfig
    from transformers.utils import logging

    from src.models.llava import LlavaPEFT, collate_fn
    from src.utils.dataset import DiscDataset
    from src.utils.log import PerformanceMonitor, Timer, pretty_print

    logging.set_verbosity_error()
    mp = train_config.model

    seed_everything(ic.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train_config.liger_kernel:
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
    print("Loading model from checkpoint: ", model_path)
    ckpt = torch.load(open(model_path, "rb"), map_location=device)
    if "ds_version" in ckpt.keys():
        print("Loading DeepSpeed checkpoint")
        model.load_state_dict(ckpt["module"])
    else:
        # TODO: Make sure this is correct
        print("Loading normal checkpoint")
        model.load_state_dict(ckpt)
    print("Model loaded successfully")

    transform = model.transform
    processor = model.processor

    for param in model.parameters():
        param.requires_grad = False
    model.to(device, torch.bfloat16)

    timer = Timer(10 * 60)  # 10 minutes
    transform = model.transform

    num_workers = ic.num_workers
    prefetch_factor = ic.prefetch_factor if num_workers > 0 else None

    dataset = DiscDataset(
        ic.dataset_path,
        transform=transform,
        train=False,
        use_processed=mp.use_processed,
        depth_model_id=mp.depth_model_id,
        segmentation_model_id=mp.segmentation_model_id,
    )
    inference_loader = DataLoader(
        dataset,
        batch_size=ic.batch_size,  # max for 20GB GPU
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )

    generation_config = GenerationConfig.from_dict(ic.generation_config)

    print("Generation Config:")
    generation_config_diff = generation_config.to_diff_dict()
    sorted(generation_config_diff.keys())
    if len(generation_config_diff.keys()) > 0:
        pretty_print(generation_config_diff)
    print()
    # Perform inference

    out_path = output_dir / "submission.json"

    data = {}
    timer = Timer(10 * 60)  # 10 minutes
    DEBUG = PerformanceMonitor(True)
    model.eval()
    model = model.merge_and_unload(inplace=True)
    for ids, batch in tqdm(inference_loader):
        with DEBUG:
            target_dtypes = [torch.float16, torch.float32, torch.float64]
            inputs = container_to(
                batch, target_dtypes, device=device, dtype=torch.bfloat16
            )
            # `input_ids` and `attention_mask` should be long tensors
            inputs["input_ids"] = inputs["input_ids"].to(device, torch.long)
            inputs["attention_mask"] = inputs["attention_mask"].to(device, torch.long)
            print(
                {k: v.dtype for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            )

            DEBUG.stamp()
            DEBUG.set_params(**{"ids": ids})

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=ic.max_new_tokens,
                    generation_config=generation_config,
                    vision_feature_select_strategy=mp.vision_feature_select_strategy,
                )

            DEBUG.stamp()
            text = processor.batch_decode(output, skip_special_tokens=True)
            DEBUG.stamp()

            res_len = []
            for idx, item in enumerate(ids):
                if ic.use_regex:
                    match = re.search(r"ASSISTANT:\s*(.*)", text[idx])
                    assistant_reply = match.group(1) if match else ""
                else:
                    parts = text[idx].split("ASSISTANT:", 1)
                    assistant_reply = parts[1].strip() if len(parts) > 1 else ""
                data[item] = assistant_reply
                res_len.append(len(assistant_reply))

            DEBUG.set_params(**{"assistant_reply": sum(res_len) / len(res_len)})
            DEBUG.stamp()
            DEBUG.log_performance(log_per=20)
        if timer.timesup():
            ## Save the results every 10 minutes
            timer.restart()
            with open(out_path, "w") as json_file:
                json.dump(data, json_file)
    with open(out_path, "w") as json_file:
        json.dump(data, json_file)


if __name__ == "__main__":
    # app()
    main()
