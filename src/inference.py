import os

import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings


@app.command()
def main(
    model_path: str = typer.Argument(
        ...,
        help="Path to the model checkpoint.",
    ),
    ## TODO: We should improve the mechanism to auto find and load the model configuration file
    config_file: str = typer.Argument(..., help="Path to the configuration file"),
    output_dir: str = typer.Option(
        "results", help="Directory to save the results", show_default=True
    ),
    infer_config_path: str = typer.Option(
        "configs/inference/default.yaml", help="Path to the generation config file."
    ),
):
    from pathlib import Path

    from omegaconf import OmegaConf

    from src.arguments.dataclass import Config, GenerateParams
    from src.utils.experiment import load_config

    config_path = Path(config_file)
    config_name = config_path.stem
    config, timestamp, _, _, _ = load_config(config_name, Config, auto_create=False)
    if config is None:
        raise FileNotFoundError("Configuration not found. Please create one.")

    model_path: Path = Path(model_path)
    assert model_path.exists(), f"Model checkpoint not found at {model_path}"

    output_dir: Path = Path(output_dir)
    infer_config_path: Path = Path(infer_config_path)

    infer_config_path.parent.mkdir(parents=True, exist_ok=True)

    output_dir = output_dir / config_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    infer_config: GenerateParams = OmegaConf.structured(GenerateParams)

    if not infer_config_path.exists():
        OmegaConf.save(infer_config, infer_config_path)
    else:
        infer_config: GenerateParams = OmegaConf.load(infer_config_path)
    infer_config.config_path = str(config_path)
    infer_config.output_dir = str(output_dir)
    infer_config.model_path = str(model_path)
    infer_config.model_config = config
    OmegaConf.save(infer_config, str(output_dir / "config.yaml"))

    ic = infer_config
    import json
    import re

    import torch
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import GenerationConfig
    from transformers.utils import logging

    from src.models.llava import LlavaPEFT
    from src.utils.dataset import DiscDataset
    from src.utils.log import PerformanceMonitor, Timer, pretty_print

    logging.set_verbosity_error()
    mp = config.model

    seed_everything(infer_config.seed)
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
    print("Loading model from checkpoint: ", model_path)
    model.load_state_dict(
        torch.load(
            open(model_path, "rb"),
            map_location=device,
            weights_only=True,
        )
    )
    print("Model loaded successfully")

    transform = model.transform
    processor = model.processor

    for param in model.parameters():
        param.requires_grad = False
    model.to(device, torch.bfloat16)

    timer = Timer(10 * 60)  # 10 minutes

    dataset = DiscDataset(ic.dataset_path, train=False)
    inference_loader = DataLoader(
        dataset,
        batch_size=ic.batch_size,  # max for 20GB GPU
        num_workers=ic.num_workers,
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
            inputs = transform(batch["image"], prompt=batch["prompt"]).to(
                device, torch.bfloat16
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
    app()
