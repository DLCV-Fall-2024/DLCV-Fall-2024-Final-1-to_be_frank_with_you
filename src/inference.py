import argparse
import cProfile
import io
import json
import os
import pstats
import re
import time
from pathlib import Path
from pstats import SortKey

import datasets
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    GenerationConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)

from src.arguments import (
    DatasetParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    YamlArgsLoader,
)
from utils.dataset import DiscDataset
from utils.log import PerformanceMonitor, Timer, pretty_print


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Inference script for Traffic specific LLaVA."
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="outputs/pred.json",
        help="File to save the results.",
    )
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        default="configs/inference.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--use_regex",
        type=bool,
        default=False,
        help="Use regex to extract assistant reply. (default: False, use split)",
    )
    mp = ModelParams(parser)
    dp = DatasetParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()

    yaml_file = Path(args.config_file)
    if yaml_file.exists():
        yaml_args = YamlArgsLoader(yaml_file)
        yaml_args.overwrite_args(args)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ################## Below should be wrap in the class ######################
    # TODO: (yck) modify this for a more general use case, e.g. load our own model
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import Trainer, TrainingArguments

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(**args.lora_config)
    model = get_peft_model(model, lora_config)
    model.load_state_dict(
        torch.load("runs/1209_212635/ckpts/1.pt", weights_only=True), strict=False
    )
    model = model.to(device)

    processor = LlavaProcessor.from_pretrained(args.model_id)
    processor.patch_size = args.patch_size
    processor.vision_feature_select_strategy = args.vision_feature_select_strategy

    transform = lambda img, prompt: processor(
        img,
        text=prompt,
        return_tensors="pt",  # return as pytorch tensors
        padding=True,
        do_rescale=False,  # since we already rescale color range to [0, 1] when loading dataset
    )
    ###########################################################################

    ### We don't transform the inputs in the dataset since we don't know the prompt size in advance (fix-sized padding introduces overhead)
    ### Instead, we will transform the inputs in the inference loop.
    dataset = DiscDataset(args.dataset_path, train=False)
    inference_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # max for 20GB GPU
        num_workers=args.num_workers,
    )

    if "generation_config" in vars(args):
        generation_config = GenerationConfig.from_dict(args.generation_config)
    else:
        generation_config = GenerationConfig()

    print("Generation Config:")
    generation_config_diff = generation_config.to_diff_dict()
    sorted(generation_config_diff.keys())
    if len(generation_config_diff.keys()) > 0:
        pretty_print(generation_config_diff)
    print()
    # Perform inference

    out_path = Path(args.output_file)

    timestamp = time.strftime("%m%d%H%M%S")
    dataset_type = args.dataset_path.split("/")[-1]
    if out_path.is_file():
        out_path = Path(args.output_file)
    else:
        out_path = out_path / f"{timestamp}-{dataset_type}" / "submission.json"
    out_config_file = out_path.parent / "config.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if yaml_file:
        yaml_args = YamlArgsLoader(out_config_file)
        if len(generation_config_diff.keys()) > 0:
            setattr(args, "generation_config", generation_config_diff)
        yaml_args.save_args(args, exclude=["config_file", "output_file"])

    data = {}
    timer = Timer(10)  # 10 minutes
    DEBUG = PerformanceMonitor(args.debug)
    for ids, batch in tqdm(inference_loader):
        with DEBUG:
            inputs = transform(batch["image"], prompt=batch["prompt"]).to(
                device, torch.bfloat16
            )
            DEBUG.stamp()
            DEBUG.set_params(**{"ids": ids})

            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                generation_config=generation_config,
                vision_feature_select_strategy=args.vision_feature_select_strategy,
            )

            DEBUG.stamp()
            text = processor.batch_decode(output, skip_special_tokens=True)
            DEBUG.stamp()

            res_len = []
            for idx, item in enumerate(ids):
                if args.use_regex:
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
    main()
