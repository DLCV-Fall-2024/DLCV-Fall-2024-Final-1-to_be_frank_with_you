import argparse
import cProfile
import io
import json
import os
import pstats
import re
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

from src.arguments import DatasetParams, ModelParams, PipelineParams, YamlArgsLoader
from utils.dataset import DiscDataset
from utils.log import PerformanceMonitor, pretty_print


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

    # TODO: modify this for a more general use case, e.g. load our own model
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
    ).to(device)
    processor = LlavaProcessor.from_pretrained(args.model_id)
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"

    transform = lambda img, prompt: processor(
        img,
        text=prompt,
        return_tensors="pt",
        padding=True,
        do_rescale=False,
    )

    dataset = DiscDataset(args.dataset_path, train=False)  # , transform=transform)
    inference_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # max for 20GB GPU
        num_workers=args.num_workers,
    )

    generation_config = GenerationConfig.from_dict(args.__dict__)
    print("Generation Config:")
    pretty_print(generation_config.__dict__)
    print()
    # Perform inference
    data = {}
    debug = PerformanceMonitor(args.debug)

    out_config_file = Path(args.output_file).parent / "config.yaml"
    if yaml_file:
        yaml_args = YamlArgsLoader(out_config_file)
        yaml_args.save_args(args)

    # raise ValueError("This is a test error")
    for ids, batch in tqdm(inference_loader):
        with debug:
            # inputs = batch.to(device, torch.float16)
            inputs = transform(batch["image"], prompt=batch["prompt"]).to(
                device, torch.float16
            )
            debug.stamp()
            debug.set_params(**{"ids": ids})

            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                generation_config=generation_config
            )

            debug.stamp()
            text = processor.batch_decode(output, skip_special_tokens=True)
            debug.stamp()

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

            debug.set_params(**{"assistant_reply": sum(res_len) / len(res_len)})
            debug.stamp()
            debug.log_performance(log_per=20)

        # with cProfile.Profile() as pr:
        # stream = io.StringIO()
        # stats = pstats.Stats(pr, stream=stream)
        # stats.sort_stats("time")  # Sort by time
        # stats.print_stats(10)  # Display top 10 results
        # print(stream.getvalue())

    # Save results to output file
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    main()
