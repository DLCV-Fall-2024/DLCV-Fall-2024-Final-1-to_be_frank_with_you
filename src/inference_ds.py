import argparse
import os
import sys
from typing import Any, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    from pathlib import Path

    # from src.arguments.dataclass import Config, GenerateParams
    from src.arguments.deepspeed import Config, GenerateParams
    from src.utils import container_to, default, extract_args, load_dataclass
    from src.utils.experiment import dump_config, load_config

    # Argument parsing
    parser = argparse.ArgumentParser(description="DeepSpeed Training Script")
    parser.add_argument(
        "--name", required=True, type=str, help="Name of the config presets"
    )
    parser.add_argument("--local_rank", type=int, required=True, help="Local rank")
    parser.add_argument(
        "--training_dir",
        required=True,
        type=str,
        help="Path to the train output directory",
    )
    parser.add_argument(
        "--ckpt_path",
        required=False,
        default=None,
        type=str,
        help="Path to the checkpoint file (relative to training_dir/checkpoint)",
    )

    GenerateParams().load(parser, sentinel=True)
    args = parser.parse_args()

    name = args.name
    local_rank = args.local_rank
    infer_config, _, *assets = load_config(
        GenerateParams,
        name=name,
        cli_args=args,
        auto_create=True,
        prefix="inference",
        return_config_path=True,
    )
    if infer_config is None:
        print("Configuration created")
        sys.exit()

    output_dir, checkpoint_dir, log_dir, infer_config_path = assets
    print(f"Output dir: {output_dir.relative_to(Path.cwd())}")
    print(f"Checkpoint dir: {checkpoint_dir.relative_to(Path.cwd())}")
    print(f"Log dir: {log_dir.relative_to(Path.cwd())}")

    ## Load train config and merge with inference config
    training_dir = Path(args.training_dir)
    assert training_dir.exists(), f"Train output dir not found at {training_dir}"

    train_config_path = training_dir / "config.yaml"
    assert train_config_path.exists(), f"Train config not found at {train_config_path}"
    infer_config.training_dir = str(training_dir)

    # Check if checkpoint exists
    has_ckpt = False
    if args.ckpt_path is not None:
        model_path: Path = training_dir / "checkpoint" / args.ckpt_path
        assert model_path.exists(), f"Checkpoint not found at {model_path}"
        infer_config.ckpt_path = str(model_path)
        has_ckpt = True

    train_config = load_dataclass(Config, train_config_path, strict=False)
    train_config: Config = extract_args(train_config, args)
    infer_config.model_config = train_config

    dump_config(infer_config, output_dir / "config.yaml")
    dump_config(infer_config, infer_config_path)

    ic = infer_config
    dsp = infer_config.model_config.deepspeed

    # Get DeepSpeed metadata
    import deepspeed
    from torch import distributed as dist

    name = args.name
    local_rank = args.local_rank

    dist.init_process_group()
    world_size = dist.get_world_size()

    accelerator = deepspeed.get_accelerator()
    device = f"{accelerator.device_name()}:{local_rank}"

    import json
    import re

    import torch
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader, DistributedSampler
    from tqdm import tqdm
    from transformers import GenerationConfig
    from transformers.utils import logging

    from src.models.llava import LlavaPEFT, collate_fn
    from src.utils.dataset import DiscDataset
    from src.utils.log import PerformanceMonitor, Timer, pretty_print, print_once

    logging.set_verbosity_error()
    mp = train_config.model

    seed_everything(ic.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train_config.liger_kernel:
        apply_liger_kernel_to_llama()
    ### DeepSpeed Compatibility ###
    __USE_DEEPSPEED__ = True
    use_cache = not (mp.gradient_checkpointing and not __USE_DEEPSPEED__)

    model = LlavaPEFT(
        model_params=mp,
        gradient_checkpointing=not use_cache,
        lora_config=mp.lora_config,
        # device=device,
        torch_dtype=torch.bfloat16,
    )
    # TODO: Use zero3's reduction instead
    if "zero_optimization" in dsp.config:
        dsp.config.pop("zero_optimization")

    # Check if checkpoint exists
    if has_ckpt:
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
    else:
        print("Not using checkpoint. Using default model.")

    transform = model.transform
    processor = model.processor

    for param in model.parameters():
        param.requires_grad = False
    # model.to(device, torch.bfloat16)

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
        skip_no_object_info=True,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
    )
    inference_loader = DataLoader(
        dataset,
        batch_size=ic.batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=sampler,
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

    def initialize_model(model) -> deepspeed.InferenceEngine:
        model_engine = deepspeed.init_inference(
            model=model,
            tensor_parallel={"tp_size": world_size},
            dtype=torch.bfloat16,
            max_tokens=ic.max_tokens,
            replace_with_kernel_inject=True,
        )
        model_engine: deepspeed.InferenceEngine
        return model_engine

    model.finetune_language(train_config.finetune_language)
    model = model.merge_and_unload(inplace=True)

    model_engine = initialize_model(model)
    model_engine.eval()

    for ids, batch in tqdm(inference_loader):
        with DEBUG:
            target_dtypes = [torch.float16, torch.float32, torch.float64]
            inputs = container_to(
                batch, target_dtypes, device=device, dtype=torch.bfloat16
            )
            # `input_ids` and `attention_mask` should be long tensors
            inputs["input_ids"] = inputs["input_ids"].to(device, torch.long)
            inputs["attention_mask"] = inputs["attention_mask"].to(device, torch.long)
            # print(
            #     {k: v.dtype for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            # )

            DEBUG.stamp()
            DEBUG.set_params(**{"ids": ids})

            with torch.no_grad():
                output = model_engine.generate(
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
                print_once(assistant_reply)
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