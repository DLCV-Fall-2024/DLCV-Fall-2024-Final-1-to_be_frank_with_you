import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, FullyShardedDataParallelPlugin
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
)

from src.arguments import (
    DatasetParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    YamlArgsLoader,
)
from utils.dataset import DiscDataset
from utils.log import PerformanceMonitor, Timer, init_logger, init_wandb, pretty_print

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Inference script for Traffic specific LLaVA."
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="runs",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        default="configs/train.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", -1))

    mp = ModelParams(parser)
    dp = DatasetParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    args = parser.parse_args()
    return args


def activate_only_lora(model):
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name


__USE_DEEPSPEED__ = True

from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments


def main():
    args = arg_parser()
    addition_config = {}

    ###############################
    yaml_file = Path(args.config_file)
    if yaml_file.exists():
        yaml_args = YamlArgsLoader(yaml_file)
        yaml_args.overwrite_args(args)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### Load model and apply PEFT configuration ####
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(**args.lora_config)
    model = get_peft_model(model, lora_config)
    activate_only_lora(model)
    model.to(device)

    processor: LlavaProcessor = LlavaProcessor.from_pretrained(args.model_id)
    processor.patch_size = args.patch_size
    processor.vision_feature_select_strategy = args.vision_feature_select_strategy

    transform = lambda img, prompt: processor(
        img,
        text=prompt,
        return_tensors="pt",  # return as pytorch tensors
        padding="max_length",
        max_length=1024,
        truncation=True,
        # padding=True,
        # do_rescale=False,  # since we already rescale color range to [0, 1] when loading dataset
    )
    processor_partial = lambda **kwargs: processor(
        **kwargs,
        return_tensors="pt",  # return as pytorch tensors
        padding="max_length",
        max_length=1024,
        truncation=True,
        # padding=True,
        # do_rescale=False,  # since we already rescale color range to [0, 1] when loading dataset
    )
    #################################################

    ### We don't transform the inputs in the dataset since we don't know the prompt size in advance (fix-sized padding introduces overhead)
    ### Instead, we will transform the inputs in the inference loop.
    dataset_dir = Path(args.dataset_path)
    train_set = dataset_dir / "train"
    val_set = dataset_dir / "val"
    assert (
        train_set.exists() and val_set.exists()
    ), f"Dataset not found. {dataset_dir} should contain 'train' and 'val' folders."
    trainer_input_kwargs = {
        "vision_feature_select_strategy": args.vision_feature_select_strategy,
        # "patch_size": args.patch_size,
        "use_cache": not args.gradient_checkpointing,
    }

    train_dataset = DiscDataset(
        train_set,
        transform=transform,
        train=True,
        use_trainer=True,
        trainer_input_kwargs=trainer_input_kwargs,
    )

    val_dataset = DiscDataset(
        val_set,
        transform=transform,
        train=True,
        use_trainer=True,
        trainer_input_kwargs=trainer_input_kwargs,
    )

    timestamp = time.strftime("%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / timestamp

    out_config_file = out_dir / "config.yaml"
    ckpt_dir = out_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    #### Load dataset ####
    from torch.optim.lr_scheduler import OneCycleLR, StepLR

    # Define Training Arguments
    training_args = TrainingArguments(
        run_name=timestamp,
        output_dir=out_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=out_dir / "logs",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=2,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        load_best_model_at_end=True,
        gradient_accumulation_steps=args.accumulation_steps,
        lr_scheduler_type="reduce_lr_on_plateau",
        save_only_model=True,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.num_workers,
        # use_liger_kernel=True,
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # tokenizer=processor,
        # processing_class=processor_partial,
        # compute_metrics=compute_metrics_function,  # Define metrics function
    )

    # Train and Evaluate
    trainer.train()
    trainer.save_model(args.output_dir)
    results = trainer.evaluate()
    print(results)


if __name__ == "__main__":
    main()
