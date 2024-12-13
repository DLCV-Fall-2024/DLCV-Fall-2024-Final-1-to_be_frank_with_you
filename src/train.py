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


def main():
    args = arg_parser()
    addition_config = {}
    ### DeepSpeed Compatibility ###
    local_rank = -1
    try:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")  # NCCL is GPU-aware
        local_rank = torch.distributed.get_rank()
    except:
        __USE_DEEPSPEED__ = False

    ###############################
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

    ## enable gradient checkpointing for memory efficiency
    use_cache = True
    if getattr(args, "gradient_checkpointing", True) and not __USE_DEEPSPEED__:
        model.gradient_checkpointing_enable({"use_reentrant": True})
        model.enable_input_require_grads()
        use_cache = False

    ## finetune only language model
    lora_config = LoraConfig(**args.lora_config)
    model = get_peft_model(model, lora_config)
    # model.add_adapter(lora_config)

    # enable gradient
    addition_config["model_struct"] = str(model)
    activate_only_lora(model)

    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(
        f"Trainable: {trainable_params/1e6:.4f}M | All: {all_param/1e6:.4f}M | Ratio: {trainable_params/all_param * 100:.3f}%"
    )
    model.to(device)

    processor: LlavaProcessor = LlavaProcessor.from_pretrained(args.model_id)
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
    dataset_dir = Path(args.dataset_path)
    train_set = dataset_dir / "train"
    val_set = dataset_dir / "val"
    assert (
        train_set.exists() and val_set.exists()
    ), f"Dataset not found. {dataset_dir} should contain 'train' and 'val' folders."
    train_dataset = DiscDataset(train_set, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        prefetch_factor=args.prefetch_factor,
        num_workers=args.num_workers,
        shuffle=True,
    )

    val_dataset = DiscDataset(val_set, train=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    timestamp = time.strftime("%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / timestamp

    out_config_file = out_dir / "config.yaml"
    ckpt_dir = out_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ###################### Optimization ######################
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR, StepLR

    training_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(training_params, lr=args.lr)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs
    )
    optimizer.zero_grad()

    accum_steps = getattr(args, "accumulation_steps", 1)
    print(f"Effective batch size: {args.batch_size * accum_steps}")
    print(f"Using {device} device")
    if __USE_DEEPSPEED__:
        dsp_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=16,
            gradient_clipping=1.0,
            zero_stage=3,
        )
        fsdp_plugin = None
        if getattr(args, "gradient_checkpointing", True):
            fsdp_plugin = FullyShardedDataParallelPlugin(activation_checkpointing=True)
            use_cache = False
        accelerator = Accelerator(deepspeed_plugin=dsp_plugin, fsdp_plugin=fsdp_plugin)

        model, optimizer, train_loader, val_loader, scheduler, processor = (
            accelerator.prepare(
                model, optimizer, train_loader, val_loader, scheduler, processor
            )
        )
    ##########################################################

    project_name = "DLCV-FINAL-Traffic-LLaVA"
    if hasattr(args, "project_name"):
        project_name = args.project_name

    run_name = f"{args.model_id.split('/')[-1]}-{timestamp}"
    if hasattr(args, "run_name"):
        run_name = args.run_name
    init_wandb(
        project_name,
        run_name,
        config=vars(args),
        log_dir=out_dir,
        local_rank=local_rank if __USE_DEEPSPEED__ else None,
    )
    logger = init_logger(local_rank=local_rank if __USE_DEEPSPEED__ else None)
    if yaml_file:
        yaml_args = YamlArgsLoader(out_config_file)
        yaml_args.save_args(
            args, exclude=["config_file", "output_dir"], additional=addition_config
        )

    del addition_config
    epochs = args.epochs
    timer = Timer(10)  # 10 minutes
    DEBUG = PerformanceMonitor(args.debug)
    global_step = 0
    for name, param in model.named_parameters():
        if not (param.requires_grad == ("lora" in name)):
            print(f"Warning: {name}.required_grad= {param.requires_grad}.")

    for epoch in range(epochs):
        model.train()

        accum_loss = 0
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
                        torch.bfloat16,
                        torch.bfloat16,
                    ]:
                        v.requires_grad = True
                labels = inputs["input_ids"].clone()

                DEBUG.stamp()
                DEBUG.set_params(**{"labels": labels})
                out = model.forward(
                    **inputs,
                    labels=labels,
                    vision_feature_select_strategy=args.vision_feature_select_strategy,
                    use_cache=use_cache,
                )
                loss = out.loss  # / accum_steps
                loss.backward()
                if args.debug:
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
        print("Saving checkpoint...")
        ckpt_path = ckpt_dir / f"{epoch}.pt"
        torch.save(get_peft_model_state_dict(model), ckpt_path)

        model.eval()
        val_bar = tqdm(val_loader)
        val_bar.set_description(f"[Val {epoch}/{epochs}]")
        accum_loss = 0
        with torch.no_grad():
            for ids, batch in val_bar:
                with DEBUG:
                    inputs = transform(batch["image"], prompt=batch["prompt"]).to(
                        device
                    )
                    labels = inputs["input_ids"].clone()
                    DEBUG.stamp()
                    DEBUG.set_params(**{"labels": labels})

                    out = model.forward(
                        **inputs,
                        labels=labels,
                        vision_feature_select_strategy=args.vision_feature_select_strategy,
                    )
                    loss = out.loss
                    accum_loss += loss.item()
                    logger({"val/loss": loss.item()})

                    DEBUG.stamp()
                    DEBUG.log_performance(log_per=20)
        print(f"Epoch {epoch}: Val Loss: {accum_loss/len(val_loader):.4f}")


if __name__ == "__main__":
    main()
