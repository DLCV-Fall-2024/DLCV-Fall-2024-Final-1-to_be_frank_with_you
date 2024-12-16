import os
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ModelParams:
    device: str = "cuda"

    model_id: str = "llava-hf/llava-1.5-7b-hf"
    encoder_id: str = "facebook/dinov2-large"

    share_vit: bool = False
    use_depth: bool = True
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    use_segmentation: bool = True
    segmentation_model_id: str = "shi-labs/oneformer_ade20k_dinat_large"

    fuser_id: str = "gemini"
    conditional_fuser: bool = True
    condition_dropout: float = 0.3
    no_lora_but_FF_prefix: List[str] = field(
        default_factory=lambda: [
            "multi_modal_projector",
            "fuser",
            "vision_tower.AdaLNZero",
            "auxiliary_projectors",
        ]
    )
    patch_size: int = 14
    vision_feature_select_strategy: str = "full"  # "default" or "full"
    gradient_checkpointing: bool = True
    lora_config: dict = field(
        default_factory=lambda: {
            "r": 256,
            "lora_alpha": 16,
            "use_rslora": True,  # sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`
            "target_modules": [
                "q_proj",
                "v_proj",
                "multi_modal_projector.linear_1",
                "multi_modal_projector.linear_2",
            ],
            "exclude_modules": "vision_tower.*",
            "lora_dropout": 0.3,
            "use_dora": True,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
    )


@dataclass
class PipelineParams:
    wandb: bool = True
    debug: bool = False


@dataclass
class DatasetParams:
    dataset_path: str = "data"
    seed: int = 42
    num_workers: int = 10
    prefetch_factor: int = 2


@dataclass
class OptimizationParams:
    epochs: int = 2
    lr: float = 3e-4
    batch_size: int = 4
    optimizer_type: str = "default"
    accumulation_steps: int = 4
    gradient_clip_val: float = 1.0

    train_language_start_epoch: int = 1


@dataclass
class Config:
    project_name: Optional[str] = None
    run_name: Optional[str] = None

    seed: int = 42
    local_rank: Any = os.getenv("LOCAL_RANK", -1)
    liger_kernel: bool = True

    model: ModelParams = field(default_factory=ModelParams)
    pipeline: PipelineParams = field(default_factory=PipelineParams)
    dataset: DatasetParams = field(default_factory=DatasetParams)
    optimization: OptimizationParams = field(default_factory=OptimizationParams)


@dataclass
class GenerateParams:
    config_path: Optional[str] = None
    output_dir: Optional[str] = None
    dataset_path: str = "data/test"

    seed: int = 42
    batch_size: int = 4
    num_workers: int = 10
    max_new_tokens: int = 1024
    generation_config: dict = field(
        default_factory=lambda: {
            "do_sample": False,
        }
    )
    use_regex: bool = True

    model_config: Config = field(default_factory=Config)
    model_path: Optional[str] = None
