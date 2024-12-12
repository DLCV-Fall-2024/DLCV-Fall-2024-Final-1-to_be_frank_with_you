import os
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class ModelParams:
    device: str = "cuda"

    model_id: str = "llava-hf/llava-1.5-7b-hf"

    change_encoder: bool = False
    encoder_id: str = "facebook/dinov2-large"

    share_vit: bool = False
    use_depth: bool = False
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    use_segmentation: bool = False
    segmentation_model_id: str = "shi-labs/oneformer_ade20k_dinat_large"

    fuser_id: str = "gemini"

    patch_size: int = 14
    vision_feature_select_strategy: str = "full"  # "default" or "full"
    gradient_checkpointing: bool = True
    lora_config: dict = field(
        default_factory=lambda: {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": [
                "q_proj",
                "v_proj",
                "multi_modal_projector.linear_1",
                "multi_modal_projector.linear_2",
            ],
            "exclude_modules": "vision_tower.*",
            "lora_dropout": 0.1,
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
    epochs: int = 1
    lr: float = 3e-5
    batch_size: int = 4
    optimizer_type: str = "default"
    accumulation_steps: int = 4


@dataclass
class Config:
    project_name: Optional[str] = None
    run_name: Optional[str] = None

    seed: int = 42
    local_rank: Union[int, List[int]] = os.getenv("LOCAL_RANK", -1)
    liger_kernel: bool = True

    model: ModelParams = field(default_factory=ModelParams)
    pipeline: PipelineParams = field(default_factory=PipelineParams)
    dataset: DatasetParams = field(default_factory=DatasetParams)
    optimization: OptimizationParams = field(default_factory=OptimizationParams)
