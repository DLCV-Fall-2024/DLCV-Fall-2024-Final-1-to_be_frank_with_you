import os
from dataclasses import dataclass, field
from typing import Optional, Any, Dict


@dataclass
class ModelParams:
    model_id: str = "llava-hf/llava-1.5-7b-hf"

    change_encoder: bool = False
    encoder_id: str = "facebook/dinov2-large"

    share_vit: bool = False
    use_depth: bool = False
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    use_segmentation: bool = False
    segmentation_model_id: str = "shi-labs/oneformer_ade20k_dinat_large"

    fuser_id: str = "gemini"

    # patch_size: int = 14
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
class DatasetParams:
    dataset_path: str = "data"
    seed: int = 42
    num_workers: int = 10
    prefetch_factor: int = 2


@dataclass
class DeepSpeedParams:
    epochs: int = 1
    learning_rate: float = 3e-5
    batch_size: int = 4
    accumulation_steps: int = 4
    config: Dict = field(default_factory=lambda: {})


@dataclass
class Config:
    seed: int = 42
    debug: bool = False
    profile: bool = False
    liger_kernel: bool = True

    wandb: bool = True
    project_name: Optional[str] = None
    run_name: Optional[str] = None

    model: ModelParams = field(default_factory=ModelParams)
    dataset: DatasetParams = field(default_factory=DatasetParams)
    deepspeed: DeepSpeedParams = field(default_factory=DeepSpeedParams)
