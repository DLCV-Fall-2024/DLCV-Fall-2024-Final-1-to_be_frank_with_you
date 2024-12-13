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
class OptimizationParams:
    epochs: int = 1
    lr: float = 3e-5
    batch_size: int = 4
    optimizer_type: str = "default"
    accumulation_steps: int = 4


@dataclass
class DeepSpeedParams:
    wandb: bool = True
    # Not working
    log_level: str = "info"
    config: Dict = field(default_factory=lambda: {})
    # field(
    #     default_factory=lambda: {
    #         "train_micro_batch_size_per_gpu": 4,
    #         "gradient_accumulation_steps": 4,
    #         "bf16": {
    #             "enabled": True,
    #         },
    #         "flops_profiler": {
    #             "enabled": True,
    #             "detailed": True,
    #         },
    #         "zero_optimization": {
    #             "stage": 3,
    #             "offload_optimizer": {
    #                 "device": "cpu",
    #                 "pin_memory": True,
    #             },
    #             "offload_param": {
    #                 "device": "cpu",
    #                 "pin_memory": True,
    #             },
    #             "overlap_comm": True,
    #             "contiguous_gradients": True,
    #             "sub_group_size": 1e9,
    #             "reduce_bucket_size": "auto",
    #             "stage3_prefetch_bucket_size": "auto",
    #             "stage3_param_persistence_threshold": "auto",
    #             "stage3_max_live_parameters": 1e9,
    #             "stage3_max_reuse_distance": 1e9,
    #             "stage3_gather_16bit_weights_on_model_save": True,
    #         },
    #     }
    # )


@dataclass
class Config:
    project_name: Optional[str] = None
    run_name: Optional[str] = None

    seed: int = 42
    debug: bool = False
    profile: bool = False
    liger_kernel: bool = True

    model: ModelParams = field(default_factory=ModelParams)
    dataset: DatasetParams = field(default_factory=DatasetParams)
    optimization: OptimizationParams = field(default_factory=OptimizationParams)
    deepspeed: DeepSpeedParams = field(default_factory=DeepSpeedParams)
