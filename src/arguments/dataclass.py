from typing import Optional

import os
from dataclasses import dataclass, field


@dataclass
class ModelParams:
    model_id: str = "llava-hf/llava-1.5-7b-hf"
    device: str = "cuda"
    patch_size: int = 14
    vision_feature_select_strategy: str = "full"  # "default" or "full"
    gradient_checkpointing: bool = False
    lora_config: dict = field(
        default_factory=lambda: {
            "r": 4,
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
    debug: bool = False


@dataclass
class DatasetParams:
    dataset_path: str = "data/test"
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 10
    prefetch_factor: int = 2


@dataclass
class OptimizationParams:
    epochs: int = 100
    lr: float = 3e-5
    optimizer_type: str = "default"
    accumulation_steps: int = 4


@dataclass
class Config:
    project_name: Optional[str] = None
    run_name: Optional[str] = None

    seed: int = 42
    lora_rank: int = os.getenv("LOCAL_RANK", -1)

    model: ModelParams = field(default_factory=ModelParams)
    pipeline: PipelineParams = field(default_factory=PipelineParams)
    dataset: DatasetParams = field(default_factory=DatasetParams)
    optimization: OptimizationParams = field(default_factory=OptimizationParams)