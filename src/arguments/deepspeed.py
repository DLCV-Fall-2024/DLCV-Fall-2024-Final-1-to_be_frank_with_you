from dataclasses import dataclass, field
from typing import Optional, Dict
from argparse import ArgumentParser

from src.arguments import ParamGroup


@dataclass
class ModelParams(ParamGroup):
    model_id: str = "llava-hf/llava-1.5-7b-hf"
    encoder_id: str = "facebook/dinov2-large"

    share_vit: bool = False
    use_depth: bool = False
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    use_segmentation: bool = False
    segmentation_model_id: str = "shi-labs/oneformer_ade20k_dinat_large"

    fuser_id: str = "gemini"
    conditional_fuser: bool = True
    condition_dropout: float = 0.3

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

    def load(self, parser: ArgumentParser, sentinel: bool = False):
        super().__init__(parser=parser, name="Loading Parameters", fill_none=sentinel)

    def extract(self, args):
        return super().extract(args)


@dataclass
class DatasetParams(ParamGroup):
    dataset_path: str = "data"
    num_workers: int = 10
    prefetch_factor: int = 2

    def load(self, parser: ArgumentParser, sentinel: bool = False):
        super().__init__(parser=parser, name="Dataset Parameters", fill_none=sentinel)

    def extract(self, args):
        return super().extract(args)

@dataclass
class DeepSpeedParams(ParamGroup):
    epochs: int = 1
    learning_rate: float = 3e-5
    batch_size: int = 4
    accumulation_steps: int = 4
    config: Dict = field(default_factory=lambda: {})

    def load(self, parser: ArgumentParser, sentinel: bool = False):
        super().__init__(parser=parser, name="DeepSpeed Parameters", fill_none=sentinel)

    def extract(self, args):
        return super().extract(args)


@dataclass
class Config(ParamGroup):
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

    def load(self, parser: ArgumentParser, sentinel: bool = False):
        super().__init__(parser=parser, fill_none=sentinel)
        self.model.load(parser, sentinel)
        self.dataset.load(parser, sentinel)
        self.deepspeed.load(parser, sentinel)

    def extract(self, args):
        p = super().extract(args)
        p["model"] = self.model.extract(args)
        p["dataset"] = self.dataset.extract(args)
        p["deepspeed"] = self.deepspeed.extract(args)
        return p
