from dataclasses import dataclass, field
from typing import Optional, Dict, List
from argparse import ArgumentParser

from src.arguments import ParamGroup


@dataclass
class ModelParams(ParamGroup):
    model_id: str = "llava-hf/llava-1.5-7b-hf"
    encoder_id: str = "facebook/dinov2-large"

    use_clip: bool = False
    interpolation_mode: str = "bilinear"

    share_vit: bool = False
    use_processed: bool = True

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
            "vision_tower.AdaLNZeroCLIP",
            "auxiliary_projectors",
        ]
    )

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

    def load(self, parser: ArgumentParser, sentinel: bool = False):
        super().__init__(parser=parser, name="Loading Parameters", fill_none=sentinel)

    def extract(self, args):
        return super().extract(args)


@dataclass
class DatasetParams(ParamGroup):
    dataset_path: str = "data"
    num_workers: int = 20
    prefetch_factor: int = 40

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

    finetune_language: bool = True
    resume: Optional[str] = None
    resume_tag: Optional[str] = None

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


@dataclass
class GenerateParams(ParamGroup):
    config_path: Optional[str] = None
    output_dir: Optional[str] = None
    dataset_path: str = "data/test"

    seed: int = 42
    batch_size: int = 4
    num_workers: int = 10
    prefetch_factor: int = 20
    max_new_tokens: int = 1024
    generation_config: dict = field(
        default_factory=lambda: {
            "do_sample": False,
        }
    )
    use_regex: bool = True

    model_path: Optional[str] = None
    model_config: Config = field(default_factory=Config)

    def load(self, parser: ArgumentParser, sentinel: bool = False):
        super().__init__(
            parser=parser, name="Generation Parameters", fill_none=sentinel
        )
        self.model_config.load(parser, sentinel)

    def extract(self, args):
        p = super().extract(args)
        p["model_config"] = self.model_config.extract(args)
        return p
