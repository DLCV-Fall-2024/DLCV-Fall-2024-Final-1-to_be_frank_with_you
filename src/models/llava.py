from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, PeftMixedModel, PeftModel, get_peft_model
from PIL.Image import Image
from transformers import (
    BaseImageProcessor,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

# try:
#     from deepspeed.runtime.zero.stage3 import (
#         estimate_zero3_model_states_mem_needs_all_live,
#     )
# except ImportError:
#     print("DeepSpeed not installed")
#     pass

from pathlib import Path

from src.arguments.dataclass import ModelParams
from src.utils import default

from .encoder import (
    DepthEncoder,
    ImageEncoderOutput,
    SegmentationEncoder,
    VisionEncoder,
)
from .fuser import FUSERS
from .utils import AdaLNZero, ensure_all_on_device, ensure_all_same_dtype
from .vision_tower import MergedImageProcessor, VisionTower


class LlavaPEFT(torch.nn.Module):
    def __init__(
        self,
        model_params: ModelParams,
        gradient_checkpointing: bool,
        lora_config: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super().__init__()

        self.model_id = model_params.model_id
        llava = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        )

        # Change the vision tower to the encoder
        vision_feature_layer = default(llava.config.vision_feature_layer, -2)
        self.conditional_fuser = model_params.conditional_fuser
        self.condition_dropout = model_params.condition_dropout
        language_embeds_dim = llava.get_input_embeddings().weight.shape[1]

        self.vision_tower = VisionTower(
            model_params,
            vision_feature_layer,
            language_embeds_dim=language_embeds_dim,
            torch_dtype=torch_dtype,
            device=device,
        )
        llava.vision_tower = self.vision_tower

        # Update vision related config
        encoder_config = self.vision_tower.encoder.model.config
        llava.config.vision_config = encoder_config

        image_size = self.vision_tower.encoder.processor.crop_size
        patch_size = self.vision_tower.encoder.model.config.patch_size
        image_seq_length = (image_size["height"] // patch_size) * (
            image_size["width"] // patch_size
        )
        llava.config.image_seq_length = image_seq_length

        # Remove projector layers from lora for direct finetuning
        self.no_lora_but_FF_prefix = getattr(
            model_params,
            "no_lora_but_FF_prefix",
            [
                "multi_modal_projector",
                "fuser",
                "vision_tower.AdaLNZero",
                "auxiliary_projectors",
            ],
        )
        model_params.lora_config["target_modules"] = [
            layer
            for layer in model_params.lora_config["target_modules"]
            if not any([prefix in layer for prefix in self.no_lora_but_FF_prefix])
        ]

        # These should be apply before apply PEFT
        ensure_all_same_dtype(llava, torch_dtype)
        ensure_all_on_device(llava, device)

        # Activate gradient checkpointing
        if gradient_checkpointing:
            llava.enable_input_require_grads()
            llava.gradient_checkpointing_enable({"use_reentrant": True})

        # Apply LoRA
        lora_config = LoraConfig(**lora_config)
        self.llava: PeftMixedModel = get_peft_model(llava, lora_config)
        self.activate_only_lora()

        # Estimate memory for zero3
        # try:
        #     estimate_zero3_model_states_mem_needs_all_live(self.llava)
        # except:
        #     pass

        # Activate finetuning the encoder
        for name, param in llava.named_parameters():
            if any([prefix in name for prefix in self.no_lora_but_FF_prefix]):
                param.requires_grad = True

        processor: LlavaProcessor = LlavaProcessor.from_pretrained(
            model_params.model_id
        )
        processor.image_processor = self.vision_tower.encoder.processor
        processor.patch_size = model_params.patch_size
        processor.vision_feature_select_strategy = (
            model_params.vision_feature_select_strategy
        )
        self.processor = processor

    def get_model_struct(self):
        return str(self.llava)

    def activate_only_lora(self):
        for name, param in self.llava.named_parameters():
            param.requires_grad = "lora" in name

    def transform(self, img: Image, prompt: str):
        inputs = self.processor(
            img,
            text=prompt,
            return_tensors="pt",  # return as pytorch tensors
            padding=True,
            do_rescale=False,  # since we already rescale color range to [0, 1] when loading dataset
        )

        image_inputs = self.vision_tower.processor(
            img,
            return_tensors="pt",  # return as pytorch tensors
            padding=True,
            do_rescale=False,  # since we already rescale color range to [0, 1] when loading dataset)
        )

        inputs["pixel_values"] = image_inputs["pixel_values"]
        inputs["aux_inputs"] = image_inputs["aux_inputs"]

        return inputs

    def forward(
        self,
        pixel_values: torch.Tensor,
        aux_inputs: Optional[List[torch.Tensor]] = None,
        **inputs,
    ):
        if aux_inputs is not None:
            inputs["pixel_values"] = [pixel_values, *aux_inputs]
        else:
            inputs["pixel_values"] = [pixel_values]

        #  TODO: Can delete this part
        # ## Maintain Backward compatibility and use cache feature ##
        # if "other_params" in inputs and "use_cache_feat" in inputs["other_params"]:
        #     use_cache_feat = inputs["other_params"]["use_cache_feat"]
        #     use_cache_feat = all(use_cache_feat)
        #     use_cache_feat = use_cache_feat and all(
        #         [x in inputs["other_params"] for x in ["rgb", "depth", "seg"]]
        #     )
        # image_feature_dict = {
        #     "id": inputs["id"],
        #     "pixel_values": inputs["pixel_values"],
        # }
        # image_feature_dict["feature"] = None
        # if use_cache_feat:
        #     image_feature_dict["feature"] = (
        #         inputs["other_params"]["rgb"],
        #         [
        #             inputs["other_params"]["depth"],
        #             inputs["other_params"]["seg"],
        #         ],
        #     )

        # inputs[pixel_values] = image_feature_dict
        ###########################################################

        if self.conditional_fuser:
            language_embeds = self.llava.base_model.get_input_embeddings()(
                inputs["input_ids"]
            )

            language_embeds = language_embeds.mean(dim=1)
            inputs["pixel_values"] = (inputs["pixel_values"], language_embeds)

        if "other_params" in inputs:
            del inputs["other_params"]
        return self.llava(**inputs)

    def merge_adapter(self):
        self.llava.merge_adapter()

    def merge_and_unload(self, inplace: bool = False):
        if inplace:
            self.llava = self.llava.merge_and_unload()
            return self
        return self.llava.merge_and_unload()

    def unmerge_adapter(self):
        self.llava.unmerge_adapter()

    def enssetial_state_dict(self):
        esstial_keywords = self.no_lora_but_FF_prefix
        output_state_dict = {}
        for key, value in self.llava.state_dict().items():
            if any([keyword in key for keyword in esstial_keywords]):
                output_state_dict[key] = value
        return output_state_dict

    def load_state_dict(self, state_dict, strict=False, assign=False):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("llava"):
                new_state_dict[key[6:]] = value
            else:
                new_state_dict[key] = value
        self.llava.load_state_dict(new_state_dict, strict=strict, assign=assign)

    def generate(
        self,
        pixel_values: torch.Tensor,
        aux_inputs: Optional[List[torch.Tensor]] = None,
        **inputs,
    ):
        if aux_inputs is not None:
            inputs["pixel_values"] = [pixel_values, *aux_inputs]
        else:
            inputs["pixel_values"] = [pixel_values]

        if self.conditional_fuser:
            language_embeds = self.llava.base_model.get_input_embeddings()(
                inputs["input_ids"]
            )

            language_embeds = language_embeds.mean(dim=1)
            inputs["pixel_values"] = (inputs["pixel_values"], language_embeds)
        return self.llava.generate(**inputs)

    def finetune_language(self, finetune: bool = True):
        for name, param in self.llava.named_parameters():
            if "lora" in name:
                param.requires_grad = finetune


if __name__ == "__main__":
    from pathlib import Path

    from src.arguments.dataclass import Config
    from src.utils.experiment import load_config

    name = "DINO_r32_rgbd"
    base_dir = Path("outputs/DINO_r32_rgbd_1214_164435")
    config, timestamp, output_dir, checkpoint_dir, log_dir = load_config(
        name, Config, auto_create=True
    )

    import torch
    from transformers.utils import logging

    logging.set_verbosity_error()

    from src.models.llava import LlavaPEFT
    from src.models.utils import ensure_all_on_device, ensure_all_same_dtype
    from src.utils import default

    addition_config = {}
    mp = config.model
    dp = config.dataset
    pp = config.pipeline
    op = config.optimization

    device = torch.device("cpu")

    __USE_DEEPSPEED__ = False
    local_rank = -1
    use_cache = not (mp.gradient_checkpointing and not __USE_DEEPSPEED__)

    model = LlavaPEFT(
        model_params=mp,
        gradient_checkpointing=not use_cache,
        lora_config=mp.lora_config,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    model.load_state_dict(
        torch.load(
            base_dir / "checkpoint" / "latest.pt",
            map_location=device,
            weights_only=True,
        )
    )

    torch.save(
        model.enssetial_state_dict(),
        base_dir / "checkpoint" / "shrink.pt",
    )

    ## Estimate memory saved

    # original pt file memory
    print("Original pt file memory")
    original_memory = (base_dir / "checkpoint" / "latest.pt").stat().st_size
    shirnk_memory = (base_dir / "checkpoint" / "shrink.pt").stat().st_size
    print(f"Original memory: {original_memory/ 2 ** 20} MB")
    print(f"Shrink memory: {shirnk_memory/2 ** 20} MB")
    print(f"Saved memory: {(original_memory - shirnk_memory)/2 ** 20} MB")
