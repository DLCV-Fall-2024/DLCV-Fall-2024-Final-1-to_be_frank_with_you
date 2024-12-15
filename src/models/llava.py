from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

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

try:
    from deepspeed.runtime.zero.stage3 import (
        estimate_zero3_model_states_mem_needs_all_live,
    )
except ImportError:
    print("DeepSpeed not installed")
    pass

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


# TODO: Not tested
class MergedImageProcessor(BaseImageProcessor):

    def __init__(
        self,
        processors: List[BaseImageProcessor],
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.processors = processors
        self.n_processors = len(processors)

        self.device = device
        self.torch_dtype = torch_dtype

    def preprocess(self, images, **kwargs):
        if "padding" in kwargs:
            del kwargs["padding"]
        inputs = self.processors[0](images, **kwargs)
        image_shape = inputs["pixel_values"].shape
        inputs["pixel_values"] = inputs["pixel_values"].to(
            device=self.device, dtype=self.torch_dtype
        )
        auxiliary_inputs = []
        for processor in self.processors[1:]:
            aux_inputs = processor(images, **kwargs)
            aux_inputs.to(device=self.device, dtype=self.torch_dtype)
            auxiliary_inputs.append(aux_inputs)

        inputs["aux_inputs"] = auxiliary_inputs

        return inputs


class VisionTower(torch.nn.Module):

    def __init__(
        self,
        model_params: ModelParams,
        vision_feature_layer: int,
        language_embeds_dim: int = 768,
        segment_type: str = "semantic",  # ["semantic", "instance", "panoptic"]
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.encoder = VisionEncoder(model_params.encoder_id)
        self.processors = [self.encoder.processor]
        self.vision_feature_layer = vision_feature_layer

        vit = None
        processor = None
        if model_params.share_vit:
            # Use the same vit and processor for all auxiliary encoders
            vit = self.encoder.model
            processor = self.encoder.processor

        self.auxiliary_encoders = torch.nn.ModuleList()
        self.auxiliary_projectors = torch.nn.ModuleList()
        if model_params.use_depth:
            depth_encoder = DepthEncoder(
                model_params.depth_model_id,
                vit,
                processor,
                vision_feature_layer,
                device=device,
                torch_dtype=torch_dtype,
            )
            self.auxiliary_encoders.append(depth_encoder)
            self.auxiliary_projectors.append(
                nn.Linear(
                    depth_encoder.hidden_states_dim,
                    self.encoder.hidden_states_dim,
                    device=device,
                    dtype=torch_dtype,
                )
            )
            self.processors.append(
                self.processors[0]
            )  # Use the main processor for depth encoder
            # self.processors.append(depth_encoder.processor)
        if model_params.use_segmentation:
            segmentation_encoder = SegmentationEncoder(
                model_params.segmentation_model_id,
                vit,
                processor,
                vision_feature_layer,
                device=device,
                torch_dtype=torch_dtype,
            )
            self.auxiliary_encoders.append(segmentation_encoder)

            self.auxiliary_projectors.append(
                nn.Linear(
                    depth_encoder.hidden_states_dim,
                    self.encoder.hidden_states_dim,
                    device=device,
                    dtype=torch_dtype,
                )
            )

            self.processors.append(
                partial(segmentation_encoder.processor, task_inputs=[segment_type])
            )

        self.n_auxiliary_features = len(self.auxiliary_encoders)
        self.fuser = FUSERS[model_params.fuser_id](
            n_auxiliary_features=self.n_auxiliary_features,
            d_model=self.encoder.model.config.hidden_size,
            conditional_fuser=model_params.conditional_fuser,
            # num_heads=self.encoder.model.config.num_attention_heads,
            # mlp_hidden_dim=self.encoder.model.config.mlp_hidden_dim,
        ).to(device=device, dtype=torch_dtype)

        self.conditional_fuser = model_params.conditional_fuser
        self.condition_dropout = model_params.condition_dropout
        if self.conditional_fuser:
            self.AdaLNZero = AdaLNZero(
                hidden_dim=self.encoder.model.config.hidden_size,
                condition_dim=language_embeds_dim,
            ).to(device=device, dtype=torch_dtype)

        # Create merged image processor
        self.processor = MergedImageProcessor(
            self.processors, torch_dtype=torch_dtype, device=device
        )

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        if self.conditional_fuser:
            pixel_values, language_embeds = pixel_values

        feature: ImageEncoderOutput = self.encoder(pixel_values[0], **kwargs)
        image_feature = feature.hidden_states[self.vision_feature_layer]
        main_dim = tuple(image_feature.shape)
        auxiliary_features = []
        if self.n_auxiliary_features > 0:
            for i, encoder in enumerate(self.auxiliary_encoders):
                feature: ImageEncoderOutput = encoder(**pixel_values[i + 1], **kwargs)

                if feature.use_pred:
                    preprocess = self.processors[0](
                        feature.predictions.to(dtype=torch.float32),
                        return_tensors="pt",  # return as pytorch tensors
                    )
                    preprocess = preprocess.to(
                        device=image_feature.device, dtype=image_feature.dtype
                    )
                    encoded = self.encoder(preprocess["pixel_values"], **kwargs)
                    wanted_feature = encoded.hidden_states[self.vision_feature_layer]
                else:
                    wanted_feature = feature.hidden_states[self.vision_feature_layer]
                    wanted_feature = self.auxiliary_projectors[i](wanted_feature)
                assert (
                    tuple(wanted_feature.shape) == main_dim
                ), "All auxiliary encoders must output the same shape as the main encoder, but got {} and {}".format(
                    wanted_feature.shape, main_dim
                )
                auxiliary_features.append(wanted_feature)

            fused_features = self.fuser(image_feature, auxiliary_features)
            adjust_feature = fused_features

            if self.conditional_fuser:
                concatenated_features = torch.cat(fused_features, dim=1)
                cond_adjust: torch.Tensor = self.AdaLNZero(
                    concatenated_features, language_embeds
                )

                adjust_feature = torch.stack(
                    cond_adjust.chunk(self.n_auxiliary_features, dim=1), dim=0
                ).mean(dim=0)

                if self.condition_dropout > 0:
                    mean_feature = torch.stack(fused_features, dim=0).mean(dim=0)

                    adjust_feature = nn.functional.dropout(
                        adjust_feature, p=self.condition_dropout, training=self.training
                    )
                    dropout_mask = adjust_feature == 0
                    adjust_feature[dropout_mask] = mean_feature[dropout_mask]
            # Residual connection
            out_feature = image_feature + adjust_feature
        else:
            out_feature = image_feature

        return BaseModelOutputWithPooling(
            # vision_feature_layer is -2 in Llava, so we need to add placeholder for the output
            hidden_states=[out_feature, None]
        )


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
        try:
            estimate_zero3_model_states_mem_needs_all_live(self.llava)
        except:
            pass

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

        if self.conditional_fuser:
            language_embeds = self.llava.base_model.get_input_embeddings()(
                inputs["input_ids"]
            )

            language_embeds = language_embeds.mean(dim=1)
            inputs["pixel_values"] = (inputs["pixel_values"], language_embeds)

        return self.llava(**inputs)

    def merge_adapter(self):
        self.llava.merge_adapter()

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


if __name__ == "__main__":
    from src.arguments.dataclass import Config
    from src.utils.experiment import load_config

    name = "DINO_r256_3encoder_big_lr"
    config, timestamp, output_dir, checkpoint_dir, log_dir = load_config(
        name, Config, auto_create=True
    )

    from pathlib import Path

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

    base_dir = Path("outputs/DINO_r256_3encoder_big_lr_1215_001147")
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
