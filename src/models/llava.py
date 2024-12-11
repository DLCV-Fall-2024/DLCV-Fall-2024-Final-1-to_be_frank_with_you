from typing import Any, Dict, List, Optional

import torch
from PIL.Image import Image
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    BaseImageProcessor,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from peft import LoraConfig, get_peft_model

from utils import default
from src.arguments.dataclass import ModelParams
from .encoder import VisionEncoder, DepthEncoder, SegmentationEncoder
from .fuser import FUSERS


# TODO: Not tested
class MergedImageProcessor(BaseImageProcessor):
    def __init__(self, processors: List[BaseImageProcessor]):
        super().__init__()

        self.processors = processors
        self.n_processors = len(processors)

    def preprocess(self, images, **kwargs):
        inputs = self.processors[0](images, **kwargs)
        image_shape = inputs["pixel_values"].shape

        auxiliary_pixel_values = []
        for processor in self.processors[1:]:
            pixel_values = processor(images, **kwargs)["pixel_values"]
            assert (
                pixel_values.shape == image_shape
            ), "All auxiliary encoders must output the same shape as the main encoder"
            auxiliary_pixel_values.append(pixel_values.unsqueeze(0))

        inputs["aux_pixel_values"] = (
            torch.cat(auxiliary_pixel_values, dim=0) if self.n_processors > 1 else None
        )
        return inputs


class VisionTower(torch.nn.Module):
    def __init__(self, model_params: ModelParams, vision_feature_layer: int):
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

        self.auxiliary_encoders = []
        if model_params.use_depth:
            depth_encoder = DepthEncoder(model_params.depth_model_id, vit, processor)
            self.auxiliary_encoders.append(depth_encoder)
            self.processors.append(depth_encoder.processor)
        if model_params.use_segmentation:
            segmentation_encoder = SegmentationEncoder(
                model_params.segmentation_model_id, vit, processor
            )
            self.auxiliary_encoders.append(segmentation_encoder)
            self.processors.append(segmentation_encoder.processor)

        self.n_auxiliary_features = len(self.auxiliary_encoders)
        self.fuser = FUSERS[model_params.fuser_id](
            n_auxiliary_features=self.n_auxiliary_features,
            d_model=self.encoder.model.config.hidden_size,
            # num_heads=self.encoder.model.config.num_attention_heads,
            # mlp_hidden_dim=self.encoder.model.config.mlp_hidden_dim,
        )

        # Create merged image processor
        self.processor = MergedImageProcessor(self.processors)

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        feature = self.encoder(pixel_values[0], **kwargs)
        image_feature = feature.hidden_states[self.vision_feature_layer]

        auxiliary_features = []
        if self.n_auxiliary_features > 0:
            for i, encoder in enumerate(self.auxiliary_encoders):
                feature = encoder(pixel_values[i + 1], **kwargs)
                auxiliary_features.append(
                    feature.hidden_states[self.vision_feature_layer]
                )
            out_feature = self.fuser(image_feature, auxiliary_features)
        else:
            out_feature = image_feature

        return BaseModelOutputWithPooling(
            # vision_feature_layer is -2 in Llava, so we need to add it to the output
            hidden_states=[out_feature, out_feature]
        )


class LlavaPEFT(torch.nn.Module):
    def __init__(
        self,
        model_params: ModelParams,
        gradient_checkpointing: bool,
        lora_config: Dict[str, Any],
    ):
        super().__init__()

        self.model_id = model_params.model_id
        llava = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        # Change the vision tower to the encoder
        vision_feature_layer = default(llava.config.vision_feature_layer, -2)
        self.vision_tower = VisionTower(model_params, vision_feature_layer)
        llava.vision_tower = self.vision_tower

        # Update vision related config
        encoder_config = self.vision_tower.encoder.model.config
        llava.config.vision_config = encoder_config

        image_size = self.vision_tower.encoder.processor.crop_size
        patch_size = self.vision_tower.encoder.model.config.patch_size
        image_seq_length = (image_size["height"] // patch_size) * (image_size["width"] // patch_size)
        llava.config.image_seq_length = image_seq_length

        # Remove projector layers from lora for direct finetuning
        no_lora_but_FF_prefix = ["multi_modal_projector", "fuser"]
        model_params.lora_config["target_modules"] = [
            layer
            for layer in model_params.lora_config["target_modules"]
            if not any([prefix in layer for prefix in no_lora_but_FF_prefix])
        ]

        # Activate gradient checkpointing
        if gradient_checkpointing:
            llava.enable_input_require_grads()
            llava.gradient_checkpointing_enable({"use_reentrant": True})

        # Apply LoRA
        lora_config = LoraConfig(**lora_config)
        self.llava = get_peft_model(llava, lora_config)
        self.activate_only_lora()

        # Activate finetuning the encoder
        for name, param in llava.named_parameters():
            if any([prefix in name for prefix in no_lora_but_FF_prefix]):
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
        inputs["aux_pixel_values"] = image_inputs["aux_pixel_values"]
        # inputs["aux_pixel_values"] = None

        return inputs

    def forward(
        self, pixel_values: torch.Tensor, aux_pixel_values: Optional[torch.Tensor], **inputs
    ):
        # WARNING: Not sure if this concat is fine
        if aux_pixel_values is not None:
            merged_pixel_values = torch.cat(
                [pixel_values.unsqueeze(0), aux_pixel_values], dim=0
            )
            inputs["pixel_values"] = merged_pixel_values
        else:
            inputs["pixel_values"] = pixel_values.unsqueeze(0)

        return self.llava(**inputs)

