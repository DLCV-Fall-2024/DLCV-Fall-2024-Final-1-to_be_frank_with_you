from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

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
        cache_dir: Path = Path(".cache"),
    ):
        super().__init__()

        self.encoder = VisionEncoder(model_params.encoder_id)
        self.processors = [self.encoder.processor]
        self.vision_feature_layer = vision_feature_layer

        self.cache_postfix: List[str] = ["rgb"]  ## The key defined in dataset.py

        vit = None
        processor = None
        if model_params.share_vit:
            # Use the same vit and processor for all auxiliary encoders
            vit = self.encoder.model
            processor = self.encoder.processor

        self.auxiliary_encoders = torch.nn.ModuleList()
        self.auxiliary_projectors = torch.nn.ModuleList()
        self.use_depth = model_params.use_depth
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
            self.cache_postfix.append("depth")

        self.use_segmentation = model_params.use_segmentation
        if model_params.use_segmentation:
            segmentation_encoder = SegmentationEncoder(
                model_params.segmentation_model_id,
                model=vit,
                processor=processor,
                segment_type=segment_type,
                image_target_size=(800, 1200),
                vision_feature_layer=vision_feature_layer,
                device=device,
                torch_dtype=torch_dtype,
            )
            self.auxiliary_encoders.append(segmentation_encoder)
            # No projection for segmentation encoder
            self.auxiliary_projectors.append(nn.Identity())
            # self.processors.append(
            #     partial(segmentation_encoder.processor, task_inputs=[segment_type])
            # )
            self.processors.append(segmentation_encoder.task_processor)
            # self.processors.append(segmentation_encoder.resized_processor)
            self.cache_postfix.append("seg")

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
        self.cache_dir = cache_dir

    def forward(self, pixel_values: Union[torch.Tensor, Dict], **kwargs):
        language_embeds = None
        if self.conditional_fuser:
            pixel_values, language_embeds = pixel_values

        image_feature, auxiliary_features = self._get_feature(pixel_values, **kwargs)

        out_feature = self._maybe_fuse(
            image_feature, auxiliary_features, language_embeds
        )
        return BaseModelOutputWithPooling(
            # vision_feature_layer is -2 in Llava, so we need to add placeholder for the output
            hidden_states=[out_feature, None]
        )

    def _get_feature(self, pixel_values: torch.Tensor, **kwargs):
        feature: ImageEncoderOutput = self.encoder(pixel_values[0], **kwargs)
        image_feature = feature.hidden_states[self.vision_feature_layer]
        main_dim = tuple(image_feature.shape)

        auxiliary_features = self._maybe_get_auxiliary_features(
            pixel_values,
            main_dim,
            device=image_feature.device,
            dtype=image_feature.dtype,
            **kwargs,
        )

        return image_feature, auxiliary_features

    def _maybe_fuse(
        self,
        image_feature: torch.Tensor,
        auxiliary_features: Optional[List[torch.Tensor]] = None,
        language_embeds: Optional[torch.Tensor] = None,
    ):
        if self.n_auxiliary_features == 0:
            return image_feature

        fused_features = self.fuser(image_feature, auxiliary_features)
        if self.conditional_fuser:
            fused_features = self._apply_conditional_fuser(
                fused_features, language_embeds
            )
        # Residual connection
        return image_feature + fused_features

    def _maybe_get_auxiliary_features(
        self,
        pixel_values: torch.Tensor,
        main_dim: Tuple[int, int, int],
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        if self.n_auxiliary_features == 0:
            return None

        auxiliary_features = []
        for i, encoder in enumerate(self.auxiliary_encoders):
            feature: ImageEncoderOutput = encoder(**pixel_values[i + 1], **kwargs)

            if feature.use_pred:
                preprocess = self.processors[0](
                    feature.predictions.to(dtype=torch.float32),
                    return_tensors="pt",  # return as pytorch tensors
                )
                preprocess = preprocess.to(device=device, dtype=dtype)
                encoded: ImageEncoderOutput = self.encoder(
                    preprocess["pixel_values"], **kwargs
                )
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
        return auxiliary_features

    def _apply_conditional_fuser(self, fused_features, language_embeds):
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

        return adjust_feature
