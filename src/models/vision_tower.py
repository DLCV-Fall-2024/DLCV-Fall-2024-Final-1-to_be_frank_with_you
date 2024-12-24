from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BaseImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from PIL.Image import Image
from pathlib import Path

from src.arguments.deepspeed import ModelParams

from .encoder import (
    DepthEncoder,
    ImageEncoderOutput,
    SegmentationEncoder,
    VisionEncoder,
)
from .fuser import FUSERS
from .utils import AdaLNZero


class MergedImageProcessor(BaseImageProcessor):

    def __init__(
        self,
        encoder_processor: BaseImageProcessor,
        auxiliary_processors: List[BaseImageProcessor],
        encoder_index: Dict[str, int],
        clip_processor: Optional[BaseImageProcessor] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.clip_processor = clip_processor
        self.encoder_processor = encoder_processor
        self.auxiliary_processors = auxiliary_processors
        self.encoder_index = encoder_index

        self.device = device
        self.torch_dtype = torch_dtype

    def preprocess(self, images, processed_images: Dict[str, Any], **kwargs):
        if "padding" in kwargs:
            del kwargs["padding"]

        # Get the encoder index for processed images
        processed_image_encoder_index = []
        processed_image_encoder_index_map = {}
        for key, img in processed_images.items():
            processed_image_encoder_index.append(self.encoder_index[key])
            processed_image_encoder_index_map[self.encoder_index[key]] = key

        inputs = self.encoder_processor(images, **kwargs)

        if self.clip_processor:
            inputs["clip_inputs"] = self.clip_processor(images, **kwargs)
        
        auxiliary_inputs = []
        for i, processor in enumerate(self.auxiliary_processors):
            # If current encoder is given a processed image, use the processed image
            if i in processed_image_encoder_index:
                # NOTE: We use the default processor for processed images for now
                encoder_name = processed_image_encoder_index_map[i]
                aux_inputs = self.encoder_processor(
                    processed_images[encoder_name], **kwargs
                )
                aux_inputs["use_processed"] = True
            else:
                aux_inputs = processor(images, **kwargs)
                aux_inputs["use_processed"] = False

            auxiliary_inputs.append(aux_inputs)

        inputs["aux_inputs"] = auxiliary_inputs

        return inputs


class VisionTower(torch.nn.Module):

    def __init__(
        self,
        model_params: ModelParams,
        clip_encoder: Optional[VisionEncoder] = None,
        vision_feature_layer: int = -2,
        language_embeds_dim: int = 768,
        segment_type: str = "semantic",  # ["semantic", "instance", "panoptic"]
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
        cache_dir: Path = Path(".cache"),
    ):
        super().__init__()

        self.encoder = VisionEncoder(model_params.encoder_id)
        # Get the feature dimension of the encoder
        image_size = self.encoder.processor.crop_size
        self.encoder_patch_size = self.encoder.model.config.patch_size
        self.encoder_patch_width = image_size["width"] // self.encoder_patch_size
        self.encoder_patch_height = image_size["height"] // self.encoder_patch_size
        self.encoder_feature_dim = self.encoder.model.config.hidden_size

        self.use_clip = model_params.use_clip
        self.interpolation_mode = model_params.interpolation_mode
        if self.use_clip:
            self.clip_encoder = clip_encoder
            # Get the feature dimension of the clip encoder
            image_size = self.clip_encoder.processor.crop_size
            self.clip_patch_size = self.clip_encoder.model.config.patch_size
            self.clip_patch_width = image_size["width"] // self.clip_patch_size
            self.clip_patch_height = image_size["height"] // self.clip_patch_size
            self.clip_feature_dim = self.clip_encoder.model.config.hidden_size

            # NOTE: We may add projector if the feature dimension is different
            assert (
                self.clip_feature_dim == self.encoder_feature_dim
            ), "The feature dimension of the clip encoder and the encoder must be the same, but got {} and {}".format(
                self.clip_feature_dim, self.encoder_feature_dim
            )

        self.auxiliary_processors = []
        self.vision_feature_layer = vision_feature_layer
        self.cache_postfix: List[str] = ["rgb"]  ## The key defined in dataset.py

        self.auxiliary_encoders = torch.nn.ModuleList()
        self.auxiliary_projectors = torch.nn.ModuleList()
        self.encoder_index = dict()

        self.use_depth = model_params.use_depth
        if model_params.use_depth:
            self.encoder_index["depth"] = len(self.auxiliary_processors)

            depth_encoder = DepthEncoder(
                model_params.depth_model_id,
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
            self.auxiliary_processors.append(
                self.encoder.processor
            )  # Use the main processor for depth encoder
            self.cache_postfix.append("depth")

        self.use_segmentation = model_params.use_segmentation
        if model_params.use_segmentation:
            self.encoder_index["segmentation"] = len(self.auxiliary_processors)

            segmentation_encoder = SegmentationEncoder(
                model_params.segmentation_model_id,
                ignore_model=model_params.use_processed,
                segment_type=segment_type,
                image_target_size=(800, 1200),
                vision_feature_layer=vision_feature_layer,
                device=device,
                torch_dtype=torch_dtype,
            )
            self.auxiliary_encoders.append(segmentation_encoder)
            # No projection for segmentation encoder
            self.auxiliary_projectors.append(nn.Identity())
            self.auxiliary_processors.append(segmentation_encoder.task_processor)
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
            self.encoder.processor,
            self.auxiliary_processors,
            encoder_index=self.encoder_index,
            clip_processor=self.clip_encoder.processor if self.use_clip else None,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.cache_dir = cache_dir

    def forward(self, inputs, **kwargs):
        language_embeds = None
        if self.conditional_fuser:
            inputs, language_embeds = inputs

        clip_image_feature, image_feature, auxiliary_features = self._get_feature(
            inputs, **kwargs
        )
        out_feature = self._maybe_fuse(
            image_feature, auxiliary_features, language_embeds
        )

        if self.use_clip:
            out_feature = self._interpolate_feature(out_feature)
            out_feature = clip_image_feature + out_feature

        # NOTE: Interpolate before fusion is also possible
        return BaseModelOutputWithPooling(
            # vision_feature_layer is -2 in Llava, so we need to add placeholder for the output
            hidden_states=[out_feature, None]
        )

    def _interpolate_feature(self, features: torch.Tensor):
        batch_size = features.shape[0]

        class_feature = features[:, :1, :]
        patch_features = features[:, 1:, :]
        patch_features = patch_features.reshape(
            batch_size, self.encoder_patch_height, self.encoder_patch_width, -1
        ).permute(0, 3, 1, 2)

        patch_features = F.interpolate(
            patch_features,
            size=(self.clip_patch_height, self.clip_patch_width),
            align_corners=True if self.interpolation_mode != "nearest" else None,
            mode=self.interpolation_mode,
        ).permute(0, 2, 3, 1)
        patch_features = patch_features.reshape(
            batch_size, -1, self.encoder_feature_dim
        )

        features = torch.cat([class_feature, patch_features], dim=1)
        return features

    def _get_feature(self, inputs, **kwargs):
        pixel_values = inputs[0]
        clip_inputs = inputs[1]  # Would be zero if not use clip
        auxiliary_inputs = inputs[2:]

        feature: ImageEncoderOutput = self.encoder(pixel_values, **kwargs)
        image_feature = feature.hidden_states[self.vision_feature_layer]

        clip_image_feature = None
        if self.use_clip:
            feature: ImageEncoderOutput = self.clip_encoder(**clip_inputs, **kwargs)
            clip_image_feature = feature.hidden_states[self.vision_feature_layer]

        main_dim = tuple(image_feature.shape)
        auxiliary_features = self._maybe_get_auxiliary_features(
            auxiliary_inputs,
            main_dim,
            device=image_feature.device,
            dtype=image_feature.dtype,
            **kwargs,
        )

        return clip_image_feature, image_feature, auxiliary_features

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
        auxiliary_inputs,
        main_dim: Tuple[int, int, int],
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        if self.n_auxiliary_features == 0:
            return None

        auxiliary_features = []
        for i, encoder in enumerate(self.auxiliary_encoders):
            inputs = auxiliary_inputs[i]

            use_processed = inputs["use_processed"].all()
            inputs.pop("use_processed")
            if use_processed:
                # NOTE: Directly use the main encoder if this feature is uses processed image
                output: ImageEncoderOutput = self.encoder(**inputs, **kwargs)
            else:
                output: ImageEncoderOutput = encoder(**inputs, **kwargs)

            if output.use_pred:
                # NOTE: A special case for segmentation
                preprocess = self.auxiliary_processors[0](
                    output.predictions.to(dtype=torch.float32),
                    return_tensors="pt",  # return as pytorch tensors
                )
                preprocess = preprocess.to(device=device, dtype=dtype)
                encoded: ImageEncoderOutput = self.encoder(
                    preprocess["pixel_values"], **kwargs
                )
                wanted_feature = encoded.hidden_states[self.vision_feature_layer]
            else:
                wanted_feature = output.hidden_states[self.vision_feature_layer]
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
