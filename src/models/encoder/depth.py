from typing import Optional

import torch
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForDepthEstimation,
    DPTImageProcessor,
)
from transformers.modeling_outputs import DepthEstimatorOutput

from .base import ImageEncoderOutput, VisionEncoder


class DepthEncoder(VisionEncoder):
    def __init__(
        self,
        model_id: Optional[str] = None,
        model: Optional[AutoModel] = None,
        processor: Optional[AutoImageProcessor] = None,
        vision_feature_layer: Optional[int] = None,
        torch_dtype: Optional[str] = "float16",
        device: Optional[str] = "cuda",
        **kwargs,
    ):
        if model_id is not None:
            model = AutoModelForDepthEstimation.from_pretrained(
                model_id, torch_dtype=torch_dtype
            )
            # image_mean = [0.485, 0.456, 0.406]
            # image_std = [0.229, 0.224, 0.225]
            processor: DPTImageProcessor = AutoImageProcessor.from_pretrained(
                model_id, torch_dtype=torch_dtype
            )
            model = model.to(device)
            model_id = None
        super().__init__(
            model=model, processor=processor, vision_feature_layer=vision_feature_layer
        )

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> ImageEncoderOutput:
        outputs: DepthEstimatorOutput = self.model(pixel_values, **kwargs)

        out = ImageEncoderOutput(
            predictions=outputs.predicted_depth,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss=outputs.loss,
            use_pred=False,
        )

        return out

    def normalize_depth(
        self, origin_image: torch.Tensor, outputs: DepthEstimatorOutput
    ):
        target_sizes = [(img.shape[1], img.shape[2]) for img in origin_image]
        post_processed_output = self.processor.post_process_depth_estimation(
            outputs,
            target_sizes,
        )
        normalized_depths = []
        for i in range(len(post_processed_output)):
            # Extract depth map for the current image
            depth = post_processed_output[i]["predicted_depth"]

            # Normalize each depth map (scale to 0-255)
            normalized_depth = (depth - depth.min()) / (
                depth.max() - depth.min()
            )  # Normalize to [0, 1]
            normalized_depth = normalized_depth * 255  # Convert to [0, 255] range

            # Append the normalized depth map to the list
            normalized_depths.append(normalized_depth)

        normalized_depths = torch.stack(normalized_depths)
        return normalized_depth
