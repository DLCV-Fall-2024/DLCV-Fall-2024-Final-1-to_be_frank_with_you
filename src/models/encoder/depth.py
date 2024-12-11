from typing import Optional

from transformers import AutoImageProcessor, AutoModel

from .base import VisionEncoder

import torch


class DepthEncoder(VisionEncoder):
    def __init__(
        self,
        model_id: Optional[str] = None,
        model: Optional[AutoModel] = None,
        processor: Optional[AutoImageProcessor] = None,
    ):
        super().__init__(model_id, model, processor)
    
    def forward( self, pixel_values: torch.Tensor, **kwargs ):
        inputs = self.processor(images=pixel_values, return_tensors="pt")  # I guess that pixel_values is between 0-255

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [(img.shape[1], img.shape[2]) for img in pixel_values]
        # interpolate to original size and visualize the prediction
        post_processed_output = self.processor.post_process_depth_estimation(
            outputs,
            target_sizes,
        )
        normalized_depths = []
        for i in range(len(post_processed_output)):
            # Extract depth map for the current image
            depth = post_processed_output[i]["predicted_depth"]

            # Normalize each depth map (scale to 0-255)
            normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize to [0, 1]
            normalized_depth = normalized_depth.detach().cpu().numpy() * 255  # Convert to [0, 255] range

            # Append the normalized depth map to the list
            normalized_depths.append(normalized_depth)
        normalized_depths = torch.tensor(normalized_depths)
        return normalized_depths
