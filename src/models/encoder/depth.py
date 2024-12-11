from typing import Optional

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .base import VisionEncoder


class DepthEncoder(VisionEncoder):
    def __init__(
        self,
        model_id: Optional[str] = None,
        model: Optional[AutoModelForDepthEstimation] = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf"),
        processor: Optional[AutoImageProcessor] = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf"),
    ):
        super().__init__(model_id, model, processor)
