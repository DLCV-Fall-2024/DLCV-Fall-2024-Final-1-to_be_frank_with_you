from typing import Optional

from transformers import AutoModel, AutoImageProcessor

from .base import VisionEncoder


class DepthEncoder(VisionEncoder):
    def __init__(
        self,
        model_id: Optional[str] = None,
        model: Optional[AutoModel] = None,
        processor: Optional[AutoImageProcessor] = None,
    ):
        super().__init__(model_id, model, processor)
