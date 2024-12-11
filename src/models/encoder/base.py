from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


class VisionEncoder(nn.Module):
    def __init__(
        self,
        model_id: Optional[str] = None,
        model: Optional[AutoModel] = None,
        processor: Optional[AutoImageProcessor] = None,
    ):
        super().__init__()

        if model is not None and processor is not None:
            self.model = model
            self.processor = processor

        else:
            assert (
                model_id is not None
            ), "Either model_id or model and processor must be provided"
            self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            self.processor = AutoImageProcessor.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            )

        print("Using encoder: ", type(self.model))

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        return self.model(pixel_values, **kwargs)
