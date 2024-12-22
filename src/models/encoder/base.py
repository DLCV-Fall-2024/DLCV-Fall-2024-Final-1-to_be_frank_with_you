from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from transformers.utils import ModelOutput


@dataclass
class ImageEncoderOutput(ModelOutput):
    use_pred: Optional[bool] = False
    predictions: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    task_input: Optional[torch.FloatTensor] = None


class VisionEncoder(nn.Module):
    def __init__(
        self,
        model_id: Optional[str] = None,
        model: Optional[AutoModel] = None,
        processor: Optional[AutoImageProcessor] = None,
        vision_feature_layer: Optional[int] = None,
        **kwargs,
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
        self.vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else -2
        )
        print("Using encoder: ", type(self.model))

    @property
    def hidden_states_dim(self):
        try:
            return self.model.config.hidden_size
        except:
            return self.model.config.neck_hidden_sizes[-1]

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> ImageEncoderOutput:
        output = self.model(pixel_values, **kwargs)
        return ImageEncoderOutput(
            predictions=getattr(output, "last_hidden_state", None),
            hidden_states=getattr(output, "hidden_states", None),
            attentions=getattr(output, "attentions", None),
            loss=getattr(output, "loss", None),
            use_pred=getattr(output, "loss", False),
        )
