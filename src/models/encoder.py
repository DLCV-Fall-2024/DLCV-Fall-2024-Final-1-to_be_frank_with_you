import torch
import torch.nn as nn
import transformers


class VisionEncoder(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()

        self.model_id = model_id
        self.model = transformers.AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        self.processor = transformers.AutoImageProcessor.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )

        print("Using encoder: ", type(self.model))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
