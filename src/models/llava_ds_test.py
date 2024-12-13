from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model
import loralib as lora
from PIL.Image import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

from src.arguments.dataclass import ModelParams
from src.utils import default

from .encoder import DepthEncoder, SegmentationEncoder, VisionEncoder
from .fuser import FUSERS


class LlavaPEFT(torch.nn.Module):
    def __init__(
        self,
        model_params: ModelParams,
        gradient_checkpointing: bool,
        lora_config: Dict[str, Any],
    ):
        super().__init__()

        self.model_id = model_params.model_id
        model = AutoModelForImageClassification.from_pretrained(
            self.model_id,
            # torch_dtype=torch.bfloat16,
        )

        processor = AutoImageProcessor.from_pretrained(model_params.model_id)
        self.processor = processor

        # Remove projector layers from lora for direct finetuning
        no_lora_but_FF_prefix = ["multi_modal_projector", "fuser"]
        model_params.lora_config["target_modules"] = [
            layer
            for layer in model_params.lora_config["target_modules"]
            if not any([prefix in layer for prefix in no_lora_but_FF_prefix])
        ]

        # Activate gradient checkpointing
        if gradient_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable({"use_reentrant": True})

        # Apply LoRA
        lora_config = LoraConfig(**lora_config)
        # classifier = lora.Linear(
        #     in_features=model.classifier.in_features,
        #     out_features=model.classifier.out_features,
        #     bias=model.classifier.bias is not None,
        # )
        self.model = model
        # self.model = get_peft_model(model, lora_config)
        # self.activate_only_lora()

        estimate_zero3_model_states_mem_needs_all_live(self.model)

        # Activate finetuning the encoder
        for name, param in self.model.named_parameters():
            if any([name.startswith(prefix) for prefix in no_lora_but_FF_prefix]):
                param.requires_grad = True

        self.mp = model_params

    def get_model_struct(self):
        return str(self.model)

    def activate_only_lora(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora" in name
            if param.requires_grad:
                print(f"Activating {name} for training.")

    def transform(self, img: Image, prompt: str):
        inputs = self.processor(
            img,
            return_tensors="pt",  # return as pytorch tensors
            padding=True,
            do_rescale=False,  # since we already rescale color range to [0, 1] when loading dataset
        )

        return inputs

    def forward(self, inputs):
        return self.model(**inputs)
