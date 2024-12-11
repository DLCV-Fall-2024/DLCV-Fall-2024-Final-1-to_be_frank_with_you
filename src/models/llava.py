from typing import Any, Dict

import torch
from PIL.Image import Image
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from peft import LoraConfig, get_peft_model

from src.arguments import ModelParams


class LlavaPEFT(torch.nn.Module):
    def __init__(
        self,
        model_id: str,
        model_params: ModelParams,
        gradient_checkpointing: bool,
        lora_config: Dict[str, Any],
    ):
        super().__init__()

        self.model_id = model_id
        llava = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        if gradient_checkpointing:
            llava.enable_input_require_grads()
            llava.gradient_checkpointing_enable({"use_reentrant": True})

        lora_config = LoraConfig(**lora_config)
        self.llava = get_peft_model(llava, lora_config)
        self.activate_only_lora()

        processor: LlavaProcessor = LlavaProcessor.from_pretrained(model_params.model_id)
        processor.patch_size = model_params.patch_size
        processor.vision_feature_select_strategy = model_params.vision_feature_select_strategy
        self.processor = processor

    def get_model_struct(self):
        return str(self.llava)

    def activate_only_lora(self):
        for name, param in self.llava.named_parameters():
            param.requires_grad = "lora" in name

    def transform(self, img: Image, prompt: str):
        return self.processor(
            img,
            text=prompt,
            return_tensors="pt",  # return as pytorch tensors
            padding=True,
            do_rescale=False,  # since we already rescale color range to [0, 1] when loading dataset
        )

    def forward(self, **inputs):
        return self.llava(**inputs)
