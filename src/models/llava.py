from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, PeftMixedModel, get_peft_model, LoraRuntimeConfig
from PIL.Image import Image
from transformers import LlavaForConditionalGeneration, LlavaProcessor

# from src.arguments.dataclass import ModelParams
from src.arguments.deepspeed import ModelParams
from src.utils import batch_feature_to_dict, container_cat, default, pad_sequences

from .encoder.base import VisionEncoder
from .utils import ensure_all_on_device, ensure_all_same_dtype
from .vision_tower import VisionTower


class LlavaPEFT(torch.nn.Module):
    def __init__(
        self,
        model_params: ModelParams,
        gradient_checkpointing: bool,
        lora_config: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super().__init__()

        self.model_id = model_params.model_id
        llava = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        )

        processor: LlavaProcessor = LlavaProcessor.from_pretrained(
            model_params.model_id
        )
        processor.vision_feature_select_strategy = (
            model_params.vision_feature_select_strategy
        )

        self.only_clip = model_params.only_clip
        if not self.only_clip:
            # Change the vision tower to the encoder
            vision_feature_layer = default(llava.config.vision_feature_layer, -2)
            self.conditional_fuser = model_params.conditional_fuser
            self.condition_dropout = model_params.condition_dropout
            language_embeds_dim = llava.get_input_embeddings().weight.shape[1]

            self.use_clip = model_params.use_clip
            clip_encoder = None
            if self.use_clip:
                clip_encoder = VisionEncoder(
                    model=llava.vision_tower.vision_model,
                    processor=processor.image_processor,
                )

            self.vision_tower = VisionTower(
                model_params,
                clip_encoder=clip_encoder,
                vision_feature_layer=vision_feature_layer,
                language_embeds_dim=language_embeds_dim,
                torch_dtype=torch_dtype,
                device=device,
            )
            llava.vision_tower = self.vision_tower

            # Update vision related config if not use clip
            if not self.use_clip:
                llava.config.vision_config = self.vision_tower.encoder.model.config
                llava.config.image_seq_length = (
                    self.vision_tower.encoder_patch_width
                    * self.vision_tower.encoder_patch_height
                )
                processor.image_processor = self.vision_tower.encoder.processor
                processor.patch_size = self.vision_tower.encoder_patch_size

        # Remove projector layers from lora for direct finetuning
        self.no_lora_but_FF_prefix = default(
            model_params.no_lora_but_FF_prefix,
            [
                "multi_modal_projector",
                "fuser",
                "vision_tower.AdaLNZero",
                "auxiliary_projectors",
            ],
        )
        model_params.lora_config["target_modules"] = [
            layer
            for layer in model_params.lora_config["target_modules"]
            if not any([prefix in layer for prefix in self.no_lora_but_FF_prefix])
        ]

        # These should be apply before apply PEFT
        ensure_all_same_dtype(llava, torch_dtype)
        ensure_all_on_device(llava, device)

        # Activate gradient checkpointing
        if gradient_checkpointing:
            llava.enable_input_require_grads()
            llava.gradient_checkpointing_enable({"use_reentrant": True})

        # Apply LoRA
        lora_config = LoraConfig(**lora_config, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=True))
        self.llava: PeftMixedModel = get_peft_model(llava, lora_config)
        self.processor = processor

        self.activate_only_lora()
        # Activate finetuning the encoder
        for name, param in llava.named_parameters():
            if any([prefix in name for prefix in self.no_lora_but_FF_prefix]):
                param.requires_grad = True

    def setup_requires_grad(self):
        self.activate_only_lora()
        # Activate finetuning the encoder
        for name, param in self.llava.named_parameters():
            if any([prefix in name for prefix in self.no_lora_but_FF_prefix]):
                param.requires_grad = True

    def get_model_struct(self):
        return str(self.llava)

    def activate_only_lora(self):
        for name, param in self.llava.named_parameters():
            param.requires_grad = "lora" in name

    def transform(
        self, img: Optional[Image], prompt: str, processed_images: Dict[str, Image] = {}
    ):
        inputs = self.processor(
            img,
            text=prompt,
            return_tensors="pt",  # return as pytorch tensors
            padding=True,
            do_rescale=True,
        )

        if not self.only_clip and img is not None:
            image_inputs = self.vision_tower.processor(
                img,
                processed_images=processed_images,
                return_tensors="pt",  # return as pytorch tensors
                padding=True,
                do_rescale=True,
            )

            inputs["pixel_values"] = image_inputs["pixel_values"]
            inputs["aux_inputs"] = image_inputs["aux_inputs"]
            if self.use_clip:
                inputs["clip_inputs"] = image_inputs["clip_inputs"]
            else:
                inputs["clip_inputs"] = 0  # placeholder

        return inputs

    def forward(
        self,
        pixel_values: torch.Tensor,
        clip_inputs: Optional[Dict[str, Any]] = None,
        aux_inputs: Optional[List[Dict[str, Any]]] = None,
        **inputs,
    ):
        if self.only_clip:
            inputs["pixel_values"] = pixel_values
            return self.llava(**inputs)

        if aux_inputs is not None:
            inputs["pixel_values"] = [pixel_values, clip_inputs, *aux_inputs]
        else:
            inputs["pixel_values"] = [pixel_values, clip_inputs]

        if self.conditional_fuser:
            # language_embeds = self.llava.base_model.get_input_embeddings()(
            #     inputs["input_ids"].to(torch.long)
            # )
            # language_embeds = language_embeds.mean(dim=1)
            lang_inputs = inputs.copy()
            lang_inputs.pop("pixel_values")

            outputs = self.llava.base_model(**lang_inputs, output_hidden_states=True)
            embeddings: torch.Tensor = outputs.hidden_states[-1]

            attention_mask: torch.Tensor = lang_inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1).unsqueeze(-1), min=1e-9)
            language_embeds = sum_embeddings / sum_mask

            language_embeds = language_embeds.to(pixel_values.dtype)
            inputs["pixel_values"] = (inputs["pixel_values"], language_embeds)

        return self.llava(**inputs)

    def merge_adapter(self):
        self.llava.merge_adapter()

    def merge_and_unload(self, inplace: bool = False):
        if inplace:
            self.llava = self.llava.merge_and_unload()
            return self
        return self.llava.merge_and_unload()

    def unload(self, inplace: bool = False):
        if inplace:
            self.llava = self.llava.unload()
            return self
        return self.llava.unload()

    def unmerge_adapter(self):
        self.llava.unmerge_adapter()

    def enssetial_state_dict(self):
        esstial_keywords = self.no_lora_but_FF_prefix
        output_state_dict = {}
        for key, value in self.llava.state_dict().items():
            if any([keyword in key for keyword in esstial_keywords]):
                output_state_dict[key] = value
        return output_state_dict

    def load_state_dict(self, state_dict, strict=False, assign=False):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("llava"):
                new_state_dict[key[6:]] = value
            else:
                new_state_dict[key] = value
        self.llava.load_state_dict(new_state_dict, strict=strict, assign=assign)

    def generate(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        aux_inputs: Optional[List[torch.Tensor]] = None,
        clip_inputs: Dict[str, Any] = None,
        **inputs,
    ):
        if self.only_clip:
            inputs["pixel_values"] = pixel_values
            return self.llava.generate(**inputs)

        if aux_inputs is not None and clip_inputs is not None:
            inputs["pixel_values"] = [pixel_values, clip_inputs, *aux_inputs]
        elif aux_inputs is not None:
            inputs["pixel_values"] = [pixel_values, *aux_inputs]
        elif pixel_values is not None:
            inputs["pixel_values"] = [pixel_values]
        else:
            inputs["pixel_values"] = None

        if self.conditional_fuser:
            language_embeds = self.llava.base_model.get_input_embeddings()(
                inputs["input_ids"]
            )

            language_embeds = language_embeds.mean(dim=1)
            inputs["pixel_values"] = (inputs["pixel_values"], language_embeds)

        return self.llava.generate(**inputs)

    def finetune_language(self, finetune: bool = True):
        for name, param in self.llava.named_parameters():
            if "lora" in name:
                param.requires_grad = finetune


PADDING_TOKEN = 32001
ATTENTION_MASK = 0


def collate_fn(batch):
    # From List[BatchFeature] to List[Dict]
    datamaps = [batch_feature_to_dict(item[1]) for item in batch]
    data_keys = datamaps[0].keys()
    # is_tensor = [isinstance(datamaps[0][key], torch.Tensor) for key in data_keys]

    # Extract input_ids and other data from the batch
    input_ids = [item[1]["input_ids"] for item in batch]
    attention_mask = [item[1]["attention_mask"] for item in batch]

    # Pad input_ids to the same length
    padded_input_ids = pad_sequences(input_ids, PADDING_TOKEN)
    padded_attention_mask = pad_sequences(attention_mask, ATTENTION_MASK)

    # Update the batch with padded input_ids
    for i, datamap in enumerate(datamaps):
        datamap["input_ids"] = padded_input_ids[i]
        datamap["attention_mask"] = padded_attention_mask[i]

    merged_ids = [item[0] for item in batch]
    # Return the updated batch
    return merged_ids, container_cat(datamaps, dim=0)


if __name__ == "__main__":
    from pathlib import Path

    from src.arguments.dataclass import Config
    from src.utils.experiment import load_config

    name = "DINO_r32_rgbd"
    base_dir = Path("outputs/DINO_r32_rgbd_1214_164435")
    config, timestamp, output_dir, checkpoint_dir, log_dir = load_config(
        name, Config, auto_create=True
    )

    import torch
    from transformers.utils import logging

    logging.set_verbosity_error()

    from src.models.llava import LlavaPEFT
    from src.models.utils import ensure_all_on_device, ensure_all_same_dtype
    from src.utils import default

    addition_config = {}
    mp = config.model
    dp = config.dataset
    pp = config.pipeline
    op = config.optimization

    device = torch.device("cpu")

    __USE_DEEPSPEED__ = False
    local_rank = -1
    use_cache = not (mp.gradient_checkpointing and not __USE_DEEPSPEED__)

    model = LlavaPEFT(
        model_params=mp,
        gradient_checkpointing=not use_cache,
        lora_config=mp.lora_config,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    model.load_state_dict(
        torch.load(
            base_dir / "checkpoint" / "latest.pt",
            map_location=device,
            weights_only=True,
        )
    )

    torch.save(
        model.enssetial_state_dict(),
        base_dir / "checkpoint" / "shrink.pt",
    )

    ## Estimate memory saved

    # original pt file memory
    print("Original pt file memory")
    original_memory = (base_dir / "checkpoint" / "latest.pt").stat().st_size
    shirnk_memory = (base_dir / "checkpoint" / "shrink.pt").stat().st_size
    print(f"Original memory: {original_memory/ 2 ** 20} MB")
    print(f"Shrink memory: {shirnk_memory/2 ** 20} MB")
    print(f"Saved memory: {(original_memory - shirnk_memory)/2 ** 20} MB")
