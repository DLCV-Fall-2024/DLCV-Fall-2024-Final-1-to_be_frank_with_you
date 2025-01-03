import colorsys
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModel,
    DinatModel,
    OneFormerConfig,
    OneFormerForUniversalSegmentation,
    OneFormerImageProcessor,
    OneFormerProcessor,
)
from transformers.feature_extraction_utils import BatchFeature

from .base import ImageEncoderOutput, VisionEncoder


class SegmentationEncoder(VisionEncoder):

    def __init__(
        self,
        model_id: Optional[str] = None,
        model: Optional[AutoModel] = None,
        processor: Optional[AutoImageProcessor] = None,
        ignore_model: bool = False,
        segment_type: Optional[str] = "semantic",
        vision_feature_layer: Optional[int] = None,
        image_target_size: Optional[Tuple[int, int]] = None,
        id2rgb_path: Optional[str] = "id2rgb.pt",
        torch_dtype: Optional[str] = "float16",
        device: Optional[str] = "cuda",
        **kwargs,
    ):
        if model_id is not None:
            # NOTE: Ignore model is only set if processed images are used
            if ignore_model:
                model = object()
            else:
                model = OneFormerForUniversalSegmentation.from_pretrained(
                    model_id, torch_dtype=torch_dtype
                )
                # WARNING: Not sure if this is necessary
                model = model.to(device)

            image_processor: OneFormerImageProcessor = (
                OneFormerImageProcessor.from_pretrained(
                    model_id, size=image_target_size, torch_dtype=torch_dtype
                )
            )
            processor: OneFormerProcessor = OneFormerProcessor.from_pretrained(
                model_id, torch_dtype=torch_dtype
            )
            processor.image_processor = image_processor

            model_id = None

        super().__init__(model_id, model, processor, vision_feature_layer, **kwargs)
        self.image_processor = image_processor
        self.image_target_size = image_target_size
        self.label_ids_to_fuse = set()
        self.segment_type = segment_type

        self.torch_dtype = torch_dtype
        self.device = device

        # self.id2rgb = torch.tensor(
        #     [
        #         [
        #             int(c * 255) for c in colorsys.hsv_to_rgb(i / 255, 1.0, 1.0)
        #         ]  # Saturation=1, Value=1
        #         for i in range(255)
        #     ]
        #     + [[0, 0, 0]],  # Black for ignore value
        #     dtype=torch.uint8,
        # ).to(self.device)
        self.id2rgb = torch.load(id2rgb_path, weights_only=True).to(self.device)

    def task_processor(self, images, **kwargs):
        return self.processor(images, task_inputs=[self.segment_type], **kwargs)

    @property
    def hidden_states_dim(self) -> int:
        return self.model.config.num_queries

    def forward(
        self,
        pixel_values,
        threshold: int = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        **kwargs,
    ) -> ImageEncoderOutput:
        pixel_values = pixel_values.to(torch.float32)
        self.model = self.model.to(torch.float32)

        batch_size = pixel_values.shape[0]
        task_inputs = kwargs.get("task_inputs", None)
        assert task_inputs is not None, "task_inputs is required for segmentation"
        assert isinstance(
            task_inputs, torch.Tensor
        ), "task_inputs must be a torch.Tensor, you should use `OneFormerProcessor` instead of `OneFormerImageProcessor`"
        if task_inputs.shape[0] != batch_size:
            task_inputs = task_inputs.expand(batch_size, -1)
        kwargs["task_inputs"] = task_inputs
        outputs = self.model(pixel_values, **kwargs)

        results: List[Dict[str, torch.Tensor]] = (
            self.image_processor.post_process_panoptic_segmentation(
                outputs,
                threshold,
                mask_threshold,
                overlap_mask_area_threshold,
                label_ids_to_fuse=self.label_ids_to_fuse,
                target_sizes=[
                    self.image_target_size for _ in range(pixel_values.shape[0])
                ],
            )
        )
        predictions = []
        IGNORE_VALUE = 255
        for result in results:
            segmentation = result["segmentation"]
            segmentation_info = result["segments_info"]
            infos = {v["id"]: v["label_id"] for v in segmentation_info}

            pred = torch.zeros_like(segmentation)
            label_ids = torch.unique(segmentation).tolist()
            for label_id in label_ids:
                mask = segmentation == label_id
                try:
                    pred[mask] = infos[label_id]
                except KeyError:
                    pred[mask] = IGNORE_VALUE
            pred = pred.to(self.device)
            color_pred = self.id2rgb[pred.long()]
            color_pred = color_pred.permute(2, 0, 1)
            predictions.append(color_pred)

        predictions = torch.stack(predictions).to(self.device, self.torch_dtype)
        if len(predictions.shape) == 3:
            predictions = predictions.unsqueeze(0)

        out = ImageEncoderOutput(
            predictions=predictions,
            hidden_states=outputs.auxiliary_predictions,
            attentions=outputs.attentions,
            loss=outputs.loss,
            use_pred=True,
        )
        return out


if __name__ == "__main__":
    id2rgb = torch.randint(0, 256, (256, 3), dtype=torch.uint8)
    torch.save(id2rgb, "id2rgb.pt")


"""
OneFormerConfig {
  "_attn_implementation_autoset": true,
  "_name_or_path": "shi-labs/oneformer_ade20k_dinat_large",
  "architectures": ["OneFormerForUniversalSegmentation"],
  "backbone": null,
  "backbone_config": {
    "architectures": ["DinatForImageClassification"],
    "depths": [3,4,18,5],
    "dilations": [
      [1,20,1],
      [1,5,1,10],
      [1,2,1,3,1,4,1,5,1,2,1,3,1,4,1,5,1,5],
      [1,2,1,2,1]
    ],
    "drop_path_rate": 0.35,
    "embed_dim": 192,
    "hidden_size": 1536,
    "kernel_size": 11,
    "layer_scale_init_value": 0.0,
    "mlp_ratio": 2.0,
    "model_type": "dinat",
    "num_heads": [6,12,24,48],
    "out_features": ["stage1","stage2","stage3","stage4"],
    "out_indices": [1,2,3,4],
    "path_norm": true,
    "torch_dtype": "float32"
  },
  "backbone_kwargs": null,
  "class_weight": 2.0,
  "common_stride": 4,
  "contrastive_temperature": 0.07,
  "contrastive_weight": 0.5,
  "conv_dim": 256,
  "decoder_layers": 10,
  "dice_weight": 5.0,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "encoder_feedforward_dim": 1024,
  "encoder_layers": 6,
  "enforce_input_proj": false,
  "hidden_dim": 256,
  "ignore_value": 255,
  "importance_sample_ratio": 0.75,
  "init_std": 0.02,
  "init_xavier_std": 1.0,
  "is_training": false,
  "layer_norm_eps": 1e-05,
  "mask_dim": 256,
  "mask_weight": 5.0,
  "max_seq_len": 77,
  "model_type": "oneformer",
  "no_object_weight": 0.1,
  "norm": "GN",
  "num_attention_heads": 8,
  "num_classes": 150,
  "num_hidden_layers": 10,
  "num_queries": 250,
  "output_attentions": true,
  "output_auxiliary_logits": true,
  "output_hidden_states": true,
  "oversample_ratio": 3.0,
  "pre_norm": false,
  "query_dec_layers": 2,
  "strides": [4,8,16,32],
  "task_seq_len": 77,
  "text_encoder_context_length": 77,
  "text_encoder_n_ctx": 16,
  "text_encoder_num_layers": 6,
  "text_encoder_proj_layers": 2,
  "text_encoder_vocab_size": 49408,
  "text_encoder_width": 256,
  "torch_dtype": "float32",
  "train_num_points": 12544,
  "transformers_version": "4.46.3",
  "use_auxiliary_loss": true,
  "use_pretrained_backbone": false,
  "use_task_norm": true,
  "use_timm_backbone": false
}

# scrpit
for key, value in outputs.items():
    if isinstance(value, torch.Tensor):
        print(key,":",value.shape)
    elif isinstance(value, (tuple, list)):
        print(f"{key}: {type(value)}")
        for idx, elem in enumerate(value):
            
            if isinstance(elem, torch.Tensor):
                print(f"\t{idx}: {elem.shape}")
            elif isinstance(elem, (tuple, list)):
                print(f"\t{idx}:")
                for sidx, selem in enumerate(elem):
                    if isinstance(selem, torch.Tensor):
                        print(f"\t\t{sidx}: {selem.shape}")
                    else:
                        print(f"\t\t{sidx}: {selem}")
            elif isinstance(elem, dict):
                print(f"\t{idx}: {type(elem)}")
                for dkey, delem in elem.items():        
                    if isinstance(delem, torch.Tensor):
                        print(f"\t\t{dkey}: {delem.shape}")
                    elif isinstance(delem, (tuple, list)):
                        print(f"\t\t{dkey}:")
                        for sidx, selem in enumerate(elem):
                            if isinstance(selem, torch.Tensor):
                                print(f"\t\t\t{sidx}: {selem.shape}")
                            else:    
                                print(f"\t\t\t{sidx}: {selem}")
                    else:
                        print(f"\t\t{dkey}: {type(delem)}")
            else:
                print(f"\t{idx}: {type(elem)}")
    elif isinstance(value, dict):
        print(f"{key}: {type(value)}")
        for ekey, elem in value.items():
            
            if isinstance(elem, torch.Tensor):
                print(f"\t{ekey}: {elem.shape}")
            elif isinstance(elem, (tuple, list)):
                print(f"\t{ekey}:")
                for sidx, selem in enumerate(elem):
                    if isinstance(selem, torch.Tensor):
                        print(f"\t\t{sidx}: {selem.shape}")
                    else:
                        print(f"\t\t{sidx}: {selem}")
            else:
                print(f"\t{ekey}: {type(elem)}")
    else:
        print(f"{key}: {type(value)}")
        
# outputs:
class_queries_logits : torch.Size([1, 250, 151])
masks_queries_logits : torch.Size([1, 250, 160, 160])
auxiliary_predictions: <class 'tuple'>
	0: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	1: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	2: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	3: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	4: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	5: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	6: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	7: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	8: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
encoder_hidden_states: <class 'tuple'>
	0: torch.Size([1, 192, 160, 160])
	1: torch.Size([1, 384, 80, 80])
	2: torch.Size([1, 768, 40, 40])
	3: torch.Size([1, 1536, 20, 20])
pixel_decoder_hidden_states: <class 'tuple'>
	0: torch.Size([1, 256, 160, 160])
	1: torch.Size([1, 256, 20, 20])
	2: torch.Size([1, 256, 40, 40])
	3: torch.Size([1, 256, 80, 80])
transformer_decoder_hidden_states: <class 'tuple'>
	0: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	1: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	2: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	3: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	4: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	5: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	6: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	7: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	8: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
transformer_decoder_object_queries : torch.Size([1, 250, 256])
transformer_decoder_contrastive_queries : torch.Size([1, 250, 256])
transformer_decoder_mask_predictions : torch.Size([1, 250, 160, 160])
transformer_decoder_class_predictions : torch.Size([1, 250, 151])
transformer_decoder_auxiliary_predictions: <class 'tuple'>
	0: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	1: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	2: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	3: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	4: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	5: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	6: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	7: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
	8: <class 'dict'>
		class_queries_logits: torch.Size([1, 250, 151])
		masks_queries_logits: torch.Size([1, 250, 160, 160])
task_token : torch.Size([1, 256])
attentions: <class 'tuple'>
	0:
		0: torch.Size([1, 8, 250, 250])
		1: torch.Size([1, 250, 400])
	1:
		0: torch.Size([1, 8, 250, 250])
		1: torch.Size([1, 250, 1600])
	2:
		0: torch.Size([1, 8, 250, 250])
		1: torch.Size([1, 250, 6400])
	3:
		0: torch.Size([1, 8, 250, 250])
		1: torch.Size([1, 250, 400])
	4:
		0: torch.Size([1, 8, 250, 250])
		1: torch.Size([1, 250, 1600])
	5:
		0: torch.Size([1, 8, 250, 250])
		1: torch.Size([1, 250, 6400])
	6:
		0: torch.Size([1, 8, 250, 250])
		1: torch.Size([1, 250, 400])
	7:
		0: torch.Size([1, 8, 250, 250])
		1: torch.Size([1, 250, 1600])
	8:
		0: torch.Size([1, 8, 250, 250])
		1: torch.Size([1, 250, 6400])


"""
