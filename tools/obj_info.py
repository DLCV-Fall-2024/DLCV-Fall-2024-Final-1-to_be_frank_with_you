import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import datasets
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image

# DataLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForDepthEstimation,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    AutoTokenizer,
    DPTImageProcessor,
)

from src.utils import container_to
from src.utils.dataset import BoxInfoPreProcessorDataset, preprocessor_collate_fn

batch_size = 16
prefetch_factor = 8
num_workers = 10
collate_fn = preprocessor_collate_fn
transform = transforms.Compose(
    [
        transforms.Resize((720, 1024)),
        transforms.ToTensor(),
    ]
)


labels = "a car. a motorcycle. a sign. a traffic light. a round sign. a human. a traffic cone. a barrier. a truck."
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-large-uncased"
Bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
Bert_model = AutoModel.from_pretrained(model_name).to(device)
model_id = "IDEA-Research/grounding-dino-tiny"
GD_processor = AutoProcessor.from_pretrained(model_id)
GD_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
Dep_image_processor = AutoImageProcessor.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
)
Dep_model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
).to(device)

basedir = Path("data/test")

info_store_path = basedir / "obj_info.pkl"

val_dataset = BoxInfoPreProcessorDataset(
    path=basedir,
    GD_processor=GD_processor,
    Dep_image_processor=Dep_image_processor,
    labels=labels,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    prefetch_factor=prefetch_factor,
    num_workers=num_workers,
    collate_fn=collate_fn,
)


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        l = []
        for i in range(len(detection_dict["labels"])):
            l.append(
                cls(
                    score=detection_dict["scores"][i],
                    label=detection_dict["labels"][i],
                    box=BoundingBox(
                        xmin=detection_dict["boxes"][i][0],
                        ymin=detection_dict["boxes"][i][1],
                        xmax=detection_dict["boxes"][i][2],
                        ymax=detection_dict["boxes"][i][3],
                    ),
                )
            )
        return l


#### Run This First (2000 images)

rag_data = {}


def get_text_info(GD_results, normalized_depths):
    data_one_image = []
    for i in range(len(GD_results["labels"])):
        boxes = GD_results["boxes"][i].cpu()
        # print(boxes)
        mean_value = normalized_depths.cpu()[int((boxes[1] + boxes[3]) // 2)][
            int((boxes[0] + boxes[2]) // 2)
        ]
        d_range = ""
        orient = ""
        if mean_value > 90:
            d_range = "immediate"
        elif mean_value > 70:
            d_range = "short"
        elif mean_value > 40:
            d_range = "mid"
        else:
            d_range = "long"

        x_mean_value = (boxes[0] + boxes[2]) // 2
        if x_mean_value > 0.8 * normalized_depths.shape[1]:
            orient = "right"
        elif x_mean_value > 0.6 * normalized_depths.shape[1]:
            orient = "center-right"
        elif x_mean_value > 0.4 * normalized_depths.shape[1]:
            orient = "center"
        elif x_mean_value > 0.2 * normalized_depths.shape[1]:
            orient = "center-left"
        else:
            orient = "left"

        out = [GD_results["labels"][i]]
        out.append(f"distance: {d_range}")
        out.append(f"position: {orient}")
        out.append(
            f"bbox(in px): {[int(x) for x in GD_results['boxes'][i].cpu().tolist()]}"
        )
        data_one_image.append(f'<{" | ".join(out)}>')
    json_string = ", ".join(data_one_image)
    ## [<object_name> | <distance> | <position> | <bbox>: [xmin, ymin, xmax, ymax]>]
    return f"[{json_string}]"


import pickle

# Close the client if receiving a SIGINT signal
import signal
import sys

from pymilvus import MilvusClient

base_collection_name = "object_info"
tasks = ["general", "regional", "suggestion"]
if "test" not in basedir.name:
    client = MilvusClient("data/milvus_vector.db")
    for task in tasks:
        collection_name = f"{base_collection_name}_{task}"
        if not client.has_collection(collection_name):
            client.create_collection(
                collection_name=collection_name,
                dimension=1024,  # The vectors we will use in this demo has 768 dimensions
                metric_type="L2",
                consistency_level="Strong",
                auto_id=True,
            )

    def signal_handler(sig, frame):
        client.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
json_out = {}
for index, gd_inputs, dp_inputs, gt, target_sizes, ids in tqdm(val_loader):
    gd_inputs = container_to(gd_inputs, device=device)
    dp_inputs = container_to(dp_inputs, device=device)
    target_sizes = [tuple(t) for t in target_sizes]
    with torch.no_grad():
        gd_outputs = GD_model(**gd_inputs)
        dp_outputs = Dep_model(**dp_inputs)

    results = GD_processor.post_process_grounded_object_detection(
        gd_outputs,
        gd_inputs["input_ids"],
        box_threshold=0.35,
        text_threshold=0.3,
        target_sizes=target_sizes,
    )
    detections = DetectionResult.from_dict(results[0])

    # interpolate to original size and visualize the prediction
    post_processed_output = Dep_image_processor.post_process_depth_estimation(
        dp_outputs,
        target_sizes,
    )
    # print(type(post_processed_output))
    normalized_depths = []
    for i in range(len(post_processed_output)):
        depth = post_processed_output[i]["predicted_depth"]

        normalized_depth = (depth - depth.min()) / (
            depth.max() - depth.min()
        )  # Normalize to [0, 1]
        normalized_depth = normalized_depth * 255
        normalized_depths.append(normalized_depth)
    # normalized_depths = torch.stack(normalized_depths, dim=0)

    all_info = []
    for i in range(len(results)):
        json_string = get_text_info(results[i], normalized_depths[i])
        all_info.append(json_string)
    with torch.no_grad():
        inputs = Bert_tokenizer(
            all_info, return_tensors="pt", truncation=True, padding=True
        ).to(device)
        outputs = Bert_model(**inputs)
    # Shape: [batch_size, hidden_size]
    text_features = outputs.last_hidden_state.mean(dim=1)
    __json_out = []
    for i in range(len(all_info)):
        __json_out.append(
            {
                "object_info": all_info[i],
                "vector": text_features[i].cpu().tolist(),
                "text": gt[i],
                "image_path": index[i],
            }
        )

    if "test" not in basedir.name:
        id_json = {}
        for idx, out in enumerate(__json_out):
            task = ids[idx].split("_")[1]
            collection_name = f"{base_collection_name}_{task}"
            res = client.insert(
                collection_name=collection_name,
                data=[out],
            )
            id_json[out["image_path"]] = res["ids"][0]
    else:
        id_json = {}
        for i in range(len(__json_out)):
            id_json[index[i]] = __json_out[i]

    json_out.update(id_json)
    del id_json
    # store as pickle
    with open(info_store_path, "wb") as f:
        pickle.dump(json_out, f)

if "test" not in basedir.name:
    client.close()
    print("Milvus Client Closed")
