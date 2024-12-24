import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import PIL
import PIL.Image
import typer
from dataclasses_json import dataclass_json
from datasets import load_dataset
from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from src.utils import batch_feature_to_dict, container_cat, default, pad_sequences

PADDING_TOKEN = 32001
ATTENTION_MASK = 0


@dataclass_json
@dataclass
class DiscDatasetItem:
    id: str
    prompt: int | str
    gt: Optional[str]
    img_path: str
    features: Optional[dict] = None


@dataclass_json
@dataclass
class DiscDatasetConfig:
    data: List[DiscDatasetItem]
    prompts: List[str]


ProcessorTransform = Callable[[Image, str], dict]
Transform = transforms.Compose | ProcessorTransform


class DiscDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        transform: Optional[Transform] = None,
        train: bool = True,
        use_trainer: bool = False,
        trainer_input_kwargs: Optional[dict] = None,
        cache_dir: Optional[str] = ".cache",
        use_processed: bool = False,
        depth_model_id: str = "",
        segmentation_model_id: str = "",
    ):
        path = Path(path)
        img_dir = path / "images"
        config_path = path / "config.json"
        assert config_path.exists(), f"config.json not found in {path}"
        assert (
            img_dir.exists() and img_dir.is_dir()
        ), f"images directory not found in {path}"

        # NOTE: Only segmentation is supported for now
        self.processed_dir: Optional[Dict[str, Path]] = None
        self.process_model_id: Optional[Dict[str, str]] = None
        if use_processed:
            depth_model_id = depth_model_id.replace("/", "_")
            segmentation_model_id = segmentation_model_id.replace("/", "_")

            cache_dir = path / "features"
            depth_processed_dir = cache_dir / "depth" / depth_model_id
            segmentation_processed_dir = (
                cache_dir / "segmentation" / segmentation_model_id
            )

            depth_processed_enable = depth_processed_dir.exists()
            segmentation_processed_enable = segmentation_processed_dir.exists()

            # Check if config has features
            with open(config_path, "r") as f:
                config = cast(DiscDatasetConfig, DiscDatasetConfig.from_json(f.read()))

            # TODO: Reactivate this after the generation is done
            for item in config.data:
                if item.features is None:
                    assert False, f"processed images not found for item {item.id}"
                if (
                    depth_processed_enable
                    and item.features.get("depth", {})[depth_model_id] is None
                ):
                    assert False, f"depth processed image not found for item {item.id}"
                if (
                    segmentation_processed_enable
                    and item.features.get("segmentation", {})[segmentation_model_id]
                    is None
                ):
                    assert (
                        False
                    ), f"segmentation processed image not found for item {item.id}"
            self.processed_dir = {}
            self.process_model_id = {}

            if depth_processed_enable:
                self.processed_dir["depth"] = depth_processed_dir
                self.process_model_id["depth"] = depth_model_id
                print(f"Using processed images for depth in: {depth_model_id}")
            if segmentation_processed_enable:
                self.processed_dir["segmentation"] = segmentation_processed_dir
                self.process_model_id["segmentation"] = segmentation_model_id
                print(
                    f"Using processed images for segmentation in: {segmentation_model_id}"
                )

        with open(config_path, "r") as f:
            config = cast(DiscDatasetConfig, DiscDatasetConfig.from_json(f.read()))

        self.config = config.data
        self.prompts = config.prompts

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize((720, 720)),
                    transforms.ToTensor(),
                ]
            )
        assert not (
            isinstance(transform, transforms.Compose) and use_trainer
        ), "transform shouldn't be a Compose when using trainer"
        self.transform = cast(Transform, transform)

        self.is_train = train
        self.use_trainer = use_trainer
        self.trainer_input_kwargs = trainer_input_kwargs

        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def __len__(self):
        return len(self.config)

    def __getitem__(self, idx: int):
        if self.use_trainer:
            return self.__trainer_getitem__(idx)

        item = self.config[idx]

        prompt = item.prompt
        if isinstance(prompt, int):
            prompt = self.prompts[prompt]

        prompt = apply_chat_template(prompt)
        if self.is_train:
            prompt = f"{prompt} {item.gt}"

        image = PIL.Image.open(item.img_path).convert("RGB")

        processed_images = {}
        if self.processed_dir:
            features = item.features
            for key, dir in self.processed_dir.items():
                model_id = self.process_model_id[key]
                processed_image_path = features[key][model_id]
                processed_images[key] = PIL.Image.open(processed_image_path).convert(
                    "RGB"
                )

        # NOTE: Currently, we directly use given transform for better performance
        # if isinstance(self.transform, transforms.Compose):
        #     transformed_img = self.transform(img)
        #     inputs = {
        #         "image": transformed_img,
        #         "prompt": prompt,
        #     }
        # else:
        #     inputs = self.transform(img, prompt=prompt)
        inputs = self.transform(image, processed_images=processed_images, prompt=prompt)

        # Otherwise, workers will open too many files
        image.close()
        for processed_image in processed_images.values():
            processed_image.close()
        # for key in inputs.keys():
        #     if isinstance(inputs[key], torch.Tensor):
        #         inputs[key] = inputs[key].squeeze(0)
        return (item.id, inputs)


import pickle

from pymilvus import MilvusClient


class RAGDataset(Dataset):

    def __init__(
        self,
        path: Union[str, Path],
        client: MilvusClient,
        transform: Optional[Transform] = None,
    ):
        path = Path(path)
        img_dir = path / "images"
        config_path = path / "config.json"
        obj_path = path / "obj_info.pkl"
        assert config_path.exists(), f"config.json not found in {path}"
        assert (
            img_dir.exists() and img_dir.is_dir()
        ), f"images directory not found in {path}"

        with open(config_path, "r") as f:
            config = cast(DiscDatasetConfig, DiscDatasetConfig.from_json(f.read()))

        with open(obj_path, "rb") as f:
            target_data = pickle.load(f)

        self.obj_info = target_data

        self.config = config.data
        self.prompts = config.prompts
        self.sys_prompt = (
            "You are an AI model designed to analyze traffic scenes and provide a detailed "
            "interpretation of the image based on provided detected objects information.\n\n"
            "Below are examples:\n\n"
        )
        self.usr_prompt = (
            "Now, analyze the following scene:\n\n"
            "Objects: {objects_text}\n"
            "Based on this information, generate a response similar to the example above"
        )

        self.client = client
        self.transform = cast(Transform, transform)

    def __len__(self):
        return len(self.config)

    def __getitem__(self, idx: int):
        item = self.config[idx]
        id = item.id
        prompt = item.prompt
        img_path = item.img_path
        if isinstance(prompt, int):
            prompt = self.prompts[prompt]

        my_info = self.obj_info[img_path]["object_info"]

        res = self.client.search(
            f"object_info_{id.split("_")[1]}",
            data=[self.obj_info[img_path]["vector"]],
            limit=2,
            output_fields=["text", "object_info", "image_path"],
        )
        res = res[0]
        raw_prompt = [{"role": "system", "prompt": self.sys_prompt}]
        for item in res:
            raw_prompt.append(
                {"role": "user", "prompt": f"Objects: {item["entity"]["object_info"]}"}
            )
            raw_prompt.append({"role": "assistant", "prompt": item["entity"]["text"]})
        raw_prompt.append(
            {"role": "user", "prompt": self.usr_prompt.format(objects_text=my_info)}
        )
        prompt = apply_chat_template(raw_prompt)

        # (image, prompt)
        inputs = self.transform(None, prompt)
        return id, inputs


def rag_collate_fn(batch):
    # From List[BatchFeature] to List[Dict]
    # index, inputs
    indexs = [item[0] for item in batch]
    inputs = [batch_feature_to_dict(item[1]) for item in batch]
    inputs_keys = inputs[0].keys()

    merged_data = {}
    input_ids = [elem["input_ids"] for elem in inputs]
    # token_type_ids = [elem["token_type_ids"] for elem in inputs]
    attention_mask = [elem["attention_mask"] for elem in inputs]
    padded_input_ids = pad_sequences(input_ids, PADDING_TOKEN)
    # padded_token_type_ids = pad_sequences(token_type_ids, PADDING_TOKEN)
    padded_attention_mask = pad_sequences(attention_mask, ATTENTION_MASK)

    # Update the batch with padded input_ids
    for i, datamap in enumerate(inputs):
        datamap["input_ids"] = padded_input_ids[i]
        # datamap["token_type_ids"] = padded_token_type_ids[i]
        datamap["attention_mask"] = padded_attention_mask[i]

    for i, key in enumerate(inputs_keys):
        data_list = [datamap[key] for datamap in inputs]
        data_list = container_cat(data_list, dim=0)
        merged_data[key] = data_list

    return indexs, merged_data


class BoxInfoPreProcessorDataset(Dataset):

    def __init__(
        self,
        path: Union[str, Path],
        GD_processor: Transform,
        Dep_image_processor: Transform,
        labels: str,
    ):
        path = Path(path)
        config_path = path / "config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        self.data = []
        for item in config["data"]:
            self.data.append(item)
        self.GD_processor = GD_processor
        self.Dep_image_processor = Dep_image_processor
        self.label = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        id = item["id"]
        gt = item["gt"]
        if gt is None:
            gt = ""
        img_path = item["img_path"]
        image = PIL.Image.open(str(img_path)).convert("RGB")
        gd_inputs = self.GD_processor(
            image,
            text=self.label,
            return_tensors="pt",
            do_resize=True,
            size={"width": 1024, "height": 720},
        )
        dp_inputs = self.Dep_image_processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            keep_aspect_ratio=False,
            # size={"height": 384, "width": 384},
        )

        shape = image.size[::-1]
        image.close()

        return (str(img_path), gd_inputs, dp_inputs, gt, shape, str(id))


def preprocessor_collate_fn(batch):
    # From List[BatchFeature] to List[Dict]
    # index, gd_inputs, dp_inputs, gt, target_sizes
    indexs = [item[0] for item in batch]
    gd_inputs = [batch_feature_to_dict(item[1]) for item in batch]
    dp_inputs = [batch_feature_to_dict(item[2]) for item in batch]
    gts = [item[3] for item in batch]
    target_sizes = [item[4] for item in batch]
    ids = [item[5] for item in batch]

    # ['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask']
    gd_inputs_keys = gd_inputs[0].keys()
    # ['pixel_values']
    dp_inputs_keys = dp_inputs[0].keys()

    merged_gd_data = {}
    input_ids = [elem["input_ids"] for elem in gd_inputs]
    token_type_ids = [elem["token_type_ids"] for elem in gd_inputs]
    attention_mask = [elem["attention_mask"] for elem in gd_inputs]
    padded_input_ids = pad_sequences(input_ids, PADDING_TOKEN)
    padded_token_type_ids = pad_sequences(token_type_ids, PADDING_TOKEN)
    padded_attention_mask = pad_sequences(attention_mask, ATTENTION_MASK)

    # Update the batch with padded input_ids
    for i, datamap in enumerate(gd_inputs):
        datamap["input_ids"] = padded_input_ids[i]
        datamap["token_type_ids"] = padded_token_type_ids[i]
        datamap["attention_mask"] = padded_attention_mask[i]

    for i, key in enumerate(gd_inputs_keys):
        data_list = [datamap[key] for datamap in gd_inputs]
        data_list = container_cat(data_list, dim=0)
        merged_gd_data[key] = data_list

    merged_dp_data = {}

    for i, key in enumerate(dp_inputs_keys):
        data_list = [datamap[key] for datamap in dp_inputs]
        data_list = container_cat(data_list, dim=0)
        merged_dp_data[key] = data_list

    # Return the updated batch
    return indexs, merged_gd_data, merged_dp_data, gts, target_sizes, ids


VALID_SPLIT = ["train", "val", "test", "all"]


def update_stats(stats: Dict, key: str, value: Union[int, float, str]):
    if isinstance(value, str):
        value = len(value)

    if key not in stats:
        stats[key] = {"max": value, "min": value, "avg": value, "num": 1}
    else:
        keys = ["max", "min", "avg", "num"]
        for k in keys:
            if k not in stats[key]:
                stats[key][k] = 0
                if k == "min":
                    stats[key][k] = 2**15
        stats[key]["max"] = max(stats[key]["max"], value)
        stats[key]["min"] = min(stats[key]["min"], value)
        stats[key]["avg"] += value
        stats[key]["num"] += 1


def compress_config(config_path: Union[str, Path]):

    config = json.load(open(config_path, "r"))
    if isinstance(config, dict):
        json.dump(
            config,
            open(config_path, "w"),
        )
        return
    prompt_type = {}
    prompts = []
    for item in config:
        prompt = item["prompt"]
        if prompt not in prompt_type:
            prompts.append(prompt)
            prompt_type[prompt] = len(prompt_type)
        item["prompt"] = prompt_type[prompt]
    json.dump(
        {
            "data": config,
            "prompts": prompts,
        },
        open(config_path, "w"),
    )


role_mapping = {"system": "USER", "user": "USER", "assistant": "ASSISTANT"}


def apply_chat_template(conversation):
    if isinstance(conversation, list):
        return __apply_chat_template_list(conversation)
    elif isinstance(conversation, str):
        return "USER: " + conversation + "ASSISTANT:"
    elif isinstance(conversation, dict):
        assert "role" in conversation, f"role not found in conversation {conversation}"
        assert (
            "prompt" in conversation
        ), f"text not found in conversation {conversation}"
        role = conversation["role"].lower()
        assert role in role_mapping, f'"{role}" not found in mapping'
        return role_mapping[role] + ": " + conversation["prompt"]
    else:
        raise ValueError(f"Invalid conversation type: {type(conversation)}")


def __apply_chat_template_list(conversation):

    out = ""
    for i, conv in enumerate(conversation):
        assert "role" in conv, f"role not found in conversation {conv}"
        assert "prompt" in conv, f"text not found in conversation {conv}"
        role = conv["role"].lower()
        assert role in role_mapping, f'"{role}" not found in mapping'
        out += role_mapping[role] + ": " + conv["prompt"]
        out += "\n"
    out += "ASSISTANT:"
    return out


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("preprocess")
def preprocess_dataset(
    split: str = typer.Option("all", help="Split to preprocess"),
    cache_dir: str = typer.Option("./data", help="Cache directory"),
) -> Tuple:
    assert split in VALID_SPLIT, f"split must be one of {VALID_SPLIT}"
    split = [split] if split != "all" else VALID_SPLIT[:-1]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)

    statistics = {}
    if (cache_dir / "statistics.json").exists():
        statistics = json.load(open(cache_dir / "statistics.json", "r"))
    for s in split:
        raw_dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=s, streaming=True)

        save_dir = cache_dir / s
        (save_dir / "images").mkdir(exist_ok=True, parents=True)

        config = []
        stats = {
            "num": 0,
            "image_stats": {},
        }

        print(f"Processing split: {s}")
        for i, item in tqdm(enumerate(raw_dataset)):
            save_path = save_dir / "images" / f"{i:05d}.jpg"
            img: Image = item["image"]
            if not save_path.exists():
                img.save(save_path)
            update_stats(stats["image_stats"], "width", img.width)
            update_stats(stats["image_stats"], "height", img.height)
            gt = None
            prompt = None
            chat = item["conversations"]
            assert len(chat) <= 2, f"chat length is {len(chat)}, chat: {chat}"
            if len(chat) > 0:
                assert chat[0]["from"] == "human"
                prompt = chat[0]["value"]
                update_stats(stats, "prompt_stats", prompt)

            if len(chat) > 1:
                assert chat[1]["from"] == "gpt"
                gt = chat[1]["value"]
                update_stats(stats, "gt_stats", gt)

            config.append(
                {
                    "id": item["id"],
                    "prompt": prompt,
                    "gt": gt,
                    "img_path": str(save_path),
                }
            )
        stats["num"] = len(config)
        stats["prompt_stats"]["avg"] /= stats["prompt_stats"]["num"]
        stats["image_stats"]["width"]["avg"] /= stats["num"]
        stats["image_stats"]["height"]["avg"] /= stats["num"]
        if "gt_stats" in stats and stats["gt_stats"]["num"] > 0:
            stats["gt_stats"]["avg"] /= stats["gt_stats"]["num"]
        else:
            stats["gt_stats"] = None
        statistics.update({s: stats})
        json.dump(config, open(save_dir / "config.json", "w"))
        json.dump(statistics, open(cache_dir / "statistics.json", "w"), indent=4)
        compress_config(save_dir / "config.json")


from functools import partial

import torch
from transformers import AutoModel, AutoProcessor

from src.models.encoder.segmentation import SegmentationEncoder


def get_segmentaion_processor(
    model_id: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16
):
    segmentation_encoder = SegmentationEncoder(
        model_id,
        segment_type="panoptic",
        # segment_type="semantic",
        image_target_size=(800, 1200),
        vision_feature_layer=-2,
        device=device,
        torch_dtype=torch_dtype,
    )
    for p in segmentation_encoder.parameters():
        p.requires_grad = False
    segmentation_encoder.eval()

    def processor(batch_img):
        processed = segmentation_encoder.task_processor(
            batch_img,
            return_tensors="pt",  # return as pytorch tensors
            padding=True,
            do_rescale=True,
        )
        processed.to(device="cuda", dtype=torch.float16)

        output = segmentation_encoder(**processed)
        preds = output["predictions"]
        preds = preds.detach().cpu()  # [1, 3, 800, 1200]
        # convert back to PIL
        preds = [
            PIL.Image.fromarray(pred.permute(1, 2, 0).to(torch.uint8).numpy())
            for pred in preds
        ]

        return preds

    return processor


SETUP_MODEL_FUNC = {
    "segmentation": get_segmentaion_processor,
}


# NOTE: The is actually only for segmentation
@app.command("extract_processed")
def extract_processed_images(
    split: str = typer.Option("all", help="Split to preprocess"),
    cache_dir: str = typer.Option("./data", help="Cache directory"),
    feature_name: str = typer.Option("segmentation", help="Feature name"),
    model_id: str = typer.Option(
        "shi-labs/oneformer_ade20k_dinat_large", help="Encoder id"
    ),
    batch_size: int = typer.Option(6, help="Batch size"),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    assert split in VALID_SPLIT, f"split must be one of {VALID_SPLIT}"
    split = [split] if split != "all" else VALID_SPLIT[:-1]

    cache_dir = Path(cache_dir)
    assert cache_dir.exists(), f"cache directory {cache_dir} not found"

    stat_path = cache_dir / "statistics.json"
    assert stat_path.exists(), f"statistics.json not found in {cache_dir}"
    statistics = json.load(open(stat_path, "r"))

    # Check if splits exist
    for s in split:
        split_dir = cache_dir / s
        assert split_dir.exists(), f"split directory {split_dir} not found"
        config_path = split_dir / "config.json"
        assert config_path.exists(), f"config.json not found in {split_dir}"

    # Load model
    print(f"Loading model {model_id}")
    processor = SETUP_MODEL_FUNC[feature_name](
        model_id, device, torch_dtype=torch.float16
    )

    model_name = model_id.replace("/", "_")

    for s in split:
        split_dir = cache_dir / s
        config_path = split_dir / "config.json"
        config = json.load(open(config_path, "r"))

        feature_dir = split_dir / "features" / feature_name / model_name
        feature_dir.mkdir(exist_ok=True, parents=True)

        batch_index = []
        batch_img = []
        batch_name = []

        data = config["data"]
        for i, item in enumerate(tqdm(data)):
            img_path = item["img_path"]
            img_name = Path(img_path).stem

            img = PIL.Image.open(img_path).convert("RGB")

            batch_index.append(i)
            batch_img.append(img)
            batch_name.append(img_name)

            if len(batch_index) == batch_size or i == len(data) - 1:
                # processed = processor(
                #     batch_img,
                #     return_tensors="pt",  # return as pytorch tensors
                #     padding=True,
                #     do_rescale=True,
                # )
                # processed.to(device="cuda", dtype=torch.float16)

                # output = model(**processed)
                # preds = output["predictions"]
                # preds = preds.detach().cpu()  # [1, 3, 800, 1200]
                # # convert back to PIL
                # preds = [
                #     PIL.Image.fromarray(pred.permute(1, 2, 0).to(torch.uint8).numpy())
                #     for pred in preds
                # ]
                processed_images = processor(batch_img)

                for i, processed_image, name in zip(
                    batch_index, processed_images, batch_name
                ):
                    processed_image_path = feature_dir / f"{name}.jpg"
                    processed_image.save(processed_image_path)

                    if "features" not in data[i]:
                        data[i]["features"] = {}
                    if feature_name not in data[i]["features"]:
                        data[i]["features"][feature_name] = {}
                    data[i]["features"][feature_name][model_name] = str(
                        processed_image_path
                    )

                batch_index = []
                batch_img = []
                batch_name = []

                for img in batch_img:
                    img.close()

                del processed_images

        json.dump(config, open(config_path, "w"), indent=4)


def process_output(output, vision_feature_layer: int = -2):
    hidden_states = output.hidden_states
    features = hidden_states[vision_feature_layer]
    return features


def process_depth_output(output, vision_feature_layer: int = -2):
    raise NotImplementedError("Depth output processing not implemented")


def process_segmentation_output(output, vision_feature_layer: int = -2):
    raise NotImplementedError("Segmentation output processing not implemented")


PROCESS_OUTPUT_FUNC = {
    "default": process_output,
    "depth": process_depth_output,
    "segmentation": process_segmentation_output,
}


@app.command("extract")
def extract_image_features(
    split: str = typer.Option("all", help="Split to preprocess"),
    cache_dir: str = typer.Option("./data", help="Cache directory"),
    vision_feature_layer: int = typer.Option(-2, help="Vision feature layer"),
    feature_name: str = typer.Option("default", help="Feature name"),
    model_id: str = typer.Option("facebook/dinov2-large", help="Encoder id"),
    batch_size: int = typer.Option(16, help="Batch size"),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    assert split in VALID_SPLIT, f"split must be one of {VALID_SPLIT}"
    split = [split] if split != "all" else VALID_SPLIT[:-1]

    cache_dir = Path(cache_dir)
    assert cache_dir.exists(), f"cache directory {cache_dir} not found"

    stat_path = cache_dir / "statistics.json"
    assert stat_path.exists(), f"statistics.json not found in {cache_dir}"
    statistics = json.load(open(stat_path, "r"))

    # Check if splits exist
    for s in split:
        split_dir = cache_dir / s
        assert split_dir.exists(), f"split directory {split_dir} not found"
        config_path = split_dir / "config.json"
        assert config_path.exists(), f"config.json not found in {split_dir}"

    # Load model
    print(f"Loading model {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model.to(device)
    print(f"Model loaded: {model}")

    model_name = model_id.replace("/", "_")

    transform = transforms.Compose(
        [
            transforms.Resize((720, 720)),
            transforms.ToTensor(),
        ]
    )

    for s in split:
        split_dir = cache_dir / s
        config_path = split_dir / "config.json"
        config = json.load(open(config_path, "r"))

        feature_dir = split_dir / "features" / feature_name / model_name
        feature_dir.mkdir(exist_ok=True, parents=True)

        batch_index = []
        batch_img = []
        batch_name = []

        data = config["data"]
        for i, item in enumerate(tqdm(data)):
            img_path = item["img_path"]
            img_name = Path(img_path).stem

            img = PIL.Image.open(img_path).convert("RGB")
            img = transform(img)
            batch_index.append(i)
            batch_img.append(img)
            batch_name.append(img_name)

            if len(batch_index) == batch_size or i == len(data) - 1:
                inputs = processor(
                    batch_img, return_tensors="pt", padding=True, do_rescale=False
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                out = model.forward(**inputs, output_hidden_states=True)
                process_func = PROCESS_OUTPUT_FUNC[feature_name]
                features = process_func(out, vision_feature_layer)
                features = features.detach().cpu()

                for i, feature, name in zip(batch_index, features, batch_name):
                    feature_path = feature_dir / f"{name}.pt"
                    torch.save(feature, feature_path)

                    if "features" not in data[i]:
                        data[i]["features"] = {}
                    if feature_name not in data[i]["features"]:
                        data[i]["features"][feature_name] = {}
                    data[i]["features"][feature_name][model_name] = str(feature_path)

                batch_index = []
                batch_img = []
                batch_name = []

        json.dump(config, open(config_path, "w"), indent=4)


if __name__ == "__main__":
    app()
