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

        return (item.id, inputs)

    def __trainer_getitem__(self, idx: int):
        item = self.config[idx]
        img = PIL.Image.open(item.img_path).convert("RGB")
        prompt = item.prompt
        if isinstance(prompt, int):
            prompt = self.prompts[prompt]

        inputs: dict = self.transform(img, prompt=apply_chat_template(prompt))
        for key in inputs.keys():
            inputs[key] = inputs[key].squeeze(0)

        inputs["labels"] = inputs["input_ids"].clone()
        inputs["id"] = item.id
        inputs.update(self.trainer_input_kwargs)

        return inputs


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


role_mapping = {"user": "USER", "assistant": "ASSISTANT"}


def apply_chat_template(conversation):
    return "USER: " + conversation + "ASSISTANT:"


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


import torch
from transformers import AutoModel, AutoProcessor


from src.models.encoder.segmentation import SegmentationEncoder
from functools import partial


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
