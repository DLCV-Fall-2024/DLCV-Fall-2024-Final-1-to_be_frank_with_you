from typing import Dict, Optional, Tuple, Union, List, cast, Callable
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import json
from pathlib import Path

import PIL
import PIL.Image
from datasets import load_dataset
from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from utils.log import pretty_print


@dataclass_json
@dataclass
class DiscDatasetItem:
    id: str
    prompt: int | str
    img_path: str
    gt: Optional[str]


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
    ):
        path = Path(path)
        img_dir = path / "images"
        config_path = path / "config.json"
        assert config_path.exists(), f"config.json not found in {path}"
        assert (
            img_dir.exists() and img_dir.is_dir()
        ), f"images directory not found in {path}"

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
        assert (
            isinstance(transform, transforms.Compose) and not use_trainer
        ), "transform shouldn't be a Compose when using trainer"
        self.transform = cast(Transform, transform)

        self.is_train = train
        self.use_trainer = use_trainer
        self.trainer_input_kwargs = trainer_input_kwargs

    def __len__(self):
        return len(self.config)

    def __getitem__(self, idx: int):
        if self.use_trainer:
            return self.__trainer_getitem__(idx)

        item = self.config[idx]
        img = PIL.Image.open(item.img_path).convert("RGB")
        prompt = item.prompt
        if isinstance(prompt, int):
            prompt = self.prompts[prompt]

        if isinstance(self.transform, transforms.Compose):
            img = self.transform(img)
            inputs = {
                "image": img,
                "prompt": apply_chat_template(prompt),
            }
        else:
            inputs = self.transform(img, prompt=apply_chat_template(prompt))
            for key in inputs.keys():
                inputs[key] = inputs[key].squeeze(0)

        if self.is_train:
            inputs = {
                "image": inputs["image"],
                "prompt": f"{inputs['prompt']} {item['gt']}",
            }

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
        # pretty_print(inputs)

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


def preprocess_dataset(
    split: str = "all", cache_dir: Union[Path, str] = "./data"
) -> Tuple:
    assert split in VALID_SPLIT, f"split must be one of {VALID_SPLIT}"
    split = [split] if split != "all" else VALID_SPLIT[:-1]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)

    statistics = {}
    if (cache_dir / "statistics.json").exists():
        statistics = json.load(open(cache_dir / "statistics.json", "r"))
    for s in split:
        raw_dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=s)

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


if __name__ == "__main__":
    preprocess_dataset()
