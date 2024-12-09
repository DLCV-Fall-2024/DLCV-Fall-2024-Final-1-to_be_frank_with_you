import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
from datasets import load_dataset
from PIL.Image import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm

from utils.log import pretty_print


class DiscDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        transform=None,
        train=True,
        use_trainer=False,
        trainer_input_kwargs=None,
    ):

        # self.data = [
        #     transform(sample) if transform else sample for sample in raw_dataset
        # ]
        path = Path(path)
        img_dir = path / "images"
        assert (path / "config.json").exists(), f"config.json not found in {path}"
        assert (
            img_dir.exists() and img_dir.is_dir()
        ), f"images directory not found in {path}"

        self.config = json.load(open(path / "config.json", "r"))
        if isinstance(self.config, dict):
            self.prompts = self.config["prompts"]
            self.config = self.config["data"]

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((720, 720)),
                    transforms.ToTensor(),
                ]
            )

        self.is_train = train
        self.use_trainer = use_trainer
        self.trainer_input_kwargs = trainer_input_kwargs

    def __len__(self):
        return len(self.config)

    def __getitem__(self, idx):
        if self.use_trainer:
            return self.__trainer_getitem__(idx)
        item = self.config[idx]
        img = PIL.Image.open(item["img_path"]).convert("RGB")
        prompt = item["prompt"]
        if isinstance(prompt, int):
            prompt = self.prompts[prompt]
        try:
            inputs = self.transform(img, prompt=apply_chat_template(prompt))
            for key in inputs.keys():
                inputs[key] = inputs[key].squeeze(0)
        except Exception as e:
            img = self.transform(img)
            inputs = {
                "image": img,
                "prompt": apply_chat_template(prompt),
            }

        if self.is_train:
            inputs = {
                "image": inputs["image"],
                "prompt": f"{inputs['prompt']} {item['gt']}",
            }
        return (item["id"], inputs)

    def __trainer_getitem__(self, idx):
        item = self.config[idx]
        img = PIL.Image.open(item["img_path"]).convert("RGB")
        prompt = item["prompt"]
        if isinstance(prompt, int):
            prompt = self.prompts[prompt]
        inputs: dict = self.transform(img, prompt=apply_chat_template(prompt))
        for key in inputs.keys():
            inputs[key] = inputs[key].squeeze(0)

        inputs["labels"] = inputs["input_ids"].clone()
        inputs["id"] = item["id"]
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
