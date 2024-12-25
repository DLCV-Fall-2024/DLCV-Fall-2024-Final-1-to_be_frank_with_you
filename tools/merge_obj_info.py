import json
import pickle

pairs = {
    "test": {
        "config_path": "data/test/config.json",
        "obj_info_path": "outputs/inference/inference_first_stage/1224_222940/first_stage.json",
    },
    "train": {
        "config_path": "data/train/config.json",
        # "obj_info_path": "data/train/config_aug.json",
        # "config_path": "data/train/config_aug.json",
        "obj_info_path": "outputs/inference/inference_first_stage/1225_184812/first_stage.json",
    },
    "val": {
        "config_path": "data/val/config.json",
        "obj_info_path": "outputs/inference/inference_first_stage_val/1226_034721/first_stage.json",
    },
}

task = "val"
config_path = pairs[task]["config_path"]
obj_info_path = pairs[task]["obj_info_path"]

with open(config_path, "r") as f:
    original = json.load(f)

if obj_info_path.endswith(".json"):
    with open(obj_info_path, "r") as f:
        obj_info = json.load(f)
    for item in original["data"]:
        if item["id"] in list(obj_info.keys()):
            item["features"]["object_info"] = obj_info[item["id"]]

if obj_info_path.endswith(".pkl"):
    with open(obj_info_path, "rb") as f:
        obj_info = pickle.load(f)
    for item in obj_info["data"]:
        # print(item["img_path"])
        # print(item["id"])
        # print(obj_info.keys())
        if item["img_path"] in obj_info.keys():
            item["features"]["object_info"] = obj_info[item["img_path"]]


with open(config_path, "w") as f:
    json.dump(original, f)
