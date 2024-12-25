import json
import pickle

pairs = {
    "test": {
        "config_path": "data/test/config.json",
        "obj_info_path": "outputs/inference/inference_first_stage/1224_222940/first_stage.json",
    },
    "val": {
        "path1": "outputs/inference/inference_direct_val/1226_055740/submission.json",
        "path2": "outputs/inference/inference_direct_val/1226_055745/submission.json",
        "store_path": "outputs/inference/merged/1_1_direct/submission.json",
    },
}

task = "val"
path1 = pairs[task]["path1"]
path2 = pairs[task]["path2"]
store_path = pairs[task]["store_path"]

with open(path1, "r") as f:
    j1 = json.load(f)
with open(path2, "r") as f:
    j2 = json.load(f)

new_j = {**j1, **j2}

import os

os.makedirs(os.path.dirname(store_path), exist_ok=True)

with open(store_path, "w") as f:
    json.dump(new_j, f)
