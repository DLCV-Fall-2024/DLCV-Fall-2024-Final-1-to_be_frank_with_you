import json
import pickle

arch = "default_1226_084508"

pairs = {
    "fs": {
        "path1": f"outputs/inference/inference_val_fs/{arch}/0/submission.json",
        "path2": f"outputs/inference/inference_val_fs/{arch}/1/submission.json",
        "store_path": f"outputs/inference/merged/{arch}_rag/submission.json",
    },
    "direct": {
        "path1": f"outputs/inference/inference_val_direct/{arch}/0/submission.json",
        "path2": f"outputs/inference/inference_val_direct/{arch}/1/submission.json",
        "store_path": f"outputs/inference/merged/{arch}_direct/submission.json",
    },
}

task = "fs"
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
