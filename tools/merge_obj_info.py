import json

with open("data/test/config.json", "r") as f:
    original = json.load(f)

obj_info = {}
obj_info_path = "outputs/inference/inference_first_stage/1224_222940/first_stage.json"
with open(obj_info_path, "r") as f:
    obj_info = json.load(f)

for item in original["data"]:
    item["features"]["object_info"] = obj_info[item["id"]]

with open("data/test/config.json", "w") as f:
    json.dump(original, f)
