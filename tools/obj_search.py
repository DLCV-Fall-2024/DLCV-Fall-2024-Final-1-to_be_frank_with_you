from pymilvus import MilvusClient

client = MilvusClient("data/milvus_vector.db")
collection_name = "object_info"
tasks = ["general", "regional", "suggestion"]

client.drop_collection(collection_name)

for task in tasks:
    coll = f"{collection_name}_{task}"

    if not client.has_collection(coll):
        raise Exception(f"Collection {coll} does not exist")
#  /home/yck/cv/final/data/val/images/01839.jpg

import pickle

with open("data/train/obj_info.pkl", "rb") as f:
    target_data = pickle.load(f)

testdata = "data/train/images/00004.jpg"
# print(target_data.keys())
print(target_data[testdata])

data = target_data[testdata]
if isinstance(target_data[testdata], int):
    data = client.get(
        f"{collection_name}_general",
        target_data[testdata],
    )
    data = data[0]

res = client.search(
    f"{collection_name}_general",
    data=[data["vector"]],
    limit=4,
    output_fields=["text", "object_info", "image_path"],
)
print()
import json
import sys

print("target:", data["object_info"])
data.pop("vector")
out = {
    "target": data,
    "result": res,
}


with open("rag_test.json", "w") as f:
    json.dump(out, f, indent=4)


client.close()
