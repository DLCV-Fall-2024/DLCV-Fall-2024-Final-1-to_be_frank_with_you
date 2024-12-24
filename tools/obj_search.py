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

with open("data/test/obj_info.pkl", "rb") as f:
    target_data = pickle.load(f)

testdata = "data/test/images/00000.jpg"
print(target_data[testdata].keys())
print("target:", target_data[testdata]["object_info"])
res = client.search(
    f"{collection_name}_general",
    data=[target_data[testdata]["vector"]],
    limit=2,
    output_fields=["text", "object_info", "image_path"],
)
print()
import json
import sys

json.dump(res, sys.stdout, indent=4)
print()

client.close()
