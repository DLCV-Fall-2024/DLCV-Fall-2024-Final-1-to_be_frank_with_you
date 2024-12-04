from datasets import load_dataset

from utils.log import pretty_print

splits = ["train", "val", "test"]

for split in splits:
    dataset = load_dataset(
        "ntudlcv/dlcv_2024_final1",
        # split=split,
        streaming=True,
    )
    print(f"=========Dataset: {split}=========")
    print(dataset)
    print(f"Columns: {dataset["val"]}")
    for item in dataset["val"]:
        pretty_print(item)
        break
    print("\n")
    break
