import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from src.utils.log import pretty_print

# Define the dataset and transformation

splits = ["train", "val", "test"]

dataset = load_dataset(
    "ntudlcv/dlcv_2024_final1",
    streaming=True,
)
print(f"=========Dataset=========")
print(dataset)
seed = 400
num = 10
row_per_task = 2
size = 256
transform = transforms.Compose(
    [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]
)

print(f"Columns: {dataset["train"]}")

tasks = ["general", "regional", "suggestion"]
tasks_count = [0, 0, 0]
tasks_image = [[], [], []]
prev = (0, 0, 0)
images = []

dataset["train"] = dataset["train"].shuffle(seed=seed)
for item in dataset["train"]:

    t = item["id"].split("_")[1]
    if t in item["id"] and tasks_count[tasks.index(t)] < num * row_per_task:
        tasks_count[tasks.index(t)] += 1
        tasks_image[tasks.index(t)].append(transform(item["image"]))
    num_of_images = sum([len(t_imgs) for t_imgs in tasks_image])
    if num_of_images == num * row_per_task * len(tasks):
        break

    if num_of_images % 10 == 0 and tuple(tasks_count) != prev:
        print(f"Tasks count: {tasks_count}")
        prev = tuple(tasks_count)
print(f"Done collecting images")
# Stack the images into a grid
images = tasks_image[0] + tasks_image[1] + tasks_image[2]
grid = make_grid(images, nrow=num, padding=2)

# Convert to a PIL Image and display or save
to_pil = ToPILImage()
grid_image = to_pil(grid)

grid_image.save("images/preview_grid.png")
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.show()
