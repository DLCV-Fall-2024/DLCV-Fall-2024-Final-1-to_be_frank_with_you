import os

from PIL import Image
from tqdm import tqdm


def convert_png_to_jpg(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in tqdm(list(os.listdir(input_folder))):
        if filename.endswith(".png"):
            png_path = os.path.join(input_folder, filename)
            jpg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(output_folder, jpg_filename)
            if os.path.exists(jpg_path):
                continue
            try:
                # Open the PNG image
                with Image.open(png_path) as img:
                    # Convert to RGB mode (JPG does not support transparency)
                    rgb_img = img.convert("RGB")
                    # Save as JPG
                    rgb_img.save(jpg_path, "JPEG", quality=95)
                    # print(f"Converted: {filename} -> {jpg_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")


# Define input and output folders
input_folder = "/home/yck/cv/final/tools/DiffBIR/results/images"
output_folder = "/home/yck/cv/final/tools/DiffBIR/results/images"

# Run the conversion
convert_png_to_jpg(input_folder, output_folder)
