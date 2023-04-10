"""
Filename: tiff_to_png.py
------------------------
By: chatgpt and marcus forst
"""

from PIL import Image
import os

# define input and output directories
input_dir = "D:\\Marcus\\train_backgrounds_export"
output_dir = "D:\\Marcus\\train_backgrounds_export_png"

# loop through all files in input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".tiff") or file_name.endswith(".tif"):
        # open TIFF file and convert to PNG
        with Image.open(os.path.join(input_dir, file_name)) as im:
            # save PNG with minimal compression
            png_filename = os.path.splitext(file_name)[0] + ".png"
            im.save(os.path.join(output_dir, png_filename), format="PNG", compress_level=1)