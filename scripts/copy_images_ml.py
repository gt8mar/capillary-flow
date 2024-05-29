"""
Filename: copy_images_ml.py
---------------------------
This script copies images larger than 256x256 pixels from a source folder to a target folder.

By: Marcus Forst
"""

import os
import shutil
from PIL import Image

def copy_large_images(source_folder):
    # Create the target folder based on the source folder name
    target_folder = f"big_{os.path.basename(source_folder)}"
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        
        # Check if the file is an image
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    
                    # Check if both dimensions are greater than 256 pixels
                    if width > 256 and height > 256:
                        # Copy the image to the target folder
                        shutil.copy(file_path, os.path.join(target_folder, filename))
            except IOError:
                # Handle cases where the file could not be opened as an image
                print(f"Skipping file {filename}, which could not be opened as an image.")
                
    print(f"Images larger than 256x256 have been copied to {target_folder}")

# Usage
source_folder = '/hpc/projects/capillary-flow/results/ML/kymographs'  # Replace 'path_to_your_folder' with the actual folder path
copy_large_images(source_folder)
