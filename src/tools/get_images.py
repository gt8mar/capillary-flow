"""
Filename: get_images.py
-------------------------------------------------------------
This file imports a list of image names from a filefolder in numeric order.
by: Marcus Forst
"""

import os
from src.tools.sort_nicely import sort_nicely

def get_images(filefolder):
    """
    Grabs image names, sorts them, and puts them in a list.
    :param filefolder: string
    :return: images: list of image names
    """
    images = [img for img in os.listdir(filefolder) if img.endswith(".tif") or img.endswith(
        ".tiff")]  # if this came out of moco the file suffix is .tif otherwise it's tiff
    sort_nicely(images)
    return images

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    get_images()
