"""
Filename: get_image_paths.py
-------------------------------------------------------------
This file imports a list of image paths from a filefolder in numeric order.
by: Marcus Forst
"""

import os
from src.tools import sort_nicely

def main(filefolder):
    """
    Grabs image names, sorts them, and puts them in a list.
    :param filefolder: string
    :return: images: list of images
    """
    images = []
    for img in os.listdir(filefolder):
        if img.endswith(".tif") or img.endswith(".tiff"):
            images.append(os.path.join(filefolder, img))
    return images

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    main()
