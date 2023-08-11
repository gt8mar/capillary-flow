"""
Filename: get_images.py
-------------------------------------------------------------
This file imports a list of image names from a filefolder in numeric order.
by: Marcus Forst
"""

import os
from src.tools.sort_nicely import sort_nicely

def get_images(filefolder, extension = 'tiff'):
    """
    Grabs image names, sorts them, and puts them in a list.
    :param filefolder: string
    :param extension: string. Choose "tiff" or "png" tiff works for 
    :return: images: list of image names
    """
    if extension == 'tiff':
        images = [img for img in os.listdir(filefolder) if img.endswith(".tif") or img.endswith(
            ".tiff")]  # if this came out of moco the file suffix is .tif otherwise it's tiff
    elif extension == 'png':
        images = [img for img in os.listdir(filefolder) if img.endswith(".png")]  # if this came out of moco the file suffix is .tif otherwise it's tiff
    elif extension == 'jpg':
        images = [img for img in os.listdir(filefolder) if img.endswith(".jpg")]  # if this came out of moco the file suffix is .tif otherwise it's tiff
    else:
        raise Exception('incorrect file extension')
    sort_nicely(images)
    return images

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    get_images()
