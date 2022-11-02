"""
Filename: get_images.py
-------------------------------------------------------------
This file imports a list of image names from a filefolder in numeric order.
by: Marcus Forst
"""

import os
import sort_nicely

def main(filefolder):
    """
    Grabs image names, sorts them, and puts them in a list.
    :param filefolder: string
    :return: images: list of images
    """
    images = [img for img in os.listdir(filefolder) if img.endswith(".tif") or img.endswith(
        ".tiff")]  # if this came out of moco the file suffix is .tif otherwise it's tiff
    sort_nicely.main(images)
    return images

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    main()
