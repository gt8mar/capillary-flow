"""
Filename: standard_dev.py
------------------------------------------------------
Compare standard deviations of each pixel.

By: Marcus Forst
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import time
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable


FILEFOLDER = 'C:\\Users\\gt8mar\\Desktop\\data\\221010\\vid4_moco'
BIN_FACTOR = 8
BORDER = 50

def tryint(s):
    try:
        return int(s)
    except:
        return s
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]
def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
def get_images(FILEFOLDER):
    """
    this function grabs image names, sorts them, and puts them in a list.
    :param FILEFOLDER: string
    :return: images: list of images
    """
    images = [img for img in os.listdir(FILEFOLDER) if img.endswith(".tif") or img.endswith(
        ".tiff")]  # if this came out of moco the file suffix is .tif otherwise it's tiff
    sort_nicely(images)
    return images
def load_image_array(image_list):
    """
    This function loads images into a numpy array.
    :param image_list: List of images
    :return: image_array: 3D numpy array
    """
    # Initialize array for images
    z_time = len(image_list)
    image_example = cv2.imread(os.path.join(FILEFOLDER, image_list[0]))
    rows, cols, layers = image_example.shape
    image_array = np.zeros((z_time, rows, cols), dtype='uint16')
    # loop to populate array
    for i in range(z_time):
        image_array[i] = cv2.imread(os.path.join(FILEFOLDER, image_list[i]), cv2.IMREAD_GRAYSCALE)
    return image_array

def main():
    # Import images
    images = get_images(FILEFOLDER)
    image_array = load_image_array(images)
    background = np.mean(image_array, axis=0)

    # Get rid of edges
    image_array_slice = image_array[:, BORDER:-BORDER, BORDER:-BORDER]
    standard_dev = np.std(image_array_slice, axis=0)

    # plot image background
    ax = plt.subplot()
    im = ax.imshow(standard_dev)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    return 0

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print(time.time() - ticks)
