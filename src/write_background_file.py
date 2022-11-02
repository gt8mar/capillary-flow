"""
Filename: write_background_file.py
------------------------------------------------------
This file takes a series of images, creates a background file, and creates a folder with
background subtracted files.
By: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
"""

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import cv2
import os
import re
import time

FILEFOLDER_PATH = 'C:\\Users\\gt8mar\\Desktop\\data\\221010\\vid4_moco'
DATE = "221010"
PARTICIPANT = "Participant3"

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

def main(folder_name = 'folder', filefolder_path = FILEFOLDER_PATH, date = DATE, participant = PARTICIPANT, verbose = False, subtracted = False):
    images = get_images(filefolder_path)
    # Here we make a list of image files
    image_files = []
    for i in range(len(images)):
        picture = np.array(cv2.imread(os.path.join(filefolder_path, images[i]), cv2.IMREAD_GRAYSCALE))
        # # This chops the image into smaller pieces (important if there has been motion correction)
        new_new_picture = picture[15:-25, 25:-25]
        # new_new_picture[new_new_picture > 5] = 5
        image_files.append(new_new_picture)
    image_files = np.array(image_files)
    ROWS, COLS = image_files[0].shape
    background = np.mean(image_files, axis=0)
    if verbose:
        ax = plt.subplot()
        im = ax.imshow(background)

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()


    """
    Extra functions
    -----------------------------------------------------------------------------------------
    """
    if subtracted:
        # Enhance contrast
        image_files = image_files - background
        print(np.max(image_files))
        print(np.min(image_files))

        image_files = image_files - np.min(image_files)
        image_files = image_files / np.max(image_files)
        image_files = np.array(image_files * 255, dtype=np.uint8)
        print('the following should never be less than 0')
        print(np.min(image_files))

    if verbose:
        # Plot with newly enhanced contrast
        ax = plt.subplot()
        im = ax.imshow(image_files[10])
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()


    # write new folder of reduced images:
    cwd = os.getcwd()
    folder = folder_name + "_background"
    path = os.path.join(cwd, folder)
    if folder not in os.listdir(cwd):
        os.mkdir(path)
    if subtracted:
        for i in range(len(image_files)):
            file = image_files[i]
            filename = images[i]

            # write to new folder
            cv2.imwrite(os.path.join(path, filename), file)

    # Add background file
    background = background.astype('uint8')
    bkgd_name = str(images[0].strip("."))
    bkgd_name += "_background"
    bkgd_name += ".tiff"
    cv2.imwrite(os.path.join(path, bkgd_name), background)
    return 0

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))