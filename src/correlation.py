"""
Filename: correlation.py
------------------------------------------------------
TBD
By: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import time
from PIL import Image
from skimage.measure import block_reduce
from src.tools.get_images import get_images
from src.tools.load_image_array import load_image_array

# FILEFOLDER = 'C:\\Users\\gt8mar\\Desktop\\data\\220513\\pointer2'
FILEFOLDER = 'C:\\Users\\gt8mar\\Desktop\\data\\221010\\stupid2'
BIN_FACTOR = 4

# SECTION_START = 138
# SECTION_END = 984

# Sort images first

def bin_image_by_2_space(image):
    return (image[:, :-1:2, :-1:2] + image[:, 1::2, :-1:2]
            + image[:, :-1:2, 1::2] + image[:, 1::2, 1::2])//4
def bin_image_by_2_time(image):
    return (image[:-1:2,:,:] + image[1::2, :, :])//2
def make_correlation_matrix(image_array_binned):
    # Initialize correlation matrix
    up_left = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, :-2, :-2]
    up_mid = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, :-2, 1:-1]
    up_right = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, :-2, 2:]
    center_right = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 1:-1, 2:]
    center_left = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 1:-1, :-2]
    down_right = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 2:, 2:]
    down_left = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 2:, :-2]
    down_mid = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 2:, 1:-1]

    """-----------------------------------------------------------------------------------------------------------------"""

    # total correlations
    corr_total_right = 707 * (up_right + down_right) // 1000 + center_right
    corr_total_left = 707 * (up_left + down_left) // 1000 + center_left
    corr_total_up = 707 * (up_left + up_right) // 1000 + up_mid
    corr_total_down = 707 * (down_left + down_right) // 1000 + down_mid

    # Now we need to make vectors for the correllations:
    corr_total_right_2D = np.mean(corr_total_right, axis=0)
    corr_total_left_2D = np.mean(corr_total_left, axis=0)
    corr_total_up_2D = np.mean(corr_total_up, axis=0)
    corr_total_down_2D = np.mean(corr_total_down, axis=0)
    corr_x = corr_total_right_2D - corr_total_left_2D
    # corr_x /= 5000
    corr_y = corr_total_up_2D - corr_total_down_2D
    # corr_y /= 5000
    return corr_x, corr_y

def main(filefolder = FILEFOLDER, bin_factor = BIN_FACTOR):
    images = get_images(filefolder)
    image_array = load_image_array(images)
    # image_array = image_array2[SECTION_START:SECTION_END]
    background = np.mean(image_array, axis=0)
    print(image_array.shape)

    # """Bin images to conserve memory and improve resolution"""
    image_array_binned = block_reduce(image_array, (2,2,2), func= np.mean)

    print(image_array_binned.shape)
    # plt.imshow(np.mean(image_array_binned, axis = 0))
    # plt.show()

    # max = np.max(image_array_binned)
    corr_x, corr_y = make_correlation_matrix(image_array_binned)
    # Plot a subset of this array
    corr_x_slice = corr_x[10:-10:bin_factor, 10:-10:bin_factor]
    corr_y_slice = corr_y[10:-10:bin_factor, 10:-10:bin_factor]
    print(corr_y_slice.shape)

    # y, x = np.meshgrid(np.arange(0, rows // (BIN_FACTOR*2), 1), np.arange(0, cols // (BIN_FACTOR*2), 1))
    # plt.quiver(x, y, corr_x_slice, corr_y_slice)
    plt.quiver(corr_y_slice, corr_x_slice, angles = 'xy')
    plt.gca().invert_yaxis()
    plt.show()
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