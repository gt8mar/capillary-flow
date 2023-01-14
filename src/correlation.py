"""
Filename: correlation.py
------------------------------------------------------
Calculates the correlation between pixels and their nearest neighbors. 
By: Marcus Forst
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
from src.tools.get_shifts import get_shifts

BIN_FACTOR = 4

# SECTION_START = 138
# SECTION_END = 984

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

def main(SET = 'set_01', sample = 'sample_009', mask = False, verbose = True, write = False, bin_factor = 4):
    input_folder = os.path.join('C:\\Users\\ejerison\\capillary-flow\\data\\processed', str(SET), str(sample), 'B_stabilized')
    mask_folder = os.path.join('C:\\Users\\ejerison\\capillary-flow\\data\\processed', str(SET), str(sample), 'D_segmented')
    output_folder = os.path.join('C:\\Users\\ejerison\\capillary-flow\\data\\processed', str(SET), str(sample), 'G_correlation')
    images = get_images(os.path.join(input_folder, 'vid'))
    image_array = load_image_array(images, input_folder)
    gap_left, gap_right, gap_bottom, gap_top = get_shifts(input_folder)
    # Crop array based on shifts
    image_array = image_array[:, gap_top:gap_bottom, gap_left:gap_right] 
    # Read in the masks
    segmented = cv2.imread(os.path.join(mask_folder, f'{SET}_{sample}_background.png'), cv2.IMREAD_GRAYSCALE)
    # Make mask either 1 or 0
    segmented[segmented != 0] = 1
    if mask:
        image_array = segmented * image_array
    print(image_array.shape)

    # """Bin images to conserve memory and improve resolution"""
    image_array_binned = block_reduce(image_array, (2,2,2), func= np.mean)
    mask_binned = block_reduce(segmented, (2, 2), func=np.mean)

    print(image_array_binned.shape)
    print(mask_binned.shape)
    # plt.imshow(np.mean(image_array_binned, axis = 0))
    # plt.show()

    corr_x, corr_y = make_correlation_matrix(image_array_binned)
    # Plot a subset of this array
    corr_x_slice = corr_x[::bin_factor, ::bin_factor]
    corr_y_slice = corr_y[::bin_factor, ::bin_factor]
    # mask_slice = mask_binned[::bin_factor, ::bin_factor] 
    # mask_slice = mask_slice[:,:-1]  # this -1 is a result of the make_corr_matrix shift
    mask_slice = 1
    print(corr_y_slice.shape)

    # y, x = np.meshgrid(np.arange(0, rows // (BIN_FACTOR*2), 1), np.arange(0, cols // (BIN_FACTOR*2), 1))
    # plt.quiver(x, y, corr_x_slice, corr_y_slice)
    plt.quiver(corr_y_slice*mask_slice, corr_x_slice*mask_slice, angles = 'xy')
    plt.gca().invert_yaxis()
    if verbose:
        plt.show()
    if write:
        plt.imsave(os.path.join(output_folder, "correlation.png"))
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