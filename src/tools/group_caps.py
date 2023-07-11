"""
Filename: group_caps.py
-------------------------------------------------------------
This file 
by: Gabby Rincon & ChatGPT
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label
from skimage.segmentation import find_boundaries

def get_single_caps(image):
    # Label connected components
    labeled_image = label(image, connectivity=2)
    print("labeled")
    plt.imshow(labeled_image, cmap='nipy_spectral')
    plt.colorbar()
    plt.title('Labeled Image')
    plt.show()
    
    # Find boundaries of connected components
    boundaries = find_boundaries(labeled_image, connectivity=2, mode='inner')
    
    min_pixel_count = 100
    # Filter components based on minimum pixel count
    component_mask = np.zeros_like(image, dtype=bool)
    for component_label in np.unique(labeled_image):
        if component_label == 0:
            continue
        component_pixels = labeled_image == component_label
        pixel_count = np.count_nonzero(component_pixels)
        if pixel_count >= min_pixel_count:
            component_mask |= component_pixels
    
    # Create an array for each filtered connected component
    component_arrays = []
    for component_label in np.unique(labeled_image[component_mask]):
        if component_label == 0:
            continue
        component_array = np.zeros_like(image)
        component_array[labeled_image == component_label] = image[labeled_image == component_label]
        component_arrays.append(component_array)
    
    return component_arrays


def main():
    seg_imgs_fp = "E:\\Marcus\\gabby test data\\part11_segmented\\registered"
    sorted_seg_listdir = sorted(filter(lambda x: os.path.isfile(os.path.join(seg_imgs_fp, x)), os.listdir(seg_imgs_fp))) #sort numerically

    maxproject = np.zeros((1080, 1440))
    for image in sorted_seg_listdir:
        maxproject += cv2.imread(os.path.join(seg_imgs_fp, image), cv2.IMREAD_GRAYSCALE)
    maxproject = np.clip(maxproject, 0, 255)
    
    caps = get_single_caps(maxproject)

    return caps
    

    



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