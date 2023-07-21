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
import csv

#this function takes a segmented image of multiple capillaries and returns an array of images, each with one capillary
def get_single_caps(image):
    # Label connected components
    labeled_image = label(image, connectivity=2)
    """plt.imshow(labeled_image, cmap='nipy_spectral')
    plt.colorbar()
    plt.title('Labeled Image')
    plt.show()"""
    
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

"""def number_caps(caps, seg_imgs_fp):
    sorted_seg_listdir = sorted(filter(lambda x: os.path.isfile(os.path.join(seg_imgs_fp, x)), os.listdir(seg_imgs_fp))) #sort numerically
    
    #read all segmented vids as arrays
    seg_imgs = []
    for image in sorted_seg_listdir:
        image = cv2.imread(os.path.join(seg_imgs_fp, image), cv2.IMREAD_GRAYSCALE)
        seg_imgs.append(image)

    #separate capillaries in seg_imgs
    individual_caps = []
    for vid in seg_imgs:
        individual_caps.append(get_single_caps(vid))
"""

def separate_caps(registered_folder_fp):
    new_folder_fp = os.path.join(registered_folder_fp, "individual_caps")
    os.makedirs(new_folder_fp, exist_ok=True)

    for vid in os.listdir(registered_folder_fp):
        if vid.endswith('.png'):
            """cv2.imshow("seg", cv2.imread(os.path.join(registered_folder_fp, vid)))
            cv2.waitKey(0)"""
            individual_caps = get_single_caps(cv2.imread(os.path.join(registered_folder_fp, vid), cv2.IMREAD_GRAYSCALE))
            filenames = []
            for cap in individual_caps:
                renamed = False
                for row in range(cap.shape[0]):
                    if renamed == True: break
                    for col in range(cap.shape[1]):
                        if renamed == True: break
                        if cap[row][col] > 0:
                            for projcap in os.listdir(os.path.join(registered_folder_fp, "proj_caps")):
                                projcap_fp = os.path.join(registered_folder_fp, "proj_caps", projcap)
                                if cv2.imread(projcap_fp, cv2.IMREAD_GRAYSCALE)[row][col] > 0:
                                    capnum = projcap[:-4] + "a" 
                                    counter = 0
                                    filename = os.path.join(new_folder_fp, vid[:-4] + "_" + capnum + ".png")
                                    while filename in filenames:
                                        counter += 1
                                        capnum = projcap[:-4] + chr(97 + counter) 
                                        filename = os.path.join(new_folder_fp, vid[:-4] + "_" + capnum + ".png")
                                    cv2.imwrite(filename, cap)
                                    filenames.append(filename)
                                    renamed = True
                                    break



            
def main():
    seg_imgs_fp = "E:\\Marcus\\gabby test data\\part11_segmented\\registered"
    sorted_seg_listdir = sorted(filter(lambda x: os.path.isfile(os.path.join(seg_imgs_fp, x)) and x.endswith('.png'), os.listdir(seg_imgs_fp)))
    
    #get translations
    translations_fp = os.path.join(seg_imgs_fp, "translations.csv")
    with open(translations_fp, 'r') as csv_file:
        reader = csv.reader(csv_file)
        translations = list(reader)

    #get max projection
    maxproject = np.zeros((1080, 1440))
    for image in sorted_seg_listdir:
        maxproject += cv2.imread(os.path.join(seg_imgs_fp, image), cv2.IMREAD_GRAYSCALE)
    maxproject = np.clip(maxproject, 0, 255)
    
    #get array of images in which each image has 1 capillary (all frames projected on)
    caps = get_single_caps(maxproject)

    caps_fp = os.path.join(seg_imgs_fp, "proj_caps")
    os.makedirs(caps_fp, exist_ok=True)
    for x in range(len(caps)):
        filename = "cap_" + str(x).zfill(2) + ".png"
        cap_fp = os.path.join(caps_fp, filename)
        cv2.imwrite(str(cap_fp), caps[x])

    separate_caps(seg_imgs_fp)
    

    



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