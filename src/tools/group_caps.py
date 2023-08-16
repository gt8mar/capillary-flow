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
import re
import platform

def group_by_vid(vidlist):
    grouped = {}

    for file in vidlist:
        vmatch = re.search(r'vid(\d{2})', file)
        vidnum = vmatch.group(1)
        if vidnum in grouped:
            grouped[vidnum].append(file)
        else: grouped[vidnum] = [file]

    result = list(grouped.values())
    return result

#this function takes a segmented image of multiple capillaries and returns an array of images, each with one capillary
def get_single_caps(image):
    # Label connected components
    labeled_image = label(image, connectivity=2)
    
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

def separate_caps(registered_folder_fp):
    new_folder_fp = os.path.join(os.path.dirname(registered_folder_fp), "individual_caps_translated")
    os.makedirs(new_folder_fp, exist_ok=True)

    for vid in os.listdir(registered_folder_fp):
        if vid.endswith('.png'):
            individual_caps = get_single_caps(cv2.imread(os.path.join(registered_folder_fp, vid), cv2.IMREAD_GRAYSCALE))
            filenames = []
            for cap in individual_caps:
                renamed = False
                for row in range(cap.shape[0]):
                    if renamed == True: break
                    for col in range(cap.shape[1]):
                        if renamed == True: break
                        if cap[row][col] > 0:
                            for projcap in os.listdir(os.path.join(os.path.dirname(registered_folder_fp), "proj_caps")):
                                projcap_fp = os.path.join(os.path.dirname(registered_folder_fp), "proj_caps", projcap)
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

def save_untranslated(registered_folder_fp):
    indi_caps_fp = os.path.join(os.path.dirname(registered_folder_fp), "individual_caps_translated")
    translations_csv = os.path.join(os.path.dirname(registered_folder_fp), "translations.csv")
    crops_csv = os.path.join(os.path.dirname(registered_folder_fp), "crop_values.csv")
    resize_vals_csv = os.path.join(os.path.dirname(registered_folder_fp), "resize_vals.csv")

    orig_fp = os.path.join(os.path.dirname(registered_folder_fp), "individual_caps_original")
    os.makedirs(orig_fp, exist_ok=True)

    grouped_by_vid = group_by_vid(os.listdir(indi_caps_fp))

    with open(translations_csv, 'r') as translations:
        t_reader = csv.reader(translations)
        translated_rows = list(t_reader)

        with open(resize_vals_csv, 'r') as resizes:
            r_reader = csv.reader(resizes)
            resize_row = list(r_reader)[0]

            with open(crops_csv, 'r') as crops:
                c_reader = csv.reader(crops)
                crop_rows = list(c_reader)

                minx = abs(int(resize_row[0]))
                maxx = abs(int(resize_row[1]))
                miny = abs(int(resize_row[2]))
                maxy = abs(int(resize_row[3]))

                for i in range(len(grouped_by_vid)):
                    x, y = translated_rows[i] 
                    xint = int(float(x))
                    xint = xint
                    yint = int(float(y))
                    yint = yint

                    l, r, b, t = crop_rows[i]
                    lint = int(l)
                    rint = int(r)
                    bint = int(b)
                    tint = int(t)
                    
                    for cap in grouped_by_vid[i]:
                        img = cv2.imread(os.path.join(indi_caps_fp, cap), cv2.IMREAD_GRAYSCALE)
                        
                        ystart = miny + yint
                        yend = -(maxy - yint)
                        xstart = minx + xint
                        xend = -(maxx - xint)

                        ystart = None if ystart == 0 else ystart
                        yend = None if yend == 0 else yend
                        xstart = None if xstart == 0 else xstart
                        xend = None if xend == 0 else xend

                        untrans_img = img[ystart:yend, xstart:xend]
                        
                        bint = None if bint == 0 else bint
                        rint = None if rint == 0 else rint
                        crop_img = untrans_img[tint:bint, lint:rint]

                        cv2.imwrite(os.path.join(orig_fp, cap), crop_img)
            
def main(path="E:\\Marcus\\gabby_test_data\\part11\\230427\\loc02"):
    registered_fp = os.path.join(path, "segmented", "registered")
    sorted_seg_listdir = sorted(filter(lambda x: os.path.isfile(os.path.join(registered_fp, x)) and x.endswith('.png'), os.listdir(registered_fp)))
    
    #get translations
    translations_fp = os.path.join(os.path.dirname(registered_fp), "translations.csv")
    with open(translations_fp, 'r') as csv_file:
        reader = csv.reader(csv_file)
        translations = list(reader)

    #get max projection
    rows, cols = cv2.imread(os.path.join(registered_fp, sorted_seg_listdir[0]), cv2.IMREAD_GRAYSCALE).shape
    maxproject = np.zeros((rows, cols))
    for image in sorted_seg_listdir:
        maxproject += cv2.imread(os.path.join(registered_fp, image), cv2.IMREAD_GRAYSCALE)
    maxproject = np.clip(maxproject, 0, 255)
    
    #get array of images in which each image has 1 capillary (all frames projected on)
    caps = get_single_caps(maxproject)

    #save maxproj individual caps, named
    caps_fp = os.path.join(os.path.dirname(registered_fp), "proj_caps")
    os.makedirs(caps_fp, exist_ok=True)
    for x in range(len(caps)):
        filename = "cap_" + str(x).zfill(2) + ".png"
        cap_fp = os.path.join(caps_fp, filename)
        cv2.imwrite(str(cap_fp), caps[x])

    #save to results
    if platform.system() != 'Windows':
        pc_results_fp = "/hpc/projects/capillary-flow/results/segmented/proj_caps"
        os.makedirs(pc_results_fp, exist_ok=True)
        for x in range(len(caps)):
            filename = "cap_" + str(x).zfill(2) + ".png"
            cap_fp = os.path.join(pc_results_fp, filename)
            cv2.imwrite(str(cap_fp), caps[x])

    #save individual caps, named
    separate_caps(registered_fp)

    #save untranslated individual caps, named
    save_untranslated(registered_fp)
    

    



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