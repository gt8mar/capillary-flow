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
import re
import platform
import pandas as pd
import itertools
import csv
if platform.system() == 'Windows':
    from enumerate_capillaries2 import find_connected_components
else:
    from src.tools.enumerate_capillaries2 import find_connected_components

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

def separate_caps(registered_folder_fp):
    new_folder_fp = os.path.join(os.path.dirname(registered_folder_fp), "individual_caps_translated")
    os.makedirs(new_folder_fp, exist_ok=True)

    #save individual caps, named
    for vid in os.listdir(registered_folder_fp):
        if vid.endswith('.png'):
            # Create a list of individual caps (contours)
            individual_caps = find_connected_components(cv2.imread(os.path.join(registered_folder_fp, vid), cv2.IMREAD_GRAYSCALE)) 
            filenames = []
            # Iterate through each capillary contour
            for cap in individual_caps:
                renamed = False
                #iterate through each pixel in cap
                for row, col in itertools.product(range(cap.shape[0]), range(cap.shape[1])):
                    #if cap already found, break
                    if renamed == True: 
                        break
                    # if pixel is part of a capillary, then name it based on its position and the
                    # positions of capillaries before it. 
                    # Ex: check through pixels until you get a hit. Then check each projected capillary to see if
                    # that pixel is part of it. If it is, then name it based on the projected capillary's name. 
                    # If the projected capillary has already claimed another capillary, then add a letter to the end. 
                    if cap[row][col] > 0:
                        for projcap in os.listdir(os.path.join(os.path.dirname(registered_folder_fp), "proj_caps")):
                            projcap_fp = os.path.join(os.path.dirname(registered_folder_fp), "proj_caps", projcap)
                            if cv2.imread(projcap_fp, cv2.IMREAD_GRAYSCALE)[row][col] > 0:
                                capnum = projcap.replace('.png', '') + "a" 
                                counter = 0
                                filename = os.path.join(new_folder_fp, vid[:-4] + "_" + capnum + ".png")
                                #name fragments b, c, d, etc. if capnum already exists
                                while filename in filenames:
                                    counter += 1
                                    capnum = projcap[:-4] + chr(97 + counter) 
                                    filename = os.path.join(new_folder_fp, vid[:-4] + "_" + capnum + ".png")
                                cv2.imwrite(filename, cap)
                                filenames.append(filename)
                                renamed = True
                                break

def save_untranslated(registered_folder_fp):
    participant = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(registered_folder_fp))))))
    date = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(registered_folder_fp)))))
    location = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(registered_folder_fp))))
    

    indi_caps_fp = os.path.join(os.path.dirname(registered_folder_fp), "individual_caps_translated")
    translations_csv = os.path.join(os.path.dirname(registered_folder_fp), "translations.csv")
    crops_csv = os.path.join(os.path.dirname(registered_folder_fp), "crop_values.csv")
    resize_vals_csv = os.path.join(os.path.dirname(registered_folder_fp), "resize_vals.csv")

    orig_fp = os.path.join(os.path.dirname(registered_folder_fp), "individual_caps_original")
    os.makedirs(orig_fp, exist_ok=True)

    indi_caps_listdir = os.listdir(indi_caps_fp)
    sorted_indi_caps = sorted(indi_caps_listdir, key=lambda x: int(re.search(r'vid(\d{2})', x).group(1)))

    grouped_by_vid = group_by_vid(sorted_indi_caps)

    translated_df = pd.read_csv(translations_csv, header=None)
    translated_rows = translated_df.values.tolist()

    resize_df = pd.read_csv(resize_vals_csv, header=None)
    resize_row = resize_df.values.tolist()[0]

    crops_df = pd.read_csv(crops_csv, header=None)
    crop_rows = crops_df.values.tolist()

    minx = abs(int(resize_row[0]))
    maxx = abs(int(resize_row[1]))
    miny = abs(int(resize_row[2]))
    maxy = abs(int(resize_row[3]))

    saved_files = []

    for i in range(len(grouped_by_vid)):
        x, y = translated_rows[i]
        xint = int(float(x))
        yint = int(float(y))

        l, r, b, t = crop_rows[i]
        lint = int(l)
        rint = int(r)
        bint = int(b)
        tint = int(t)

        for cap in grouped_by_vid[i]:
            img = cv2.imread(os.path.join(indi_caps_fp, cap), cv2.IMREAD_GRAYSCALE)

            ystart = maxy - yint
            yend = -(miny + yint)
            xstart = maxx - xint
            xend = -(minx + xint)

            ystart = None if ystart == 0 else ystart
            yend = None if yend == 0 else yend
            xstart = None if xstart == 0 else xstart
            xend = None if xend == 0 else xend

            untrans_img = img[ystart:yend, xstart:xend]

            bint = None if bint == 0 else bint
            rint = None if rint == 0 else rint
            crop_img = untrans_img[tint:bint, lint:rint]

            save_path = os.path.join(orig_fp, cap)
            cv2.imwrite(save_path, crop_img)
            saved_files.append(cap)

    # Save the file names to a CSV file in folder
    file_names_csv = os.path.join(orig_fp, participant + '_' + date + '_' + location + "_cap_names.csv")
    with open(file_names_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File Name"])
        for file_name in saved_files:
            writer.writerow([file_name])

    # Save the CSV file to results folder
    if platform.system() != 'Windows':
        results_fp = '/hpc/projects/capillary-flow/results/size/name_csvs'
        os.makedirs(results_fp, exist_ok=True)
        file_names_csv = os.path.join(results_fp, participant + '_' + date + '_' + location + "_cap_names.csv")
        with open(file_names_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["File Name"])
            for file_name in saved_files:
                writer.writerow([file_name])
            
def main(path="f:\\Marcus\\data\\part30\\231130\\loc02"):
    """
    Groups capillaries from different videos and saves individual caps with names based on
    this grouping.

    Args:
        path (str): The path to the location directory for a given participant. 
            This will be used to load in registered segmentation files. 

    Returns:
        0, if run correctly
        
    Saves:
        - Individual caps as separate images in the 'proj_caps' directory.
        - Individual caps as separate images in the 'registered' directory.
        - Untranslated individual caps as separate images in the 'registered' directory.
    """
    registered_fp = os.path.join(path, "segmented", "hasty", "registered")
    
    sorted_seg_listdir = sorted(filter(lambda x: os.path.isfile(os.path.join(registered_fp, x)) and x.endswith('.png'), os.listdir(registered_fp)))

    #get max projection
    rows, cols = cv2.imread(os.path.join(registered_fp, sorted_seg_listdir[0]), cv2.IMREAD_GRAYSCALE).shape
    maxproject = np.zeros((rows, cols))
    for image in sorted_seg_listdir:
        maxproject += cv2.imread(os.path.join(registered_fp, image), cv2.IMREAD_GRAYSCALE)
    maxproject = np.clip(maxproject, 0, 255)
    
    #get array of images in which each image has 1 capillary (all frames projected on)
    caps = find_connected_components(maxproject)

    #save maxproj individual caps, named
    caps_fp = os.path.join(os.path.dirname(registered_fp), "proj_caps")
    os.makedirs(caps_fp, exist_ok=True)
    for x in range(len(caps)):
        filename = "cap_" + str(x).zfill(2) + ".png"
        cap_fp = os.path.join(caps_fp, filename)
        cv2.imwrite(str(cap_fp), caps[x])

    #save individual caps, named
    separate_caps(registered_fp)

    #save untranslated individual caps, named
    save_untranslated(registered_fp)

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