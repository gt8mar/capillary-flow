"""
Filename: align_segmented.py
-------------------------------------------------------------
This file aligns segmented images based on translations between moco images.
by: Gabby Rincon
"""

import os
import time
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import shutil
from skimage.color import rgb2gray
from skimage import io
import csv
from src.tools.parse_path import parse_path
from src.tools.parse_filename import parse_filename

PAD_VALUE = 250
MAX_TRANSLATION = 200

# Import the appropriate module based on the operating system
if platform.system() != 'Windows':
    from src.tools.register_images import register_images_moco, register_images
else:
    from register_images import register_images_moco, register_images

def uncrop_segmented(path, input_seg_img):
    """
    Uncrops a segmented image based on shifts from a CSV file.

    Args:
        path (str): Path to the "video" folder containing the 'Results.csv' file with shift data.
        input_seg_img (numpy.ndarray): Input segmented image.

    Returns:
        tuple: Uncropped image and the left, right, bottom, and top gaps.
    """
    shifts = pd.read_csv(os.path.join(path, 'metadata', 'Results.csv'))
    gap_left = shifts['x'].max()
    gap_right = shifts['x'].min()
    gap_bottom = shifts['y'].min()
    gap_top = shifts['y'].max()

    # Ensure that the gaps are non-negative
    if gap_left < 0:
        gap_left = 0
    if gap_top < 0:
        gap_top = 0
    if gap_right > 0:
        gap_right = 0
    if gap_bottom > 0:
        gap_bottom = 0

    # Convert the segmented image to grayscale
    input_seg_img = rgb2gray(input_seg_img)

    # Pad the image based on the calculated gaps
    uncropped_input_seg_img = np.pad(input_seg_img, ((abs(gap_top), abs(gap_bottom)), (abs(gap_left), abs(gap_right))), mode='constant', constant_values=0)
    return uncropped_input_seg_img, gap_left, gap_right, gap_bottom, gap_top



def align_segmented(path="f:\\Marcus\\data\\part30\\231130\\loc02"):
    """
    Aligns segmented images based on translations between moco images.

    Args:
        path (str): The path to the location directory for a given participant.

    Creates:
        Directories for registered moco images and registered segmented images.
        CSV files with translations, resize values, and crop values.
    """
    vid_folder_fp = os.path.join(path, "vids")
    segmented_folder_fp = os.path.join(path, "segmented", "hasty")

    # Create folder to save registered moco images
    reg_moco_folder = os.path.join(segmented_folder_fp, "moco_registered")
    os.makedirs(reg_moco_folder, exist_ok=True)

    # Make list of filepaths of vid 0 in moco folders of all vids
    moco_vids_fp = []
    sorted_vids_listdir = sorted(filter(lambda x: os.path.exists(os.path.join(vid_folder_fp, x)), os.listdir(vid_folder_fp)))  # Sort numerically
    for vid in sorted_vids_listdir:
        vid_path = os.path.join(vid_folder_fp, vid)
        if os.path.exists(os.path.join(vid_folder_fp, vid, "mocoslice")):
            moco_folder_fp = os.path.join(vid_folder_fp, vid, "mocoslice")
        elif os.path.exists(os.path.join(vid_folder_fp, vid, "mocosplit")):
            moco_folder_fp = os.path.join(vid_folder_fp, vid, "mocosplit")
        else:
            moco_folder_fp = os.path.join(vid_folder_fp, vid, "moco")
        sorted_moco_ld = sorted(filter(lambda x: os.path.exists(os.path.join(moco_folder_fp, x)), os.listdir(moco_folder_fp)))
        # append the video name and the first moco image path. This fixes a bug with naming if the moco images are not named correctly
        moco_vids_fp.append((vid, os.path.join(moco_folder_fp, sorted_moco_ld[0])))

    # Set reference image
    reference_moco_tuple = moco_vids_fp[0]
    first_video = reference_moco_tuple[0]
    first_video_path = os.path.join(vid_folder_fp, first_video)
    reference_moco_fp = reference_moco_tuple[1]
    reference_moco_img = cv2.imread(reference_moco_fp)
    reference_moco_filename = f'{first_video}_moco_0000.tif'


    # Save reference moco image with contrast adjustment
    contrast_reference_moco_img = cv2.equalizeHist(cv2.cvtColor(reference_moco_img, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(os.path.join(reg_moco_folder, reference_moco_filename), np.pad(contrast_reference_moco_img, ((PAD_VALUE, PAD_VALUE), (PAD_VALUE, PAD_VALUE))))

    # Create folder to save registered segmented images
    reg_folder_path = os.path.join(segmented_folder_fp, "registered")
    os.makedirs(reg_folder_path, exist_ok=True)

    crops = []

    # Process the first segmented frame
    sorted_seg_listdir = sorted(filter(lambda x: os.path.isfile(os.path.join(segmented_folder_fp, x)) and x.endswith(".png"), os.listdir(segmented_folder_fp)))  # Sort numerically
    first_seg_fp = os.path.join(segmented_folder_fp, sorted_seg_listdir[0])
    first_seg_img = cv2.imread(first_seg_fp)
    
    # Use the shifts from the Results.csv file to uncrop the first segmented image we will register to
    first_seg_img, left, right, bottom, top = uncrop_segmented(first_video_path, first_seg_img)

    translations = []
    prevdx = 0
    prevdy = 0
    translations.append([prevdx, prevdy]) 

    # Process remaining frames
    for i in range(1, len(sorted_seg_listdir)):
        if "vid" in sorted_vids_listdir[i]: 
            # Register vids
            input_moco_tuple = moco_vids_fp[i]
            input_moco_fp = input_moco_tuple[1]
            video = input_moco_tuple[0]
            input_moco_img = cv2.imread(input_moco_fp)
            input_moco_filename = f'{video}_moco_0000.tif'
            # [dx, dy], registered_image = register_images_moco(reference_moco_img, input_moco_img, max_shift=MAX_TRANSLATION)

            [dx, dy], registered_image = register_images(reference_moco_img, input_moco_img, prevdx, prevdy, max_shift=MAX_TRANSLATION, pin=True)
            # # Alternatively, you can clamp the translations to the max value
            # dx = np.clip(dx, -MAX_TRANSLATION, MAX_TRANSLATION)
            # dy = np.clip(dy, -MAX_TRANSLATION, MAX_TRANSLATION)


            dx = int(dx)
            dy = int(dy)
            translations.append([dx + prevdx, dy + prevdy])

            # Update reference image and previous translations
            reference_moco_img = input_moco_img
            prevdx += dx
            prevdy += dy

            # Save registered moco frame
            cv2.imwrite(os.path.join(reg_moco_folder, input_moco_filename), registered_image)

    # Calculate the maximum size of segmented images
    minx = min(0, min(entry[0] for entry in translations))
    maxx = max(0, max(entry[0] for entry in translations))
    miny = min(0, min(entry[1] for entry in translations))
    maxy = max(0, max(entry[1] for entry in translations))

    resize_vals = []

    for x in range(0, len(sorted_seg_listdir)):
        if "vid" in sorted_vids_listdir[x]: 
            participant, date, location, seg_video, __= parse_filename(sorted_seg_listdir[x])
            seg_video_filepath = os.path.join(vid_folder_fp, seg_video)
            # Get image to segment
            input_seg_fp = os.path.join(segmented_folder_fp, sorted_seg_listdir[x])
            input_seg_img = cv2.imread(input_seg_fp)

            # Make segmented image same size using Results.csv file from video folder
            input_seg_img, left, right, bottom, top = uncrop_segmented(seg_video_filepath, input_seg_img)
            crops.append((left, right, bottom, top))

            # Transform segmented image
            padbottom = abs(miny) + translations[x][1]
            padtop = abs(maxy) - translations[x][1]
            padright = abs(minx) + translations[x][0]
            padleft = abs(maxx) - translations[x][0]
            registered_seg_img = np.pad(input_seg_img, ((padtop, padbottom), (padleft, padright)), mode='constant', constant_values=0)

            resize_vals.append([minx, maxx, miny, maxy])

            # Save segmented image
            registered_seg_img = (registered_seg_img * 255).astype(np.uint8)
            io.imsave(os.path.join(reg_folder_path, os.path.basename(input_seg_fp)), registered_seg_img)

    # Save translations to CSV file
    translations_csv_fp = os.path.join(segmented_folder_fp, "translations.csv")
    with open(translations_csv_fp, 'w', newline='') as translations_csv_file:
        writer = csv.writer(translations_csv_file) 
        writer.writerows(translations)

    # Save resize values to CSV file
    resize_csv_fp = os.path.join(segmented_folder_fp, "resize_vals.csv")
    with open(resize_csv_fp, 'w', newline='') as resize_csv_file:
        writer = csv.writer(resize_csv_file) 
        writer.writerows(resize_vals)

    # Save crop values to CSV file
    crops_csv_fp = os.path.join(segmented_folder_fp, "crop_values.csv")
    with open(crops_csv_fp, 'w', newline='') as crops_csv_file:
        writer = csv.writer(crops_csv_file) 
        writer.writerows(crops)

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    align_segmented()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
