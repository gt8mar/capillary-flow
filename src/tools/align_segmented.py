"""
Filename: align_segmented.py
-------------------------------------------------------------
This file aligns segmented images based on translations between moco images.
by: Gabby Rincon
"""
#TODO location on finger
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import shutil
from src.tools.register_images import register_images
import csv

def uncrop_segmented(path, input_seg_img):
    shifts = pd.read_csv(os.path.join(path, 'metadata', 'Results.csv'))
    gap_left = shifts['x'].max()
    gap_right = shifts['x'].min()
    gap_bottom = shifts['y'].min()
    gap_top = shifts['y'].max()

    slices = [slice(None)] * input_seg_img.ndim
    rows, cols = input_seg_img.shape[:2]
    new_rows = rows + gap_top + np.abs(gap_bottom)
    new_cols = cols + gap_left + np.abs(gap_right)
    
    uncropped_input_seg_img = np.zeros((new_rows, new_cols) + input_seg_img.shape[2:], dtype=input_seg_img.dtype)
    slices[:2] = slice(gap_top, gap_top+rows), slice(gap_left, gap_left+cols)
    uncropped_input_seg_img[tuple(slices)] = input_seg_img

    return uncropped_input_seg_img, gap_left, gap_right, gap_bottom, gap_top

#this function assumes moco folder & seg imgs folder contain the same number of files & they correspond to each other 
def align_segmented(path="E:\\Marcus\\gabby_test_data\\part11\\230427\\loc02"):
    vid_folder_fp = os.path.join(path, "vids")
    segmented_folder_fp = os.path.join(path, "segmented")

    #make list of filepaths of vid 0 in moco folders of all vids
    moco_vids_fp = []
    sorted_vids_listdir = sorted(filter(lambda x: os.path.exists(os.path.join(vid_folder_fp, x)), os.listdir(vid_folder_fp))) #sort numerically
    for vid in sorted_vids_listdir:
        moco_folder_fp = os.path.join(vid_folder_fp, vid, "moco")
        moco_vids_fp.append(os.path.join(moco_folder_fp, os.listdir(moco_folder_fp)[0]))

    #set reference
    reference_moco_fp = moco_vids_fp[0]
    reference_moco_img = cv2.imread(reference_moco_fp)

    #make folder to save registered segmented imgs
    reg_folder_path = os.path.join(segmented_folder_fp, "registered")
    os.makedirs(reg_folder_path, exist_ok=True)

    """#TEMP new folder for registered frames
    temp_fp = os.path.join(segmented_folder_fp, "moco")
    os.makedirs(temp_fp, exist_ok=True)"""

    crops = []

    #first frame
    sorted_seg_listdir = sorted(filter(lambda x: os.path.isfile(os.path.join(segmented_folder_fp, x)) and x.endswith(".png"), os.listdir(segmented_folder_fp))) #sort numerically
    first_seg_fp = os.path.join(segmented_folder_fp, sorted_seg_listdir[0])
    first_seg_img = cv2.imread(first_seg_fp)
    first_seg_img, left, right, bottom, top = uncrop_segmented(os.path.join(os.path.split(os.path.split(moco_vids_fp[0])[0])[0]), first_seg_img)
    crops.append((left, right, bottom, top))
    cv2.imwrite(os.path.join(reg_folder_path, os.path.basename(first_seg_fp)), first_seg_img)

    translations = []
    prevdx = 0
    prevdy = 0
    translations.append((prevdx, prevdy)) 
    print(sorted_seg_listdir)
    for x in range(1, len(sorted_seg_listdir)):
        if "vid" in sorted_vids_listdir[x]: 
            #register vids
            input_moco_fp = moco_vids_fp[x]
            input_moco_img = cv2.imread(input_moco_fp)
            (dx, dy), registered_image = register_images(reference_moco_img, input_moco_img)

            """#TEMP
            temp_transformation_matrix = np.array([[1, 0, -(prevdx)], [0, 1, -(prevdy)]], dtype=np.float32)
            temp_registered_image = cv2.warpAffine(registered_image, temp_transformation_matrix, (1440, 1080))
            cv2.imwrite(os.path.join(temp_fp, os.path.basename(moco_vids_fp[x])), temp_registered_image)
            """
            #get image to segment
            input_seg_fp = os.path.join(segmented_folder_fp, sorted_seg_listdir[x])
            input_seg_img = cv2.imread(input_seg_fp)

            #make segmented same size
            input_seg_img, left, right, bottom, top = uncrop_segmented(os.path.join(os.path.split(os.path.split(moco_vids_fp[x])[0])[0]), input_seg_img)
            crops.append((left, right, bottom, top))

            #transform segmented image
            transformation_matrix = np.array([[1, 0, -(dx + prevdx)], [0, 1, -(dy + prevdy)]], dtype=np.float32)
            registered_seg_img = cv2.warpAffine(input_seg_img, transformation_matrix, (1440, 1080))

            translations.append((dx + prevdx, dy + prevdy))

            #save segmented image
            cv2.imwrite(os.path.join(reg_folder_path, os.path.basename(input_seg_fp)), registered_seg_img)

            #set new reference, prevdx, prevdy
            reference_moco_img = input_moco_img
            prevdx += dx
            prevdy += dy

    translations_csv_fp = os.path.join(segmented_folder_fp, "translations.csv")
    with open(translations_csv_fp, 'w', newline='') as translations_csv_file:
        writer = csv.writer(translations_csv_file) 
        writer.writerows(translations)

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