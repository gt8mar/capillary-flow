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
if platform.system() != 'Windows':
    from src.tools.register_images import register_images
else:
    from register_images import register_images
import csv
from skimage.color import rgb2gray
from skimage import io

def uncrop_segmented(path, input_seg_img):
    shifts = pd.read_csv(os.path.join(path, 'metadata', 'Results.csv'))
    gap_left = shifts['x'].max()
    gap_right = shifts['x'].min()
    gap_bottom = shifts['y'].min()
    gap_top = shifts['y'].max()

    # Check to make sure that the shifts are not negative
    if gap_left < 0:
        gap_left = 0
    if gap_top < 0:
        gap_top = 0
    if gap_right > 0:
        gap_right = 0
    if gap_bottom > 0:
        gap_bottom = 0

    input_seg_img = rgb2gray(input_seg_img)

    uncropped_input_seg_img = np.pad(input_seg_img, ((abs(gap_top), abs(gap_bottom)), (abs(gap_left), abs(gap_right))), mode='constant', constant_values=0)
    return uncropped_input_seg_img, gap_left, gap_right, gap_bottom, gap_top

#this function assumes moco folder & seg imgs folder contain the same number of files & they correspond to each other 
def align_segmented(path="D:\\data_gabby\\debugging\\part09\\230414\\loc06", verbose=False):
    vid_folder_fp = os.path.join(path, "vids")
    segmented_folder_fp = os.path.join(path, "segmented", "hasty")

    #make folder to save registered moco images
    reg_moco_folder = os.path.join(segmented_folder_fp, "moco_registered")
    os.makedirs(reg_moco_folder, exist_ok=True)

    #make list of filepaths of vid 0 in moco folders of all vids
    moco_vids_fp = []
    sorted_vids_listdir = sorted(filter(lambda x: os.path.exists(os.path.join(vid_folder_fp, x)), os.listdir(vid_folder_fp))) #sort numerically
    for vid in sorted_vids_listdir:
        if os.path.exists(os.path.join(vid_folder_fp, vid, "mocoslice")):
            moco_folder_fp = os.path.join(vid_folder_fp, vid, "mocoslice")
        elif os.path.exists(os.path.join(vid_folder_fp, vid, "mocosplit")):
            moco_folder_fp = os.path.join(vid_folder_fp, vid, "mocosplit")
        else:
            moco_folder_fp = os.path.join(vid_folder_fp, vid, "moco")
        sorted_moco_ld = sorted(filter(lambda x: os.path.exists(os.path.join(moco_folder_fp, x)), os.listdir(moco_folder_fp)))
        if verbose:
            print(sorted_moco_ld)
        moco_vids_fp.append(os.path.join(moco_folder_fp, sorted_moco_ld[0]))

    #set reference
    reference_moco_fp = moco_vids_fp[0]
    reference_moco_img = cv2.imread(reference_moco_fp)

    #save reference moco
    contrast_reference_moco_img = cv2.equalizeHist(cv2.cvtColor(reference_moco_img, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(os.path.join(reg_moco_folder, os.path. basename(reference_moco_fp)), np.pad(contrast_reference_moco_img, ((250, 250), (250, 250))))

    #make folder to save registered segmented imgs
    reg_folder_path = os.path.join(segmented_folder_fp, "registered")
    os.makedirs(reg_folder_path, exist_ok=True)

    crops = []

    #first frame
    sorted_seg_listdir = sorted(filter(lambda x: os.path.isfile(os.path.join(segmented_folder_fp, x)) and x.endswith(".png"), os.listdir(segmented_folder_fp))) #sort numerically
    first_seg_fp = os.path.join(segmented_folder_fp, sorted_seg_listdir[0])
    first_seg_img = cv2.imread(first_seg_fp)
    first_seg_img, left, right, bottom, top = uncrop_segmented(os.path.join(os.path.split(os.path.split(moco_vids_fp[0])[0])[0]), first_seg_img)

    translations = []
    prevdx = 0
    prevdy = 0
    translations.append([prevdx, prevdy]) 
    for x in range(1, len(sorted_seg_listdir)):
        if "vid" in sorted_vids_listdir[x]: 
            #register vids
            input_moco_fp = moco_vids_fp[x]
            input_moco_img = cv2.imread(input_moco_fp)
            [dx, dy], registered_image = register_images(reference_moco_img, input_moco_img, prevdx, prevdy)

            dx = int(dx)
            dy = int(dy)
            translations.append([dx + prevdx, dy + prevdy])

            #set new reference, prevdx, prevdy
            reference_moco_img = input_moco_img
            prevdx += dx
            prevdy += dy

            #save registered moco frame
            cv2.imwrite(os.path.join(reg_moco_folder, os.path.basename(input_moco_fp)), registered_image)

    #get max size of segmented img
    minx = min(0, min(entry[0] for entry in translations))
    maxx = max(0, max(entry[0] for entry in translations))
    miny = min(0, min(entry[1] for entry in translations))
    maxy = max(0, max(entry[1] for entry in translations))

    resize_vals = []

    for x in range(0, len(sorted_seg_listdir)):
        if "vid" in sorted_vids_listdir[x]: 
            #get image to segment
            input_seg_fp = os.path.join(segmented_folder_fp, sorted_seg_listdir[x])
            input_seg_img = cv2.imread(input_seg_fp)

            #make segmented same size
            input_seg_img, left, right, bottom, top = uncrop_segmented(os.path.join(os.path.split(os.path.split(moco_vids_fp[x])[0])[0]), input_seg_img)
            crops.append((left, right, bottom, top))

            #transform segmented image
            padbottom = abs(miny) + translations[x][1]
            padtop = abs(maxy) - translations[x][1]
            padright = abs(minx) + translations[x][0]
            padleft = abs(maxx) - translations[x][0]
            registered_seg_img = np.pad(input_seg_img, ((padtop, padbottom), (padleft, padright)), mode='constant', constant_values=0)

            resize_vals.append([minx, maxx, miny, maxy])

            #save segmented image
            registered_seg_img = (registered_seg_img * 255).astype(np.uint8)
            io.imsave(os.path.join(reg_folder_path, os.path.basename(input_seg_fp)), registered_seg_img)

    translations_csv_fp = os.path.join(segmented_folder_fp, "translations.csv")
    with open(translations_csv_fp, 'w', newline='') as translations_csv_file:
        writer = csv.writer(translations_csv_file) 
        writer.writerows(translations)

    resize_csv_fp = os.path.join(segmented_folder_fp, "resize_vals.csv")
    with open(resize_csv_fp, 'w', newline='') as resize_csv_file:
        writer = csv.writer(resize_csv_file) 
        writer.writerows(resize_vals)

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