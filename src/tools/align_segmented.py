"""
Filename: align_segmented.py
-------------------------------------------------------------
This file
by: Gabby Rincon
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import shutil
from register_image import register_image

def align_segmented():
    vids_folder_fp = "E:\\Marcus\\gabby test data\\part11"
    segmented_folder_fp = "E:\\Marcus\\gabby test data\\part11_segmented\\230427"

    #make list of filepaths of vid 0 in moco folders of all vids
    moco_vids_fp = []
    sorted_vids_listdir = sorted(filter(lambda x: os.path.exists(os.path.join(vids_folder_fp, x)), os.listdir(vids_folder_fp)))
    for vid in sorted_vids_listdir:
        moco_folder_fp = os.path.join(vids_folder_fp, vid, "moco")
        moco_vids_fp.append(os.path.join(moco_folder_fp, os.listdir(moco_folder_fp)[0]))

    sorted_seg_listdir = sorted(filter(lambda x: os.path.isfile(os.path.join(segmented_folder_fp, x)), os.listdir(segmented_folder_fp))) #sort vids numerically
    #reference_seg_img_fp = os.path.join(segmented_folder_fp, sorted_seg_listdir[0])
    #reference_seg_img = cv2.imread(reference_seg_img_fp)
    reference_vid_fp = moco_vids_fp[0]
    reference_vid_img = cv2.imread(reference_vid_fp)

    #make folder to save registered segmented imgs
    new_folder_path = os.path.join(segmented_folder_fp, "registered")
    os.makedirs(new_folder_path, exist_ok=True)

    for x in range(1, len(sorted_seg_listdir)):
        if "vid" in sorted_vids_listdir[x]: 
            print(sorted_vids_listdir[x])
            #register vids
            #input_vid_fp = os.path.join(vids_folder_fp, sorted_vids_listdir[x])
            input_vid_fp = moco_vids_fp[x]
            input_vid_img = cv2.imread(input_vid_fp)
            registered_img, xval, yval = register_image(reference_vid_img, input_vid_img)
            
            #TEMP for visual confirmation of registering vids
            cv2.imwrite(os.path.join("E:\\Marcus\\gabby test data\\reg", "input" + os.path.basename(input_vid_fp)), input_vid_img)
            cv2.imwrite(os.path.join("E:\\Marcus\\gabby test data\\reg", "ref" + os.path.basename(reference_vid_fp)), reference_vid_img)
            cv2.imwrite(os.path.join("E:\\Marcus\\gabby test data\\reg", "reg" + os.path.basename(input_vid_fp)), registered_img)
            
            #transform new reference image
            reference_vid_fp = moco_vids_fp[x]
            reference_vid_img = cv2.imread(reference_vid_fp)
            transformation_matrix = np.array([[1, 0, -xval], [0, 1, -yval]], dtype=np.float32)
            reference_vid_img = cv2.warpAffine(reference_vid_img, transformation_matrix, (reference_vid_img.shape[1], reference_vid_img.shape[0]))

            input_seg_img_fp = os.path.join(segmented_folder_fp, sorted_seg_listdir[x])

            #transform segmented image
            transformation_matrix = np.array([[1, 0, -xval], [0, 1, -yval]], dtype=np.float32)
            registered_seg_img = cv2.warpAffine(cv2.imread(input_seg_img_fp), transformation_matrix, (1371, 1016))

            #save segmented image
            cv2.imwrite(os.path.join(new_folder_path, os.path.basename(input_seg_img_fp)), registered_seg_img)

            #reference_seg_img_fp = os.path.join(segmented_folder_fp, sorted_seg_listdir[x])
            #reference_seg_img = cv2.imread(reference_seg_img_fp)


def align_moco():
    vids_folder_fp = "E:\\Marcus\\gabby test data\\part10\\230425"

    #make list of filepaths of vid 0 in moco folders of all vids
    moco_vids_fp = []
    sorted_vids_listdir = sorted(filter(lambda x: os.path.exists(os.path.join(vids_folder_fp, x)), os.listdir(vids_folder_fp)))
    for vid in sorted_vids_listdir:
        moco_folder_fp = os.path.join(vids_folder_fp, vid, "moco")
        moco_vids_fp.append(os.path.join(moco_folder_fp, os.listdir(moco_folder_fp)[0]))

    reference_vid_fp = moco_vids_fp[0]
    reference_vid_img = cv2.imread(reference_vid_fp)

    for x in range(1, len(sorted_vids_listdir)):
        if "vid" in sorted_vids_listdir[x]: 
            #register vids
            input_vid_fp = moco_vids_fp[x]
            input_vid_img = cv2.imread(input_vid_fp)
            registered_img, xval, yval = register_image(reference_vid_img, input_vid_img)
            
            #TEMP for visual confirmation of registering vids
            cv2.imwrite(os.path.join("E:\\Marcus\\gabby test data\\reg", "input" + os.path.basename(input_vid_fp)), input_vid_img)
            cv2.imwrite(os.path.join("E:\\Marcus\\gabby test data\\reg", "ref" + os.path.basename(reference_vid_fp)), reference_vid_img)
            cv2.imwrite(os.path.join("E:\\Marcus\\gabby test data\\reg", "reg" + os.path.basename(input_vid_fp)), registered_img)
            
            #transform new reference image
            reference_vid_fp = moco_vids_fp[x]
            reference_vid_img = cv2.imread(reference_vid_fp)
            transformation_matrix = np.array([[1, 0, -xval], [0, 1, -yval]], dtype=np.float32)
            reference_vid_img = cv2.warpAffine(reference_vid_img, transformation_matrix, (reference_vid_img.shape[1], reference_vid_img.shape[0]))
    



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