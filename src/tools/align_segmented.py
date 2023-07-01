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
from register_image import register

#this function assumes vids folder & segmented folder have the same number of files & correspond to each other in order
def main():
    vids_folder_fp = "E:\\Marcus\\gabby test data\\part11_mocovids"
    segmented_folder_fp = "E:\\Marcus\\gabby test data\\part11_segmented\\230427"
    
    reference_vid_fp = os.path.join(vids_folder_fp, os.listdir(vids_folder_fp)[0])
    reference_seg_img_fp = os.path.join(segmented_folder_fp, os.listdir(segmented_folder_fp)[0])
    reference_seg_img = cv2.imread(reference_seg_img_fp)

    new_folder_path = os.path.join(segmented_folder_fp, "registered")
    os.makedirs(new_folder_path, exist_ok=True)
    for x in range(1, len(os.listdir(vids_folder_fp))):
        #register vids
        input_vid = os.path.join(vids_folder_fp, os.listdir(vids_folder_fp)[x])
        xval, yval = register(reference_vid_fp, input_vid)
        reference_vid_fp = os.path.join(vids_folder_fp, os.listdir(vids_folder_fp)[x])

        input_seg_img_fp = os.path.join(segmented_folder_fp, os.listdir(segmented_folder_fp)[x])
        #transform
        transformation_matrix = np.array([[1, 0, xval], [0, 1, yval]], dtype=np.float32)
        registered_seg_img = cv2.warpAffine(cv2.imread(input_seg_img_fp), transformation_matrix, (reference_seg_img.shape[1], reference_seg_img.shape[0]))

        #save
        cv2.imwrite(os.path.join(new_folder_path, os.path.basename(input_seg_img_fp)), registered_seg_img)

        reference_seg_img_fp = os.path.join(segmented_folder_fp, os.listdir(segmented_folder_fp)[x])
        reference_seg_img = cv2.imread(reference_seg_img_fp)




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