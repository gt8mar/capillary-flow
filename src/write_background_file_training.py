"""
Filename: write_background_file_training.py
------------------------------------------------------
This file takes a series of images, creates a background file, and creates a folder with
background subtracted files.
By: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
"""

import os
import time
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import cv2
from src.tools.get_images import get_images
from src.tools.pic2vid_train import pic2vid

def main(SET='set_01', sample = 'sample_000', method = "median", color = False): 
    """Takes a folder of stabilized images and takes their median or
    mean to find the background of the video. 

    Args:
        SET (string): The set of the sample
        sample (string): the sample number
        method (string): Method to create background file
        color (bool): color or grayscale video

    Returns: 
        0

    Saves: 
        background (tiff image): background of stabilized images
        video (tiff image): video of stabilized images
    """   
    input_folder = os.path.join('D:\\Marcus\\train', str(SET), str(sample))
    output_folder = input_folder
    results_folder = 'D:\\Marcus\\train_backgrounds'  # I want to save all the backgrounds to the same folder for easy transfer to hasty.ai
    if 'Results.csv' in os.listdir(os.path.join(input_folder, "metadata")):
        print('banana')
        shifts = pd.read_csv(os.path.join(input_folder, "metadata", 'Results.csv'))
        gap_left = shifts['x'].max()
        gap_right = shifts['x'].min()
        gap_bottom = shifts['y'].min()
        gap_top = shifts['y'].max()        
        print(f'gap left is {gap_left}')
        print(f'gap right is {gap_right}')
        print(f'gap bottom is {gap_bottom}')
        print(f'gap top is {gap_top}')
    else:
        print('nah')
        gap_left = 0
        gap_right = 0
        gap_bottom = 0
        gap_top = 0
    images = get_images(os.path.join(input_folder, 'moco'))
    image_files = []
    for i in range(len(images)):
        image = np.array(cv2.imread(os.path.join(input_folder, 'moco', images[i]), cv2.IMREAD_GRAYSCALE))
        cropped_image = image[gap_top:image.shape[0] + gap_bottom, gap_left:image.shape[1] + gap_right]
        image_files.append(cropped_image)        
    image_files = np.array(image_files)
    pic2vid(image_files, SET, sample, color=color) 
    ROWS, COLS = image_files[0].shape
    if method == "mean":
        background = np.mean(image_files, axis=0).astype('uint8') 
    elif method =="median":
        background = np.median(image_files, axis=0).astype('uint8') # median instead of mean
    else:
        raise ValueError("Invalid operation entered, please enter either 'median' or 'mean'.")


    # """
    # Extra functions
    # -----------------------------------------------------------------------------------------
    # """
    # if subtracted:
    #     # Enhance contrast
    #     image_files = image_files - background
    #     print(np.max(image_files))
    #     print(np.min(image_files))
    #     image_files = image_files - np.min(image_files)
    #     image_files = image_files / np.max(image_files)
    #     image_files = np.array(image_files * 255, dtype=np.uint8)
    #     print('the following should never be less than 0')
    #     print(np.min(image_files))
    #"""---------------------------------------------------------------------------------------"""

    # Add background file
    bkgd_name = f'{SET}_{sample}_background.tiff'
    cv2.imwrite(os.path.join(output_folder, bkgd_name), background)
    cv2.imwrite(os.path.join(results_folder, bkgd_name), background)
    return 0

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main('set_01', 'sample_000')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))