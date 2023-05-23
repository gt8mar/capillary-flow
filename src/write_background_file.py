"""
Filename: write_background_file.py
------------------------------------------------------
This file takes a series of images, creates a background file, and creates a folder with
background subtracted files.
By: Marcus Forst
"""

import os
import time
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import cv2
from src.tools.parse_vid_path import parse_vid_path
from src.tools.get_images import get_images
from src.tools.pic2vid import pic2vid

def main(path = 'C:\\Users\\gt8mar\\capillary-flow\\data\\part_11\\230427\\vid1', 
         method = "median", color = False):  
    """
    Writes a background file and a video into results and C_background.

    Args: 
        path (str): Path to the umbrella video folder.
        method (string): Method to create background file
        color (bool): Whether to make a color video or not (grayscale)

    Returns: 
        int: 0 if executed

    Saves: 
        background (tiff image): background of stabilized images
        video (.avi): video of stabilized images
    """  
    input_folder = os.path.join(path, 'moco')
    os.makedirs(os.path.join(path, 'C_background'), exist_ok=True)
    output_folder = os.path.join(path, 'C_background')
    results_folder = '/hpc/projects/capillary-flow/results/backgrounds'  # I want to save all the backgrounds to the same folder for easy transfer to hasty.ai
    participant, date, video = parse_vid_path(path)

    # Read in shift values from stabilization algorithm
    shifts = pd.read_csv(os.path.join(path, 'metadata', 'Results.csv'))
    gap_left = shifts['x'].max()
    gap_right = shifts['x'].min()
    gap_bottom = shifts['y'].min()
    gap_top = shifts['y'].max()
    print(f'gap left is {gap_left}')
    print(f'gap right is {gap_right}')
    print(f'gap bottom is {gap_bottom}')
    print(f'gap top is {gap_top}')
    images = get_images(os.path.join(input_folder))
    image_files = []
    for i in range(len(images)):
        image = np.array(cv2.imread(os.path.join(input_folder, images[i]), cv2.IMREAD_GRAYSCALE))
        # Crop image using shifts so that there is not a black border around the outside of the video
        cropped_image = image[gap_top:image.shape[0] + gap_bottom, gap_left:image.shape[1] + gap_right]
        image_files.append(cropped_image)
    image_files = np.array(image_files)
    pic2vid(image_files, participant=participant, date=date, video_folder=video, color=color) 
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

    # Add background file
    bkgd_name = f'set_01_{participant}_{date}_{video}_background.tiff'
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