"""
Filename: write_background_file.py
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
from src.tools.pic2vid import pic2vid

def main(SET='set_01', sample = 'sample_000'):    
    input_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'B_stabilized')
    output_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'C_background')
    shifts = pd.read_csv(os.path.join(input_folder, 'Results.csv'))
    gap_left = shifts['x'].max()
    gap_right = shifts['x'].min()
    gap_bottom = shifts['y'].min()
    gap_top = shifts['y'].max()
    print(f'gap left is {gap_left}')
    print(f'gap right is {gap_right}')
    print(f'gap bottom is {gap_bottom}')
    print(f'gap top is {gap_top}')
    images = get_images(os.path.join(input_folder, 'vid'))
    image_files = []
    for i in range(len(images)):
        image = np.array(cv2.imread(os.path.join(input_folder, 'vid', images[i]), cv2.IMREAD_GRAYSCALE))
        cropped_image = image[gap_top:gap_bottom, gap_left:gap_right]
        image_files.append(cropped_image)
    image_files = np.array(image_files)
    pic2vid.main(image_files, SET, sample) 
    ROWS, COLS = image_files[0].shape
    background = np.mean(image_files, axis=0).astype('uint8')


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
    bkgd_name = f'{SET}_{sample}_background.tiff'
    cv2.imwrite(os.path.join(output_folder, bkgd_name), background)
    return 0

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main('set_01', 'sample_009')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))