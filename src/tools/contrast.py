"""
Filename: contrast.py
-------------------
This file uses rescale_intensity to increase the contrast of the images. 
By: Marcus Forst
"""

import numpy as np
import os, time
import cv2
import random
from skimage import exposure, util
import matplotlib.pyplot as plt





def parse_filename(filename):
    """
    Parses the filename of an image into its participant, date, and video number.

    Args:
        filename (str): Filename of the image. format: set_participant_date_video_background.tiff
    
    Returns:
        participant (str): Participant number.
        date (str): Date of the video.
        video (str): Video number.
    """
    filename_no_ext = filename.split('.')[0]
    participant = filename_no_ext.split('_')[-4]
    date = filename_no_ext.split('_')[-3]
    video = filename_no_ext.split('_')[-2]
    return participant, date, video

def make_histogram(im):
    """
    Plots a histogram of the image.

    Args:
        im (numpy.ndarray): Image to be plotted.
    
    Returns:
        0 (int): Returns 0 upon completion.
    """
    plt.hist(im.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.show()
    return 0

def main(path=None, verbose = False):
    """
    Uses rescale_intensity to increase the contrast of the images.

    Args:
        path (str): Path to the folder containing the images to be contrast enhanced.
        verbose (bool): If True, plots histograms of the images before and after contrast enhancement.    
    Returns:
        0 (int): Returns 0 upon completion.    
    Saves:
        file_contrast (tiff): Contrast enhanced images.
    """
    for file in os.listdir(path):
        if file.endswith('.tiff'):
            # Load the image
            im = cv2.imread(os.path.join(path, file))
            # extract the filename from the path
            filename = os.path.basename(os.path.join(path, file))
            participant, date, video = parse_filename(filename)
            # remove the file extension
            filename_without_ext = os.path.splitext(filename)[0]
            # extract the desired string from the filename
            sample = filename_without_ext.split('_background')[0]
            im_norm = util.img_as_float(im)
            im_contrast = exposure.rescale_intensity(im_norm)
            print(filename_without_ext)
            im_contrast_256 = util.img_as_ubyte(im_contrast)
            
            # Save the image
            os.makedirs(os.path.join(path, "contrast"), exist_ok=True)
            cv2.imwrite(os.path.join(path, "contrast", f'{filename_without_ext}_contrast.tiff'), im_contrast_256)
            if verbose:
                fig, ax = plt.subplots(1, 2, figsize=(8, 4))
                ax[1].hist(im_contrast.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
                ax[0].hist(im.ravel(), bins=256, range=(0.0, 256.0), fc='k', ec='k')    
                plt.show()
    return 0

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    path = "C:\\Users\\gt8mar\\capillary-flow\\results\\backgrounds"
    main(path, verbose=False)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))