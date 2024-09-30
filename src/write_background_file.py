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
import platform
if platform.system() != 'Windows':
    from src.tools.parse_path import parse_path
    from src.tools.get_images import get_images
    from src.tools.pic2vid import pic2vid
else:
    from tools.parse_path import parse_path
    from tools.get_images import get_images
    from tools.pic2vid import pic2vid
import platform
from tifffile import imwrite

def main(path = 'C:\\Users\\gt8mar\\capillary-flow\\data\\part_11\\230427\\loc01\\vids\\vid01', 
         method = "mean", make_video = True, color = False, verbose=False, plot=False):  
    """
    Writes a background file and a video into results and backgrounds.

    Args: 
        path (str): Path to the umbrella video folder within the location and vids folder.
        method (string): Method to create background file. Either 'median' or 'mean'.
        color (bool): Whether to make a color video or not (grayscale)

    Returns: 
        int: 0 if executed

    Saves: 
        background (tiff image): background of stabilized images
        video (.avi): video of stabilized images
    """  
    print(path)
    # check to see if 'mocoslice' folder exists
    if os.path.exists(os.path.join(path, 'moco-contrasted')):
        input_folder = (os.path.join(path, 'moco-contrasted'))
    elif os.path.exists(os.path.join(path, 'mocoslice')):
        input_folder = os.path.join(path, 'mocoslice')
    elif os.path.exists(os.path.join(path, 'mocosplit')):
        input_folder = os.path.join(path, 'mocosplit')
    else:
        input_folder = os.path.join(path, 'moco')
    location_folder = os.path.dirname(os.path.dirname((path)))
    location = os.path.basename(location_folder)
    os.makedirs(os.path.join(location_folder, 'backgrounds'), exist_ok=True)
    output_folder = os.path.join(location_folder, 'backgrounds')
    os.makedirs(os.path.join(location_folder, 'stdevs'), exist_ok=True)
    output_folder_stdev = os.path.join(location_folder, 'stdevs')

    if platform.system() != 'Windows':
        results_folder = '/hpc/projects/capillary-flow/results/backgrounds'  # I want to save all the backgrounds to the same folder for easy transfer to hasty.ai
        results_folder_stdev = '/hpc/projects/capillary-flow/results/stdevs'
        os.makedirs(results_folder_stdev, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)
    else:
        # results_folder = 'C:\\Users\\Luke\\Documents\\capillary-flow\\backgrounds'
        results_folder = 'C:\\Users\\ejerison\\capillary-flow\\results\\backgrounds'
        results_folder_stdev = 'C:\\Users\\ejerison\\capillary-flow\\results\\stdevs'
        # results_folder_stdev = '/hpc/projects/capillary-flow/results/stdevs'
    participant, date, location, video, file_prefix = parse_path(path, video_path=True)
    print(f'participant is {participant}, date is {date}, location is {location}, video is {video}, file_prefix is {file_prefix}')

    # Read in shift values from stabilization algorithm
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

    print(f'gap left is {gap_left}')
    print(f'gap right is {gap_right}')
    print(f'gap bottom is {gap_bottom}')
    print(f'gap top is {gap_top}')

    # Read in sorted image names
    images = get_images(os.path.join(input_folder))
    image_files = []
    for i in range(len(images)):
        # Read in image
        image = np.array(cv2.imread(os.path.join(input_folder, images[i]), cv2.IMREAD_GRAYSCALE))
        # Crop image using shifts so that there is not a black border around the outside of the video
        cropped_image = image[gap_top:image.shape[0] + gap_bottom, gap_left:image.shape[1] + gap_right]
        image_files.append(cropped_image)
        if verbose:
            print(f'cropped image shape is {cropped_image.shape}')
    # Convert to numpy array
    image_files = np.array(image_files)
    if verbose:
        print(f'image_files shape is {image_files.shape}')
    # save video
    if make_video:
        pic2vid(image_files, participant=participant, date=date, location = location, video_folder=video, color=color, overlay=True) 
    # Get dimensions of images
    ROWS, COLS = image_files[0].shape
    
    if method == "mean":
        background = np.mean(image_files, axis=0).astype('uint8') 
        if verbose:
            print(f'background shape is {background.shape}')
        if plot:
            plt.imshow(background)
            plt.show()
    elif method =="median":
        background = np.median(image_files, axis=0).astype('uint8') # median instead of mean
    else:
        raise ValueError("Invalid operation entered, please enter either 'median' or 'mean'.")
    
    # Calculate the standard deviation for each pixel
    stdevs = np.std(image_files, axis=0)

    # Normalize the standard deviation values to the range [0, 255]
    stdevs = cv2.normalize(stdevs, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the standard deviation to uint8
    stdevs_uint8 = np.uint8(stdevs)

    # Contrast enhancement
    stdevs_contrasted = cv2.equalizeHist(stdevs_uint8)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # stdevs = clahe.apply(stdevs)


    # # Enhance contrast
    # background = cv2.equalizeHist(background)


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
    bkgd_name = f'{file_prefix}_{video}_background.tiff'
    stdev_name = f'{file_prefix}_{video}_stdev.tiff'
    print(f'background name is {bkgd_name}')
    cv2.imwrite(os.path.join(output_folder, bkgd_name), background)
    cv2.imwrite(os.path.join(results_folder, bkgd_name), background)
    cv2.imwrite(os.path.join(output_folder_stdev, stdev_name), stdevs_contrasted)
    cv2.imwrite(os.path.join(results_folder_stdev, stdev_name), stdevs_contrasted)

    
    return 0

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    # path = 'D:\\Marcus\\backup\\data\\part25\\230601\\loc02\\vids\\vid27'
    # short_path = 'D:\\Marcus\\backup\\data\\part24\\230601\\loc03\\vids'
    # short_path = 'I:\\Marcus\\data\\part49\\240619\\loc02\\vids\\vid14'
    short_path = 'C:\\Users\\gt8mar\\Desktop\\data\\part81\\240718\\loc02\\vids\\vid21'
    # vids = ['vid' + str(i) for i in range(38, 49)]
    # for vid in vids:
    #     long_path = os.path.join(short_path, vid)
    #     main(path=long_path, method="mean", make_video=False, color=False, verbose=False)
    main(path=short_path, method="median", make_video=False, color=False, verbose=False)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))