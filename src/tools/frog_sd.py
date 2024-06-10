"""
Filename: frawg_sd.py
------------------------------------------------------
This file takes a series of images, creates a background file, and creates a folder with
background subtracted files.

By: Marcus Forst
"""

import os, platform
import cv2
import numpy as np


"""
USER INSTRUCTIONS:

1. Set the MAIN_FOLDER variable below to the path of the folder containing the subfolders with the .tiff files.
Example: MAIN_FOLDER = "/path/to/folder"

This folder should either be "Left" or "Right" and contain subfolders with the .tiff files.
"""

MAIN_FOLDER = "/path/to/folder"


"""
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
DO NOT MODIFY ANYTHING BELOW THIS LINE
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"""

def process_subfolder(subfolder_path, output_path, results_path):
    """
    Process a subfolder containing a series of .tiff files and calculate the standard deviation image.

    Args:
        subfolder_path (str): The path to the subfolder containing the .tiff files.
        output_path (str): The path to save the standard deviation image.
        results_path (str): The path to save the standard deviation image in the results folder.

    Returns:
        0 if successful
    """
    # List all .tiff files in the subfolder
    frame_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith('.tiff')]

    # Check if there are no .tiff files in the subfolder
    if not frame_files:
        print(f"No .tiff files found in {subfolder_path}. Skipping...")
        return 1

    # Read all frames into a list
    frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in frame_files]

    # Check if any frames are None (failed to load)
    if any(frame is None for frame in frames):
        print(f"Failed to read some images in {subfolder_path}. Skipping...")
        return 1

    # Convert the list of frames to a numpy array
    frames_array = np.array(frames)

    # Ensure there is more than one frame to calculate the standard deviation
    if frames_array.shape[0] < 2:
        print(f"Not enough images in {subfolder_path} to calculate standard deviation. Skipping...")
        return 1

    # Calculate the standard deviation for each pixel
    stdevs = np.std(frames_array, axis=0)

    # Normalize the standard deviation values to the range [0, 255]
    stdevs = cv2.normalize(stdevs, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the standard deviation to uint8
    stdevs_uint8 = np.uint8(stdevs)

    # Save the standard deviation image in .tiff format
    cv2.imwrite(output_path, stdevs_uint8)
    cv2.imwrite(results_path, stdevs_uint8)

    return 0

def main(main_folder = MAIN_FOLDER):
    """
    Process all subfolders in the main folder and calculate the standard deviation image for each subfolder.
    """
    # Define the main folder and the output folder
    output_folder = os.path.join(main_folder, "stdevs")
    if platform.system() == 'Windows':
        if 'ejerison' in os.getcwd():
            results_folder = 'C:\\Users\\ejerison\\capillary-flow\\tests\\results\\stdevs'
        elif 'gt8mar' in os.getcwd():
            results_folder = 'C:\\Users\\home/gt8mar/capillary-flow/frog/results/stdevs'
        else:
            results_folder = '/home/gt8ma/capillary-flow/frog/results/stdevs'
    else:
        if 'hpc' in os.getcwd():
            results_folder = '/hpc/projects/capillary-flow/frog/results/stdevs'
        else:
            results_folder = output_folder
    

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    # Loop through each subfolder in the main folder
    for subfolder in os.listdir(main_folder):
        if subfolder == 'stdevs':
            continue
        elif subfolder.startswith('24'):
            print(f"Processing subfolder: {subfolder}")
            subfolder_path = os.path.join(main_folder, subfolder)
            
            # Ensure it's a directory
            if os.path.isdir(subfolder_path):
                # Define the output path for the standard deviation image
                output_path = os.path.join(output_folder, f"SD_{subfolder}.tiff")
                results_path = os.path.join(results_folder, f"SD_{subfolder}.tiff")
                # Process the subfolder
                process_subfolder(subfolder_path, output_path, results_path)
        else:
            print(f"Skipping subfolder: {subfolder}")
    print("Processing complete!")

if __name__ == "__main__":
    main()


