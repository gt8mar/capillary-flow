"""
Filename: frawg_sd.py
------------------------------------------------------
This file takes a series of images, creates a background file, and creates a folder with
background subtracted files.

By: Marcus Forst
"""

import os
import cv2
import numpy as np


"""
USER INSTRUCTIONS:

1. Set the main_folder variable to the path of the folder containing the subfolders with the .tiff files.
Example: main_folder = "/path/to/folder"
"""




def process_subfolder(subfolder_path, output_path):
    """
    Process a subfolder containing a series of .tiff files and calculate the standard deviation image.

    Args:
        subfolder_path (str): The path to the subfolder containing the .tiff files.
        output_path (str): The path to save the standard deviation image.

    Returns:
        None
    """
    # List all .tiff files in the subfolder
    frame_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith('.tiff')]

    # Read all frames into a list
    frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in frame_files]

    # Convert the list of frames to a numpy array
    frames_array = np.array(frames)

    # Calculate the standard deviation for each pixel
    stdevs = np.std(frames_array, axis=0)

    # Normalize the standard deviation values to the range [0, 255]
    stdevs = cv2.normalize(stdevs, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the standard deviation to uint8
    stdevs_uint8 = np.uint8(stdevs)

    # Save the standard deviation image in .tiff format
    cv2.imwrite(output_path, stdevs_uint8)

def main():
    """
    Process all subfolders in the main folder and calculate the standard deviation image for each subfolder.
    """
    # Define the main folder and the output folder
    main_folder = "/path/to/folder"
    output_folder = os.path.join(main_folder, "stdevs")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each subfolder in the main folder
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        
        # Ensure it's a directory
        if os.path.isdir(subfolder_path):
            # Define the output path for the standard deviation image
            output_path = os.path.join(output_folder, f"SD_{subfolder}.tiff")
            
            # Process the subfolder
            process_subfolder(subfolder_path, output_path)

    print("Processing complete!")

if __name__ == "__main__":
    main()


