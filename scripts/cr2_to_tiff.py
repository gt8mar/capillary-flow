"""
Filename: scripts/cr2_to_tiff.py

Description:
    Convert CR2 files to TIFF files.

Author:
    Marcus Forst
"""

import os
import sys

import rawpy
import tifffile
import numpy as np
import cv2

def main():
    """
    Main function to convert CR2 files to TIFF files.
    """
    image_folder = "H:\\WkSleep_Trans_Up_to_25-5-1_Named"
    output_folder = "H:\\WkSleep_Trans_Up_to_25-5-1_Named_TIFF"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all CR2 files in the input folder
    cr2_files = [f for f in os.listdir(image_folder) if f.endswith(".CR2")]

    # Convert each CR2 file to TIFF
    for cr2_file in cr2_files:
        # Read the CR2 file using rawpy
        with rawpy.imread(os.path.join(image_folder, cr2_file)) as raw:
            # Convert the raw data to a numpy array
            image = raw.postprocess()

            # Get the base name of the CR2 file (without extension)
            base_name = os.path.splitext(cr2_file)[0]

            # Save the TIFF file with compression
            tiff_file = os.path.join(output_folder, f"{base_name}.tiff")
            tifffile.imwrite(
                tiff_file,
                image,
                compression=("zlib", 9),  # Using maximum compression level
                photometric="rgb"         # keep colours correctly tagged
            )  

            # Print a message to indicate that the file has been converted
            print(f"Converted {cr2_file} to {tiff_file}")

if __name__ == "__main__":
    main()







