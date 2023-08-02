"""
Filename: extract_centerlines.py
------------------------------------------------------
This program copies all the kymograph TIFF files from the data folder to the results folder.

By: Marcus Forst
"""

import os
import shutil

def search_for_centerline_csv_files(source_folder, destination_folder):
    """
    This function searches for kymograph TIFF files in the data folder and copies them to the results folder.

    Args:
        source_folder (str): the path to the data folder
        destination_folder (str): the path to the results folder
    Returns:
        list: a list of the paths to the TIFF files that have been copied
    """
    csv_files = []
    # Walk through the data folder
    for root, dirs, files in os.walk(source_folder):
        if "coords" in dirs:
            centerline_folder = os.path.join(root, "coords")
            for file in os.listdir(centerline_folder):
                if file.lower().endswith(".csv"):
                    source_path = os.path.join(centerline_folder, file)
                    csv_files.append(source_path)
                    destination_path = os.path.join(destination_folder, file)
                    os.makedirs(destination_folder, exist_ok=True)
                    shutil.copy(source_path, destination_path)
    return csv_files

source_folder = "/hpc/projects/capillary-flow/data"
destination_folder = "/hpc/projects/capillary-flow/results/centerlines"
csv_files = search_for_centerline_csv_files(source_folder, destination_folder)

# Print the paths of all the TIFF files that have been copied
for file in csv_files:
    print("Copied:", file)

# Print the path of the destination folder
print("Destination folder:", destination_folder)