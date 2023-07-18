"""
File: count_capillaries.py
-------------------------
This program counts the maximum number of TIFF files in a 'kymo' folder.

By: Marcus Forst
"""

import os

def count_max_tiff_files(folder_path):
    max_tiff_count = 0
    for root, dirs, files in os.walk(folder_path):
        if "kymo" in dirs:
            kymo_folder = os.path.join(root, "kymo")
            tiff_count = len([file for file in os.listdir(kymo_folder) if file.lower().endswith(".tiff") or file.lower().endswith(".tif")])
            if tiff_count > max_tiff_count:
                max_tiff_count = tiff_count
    return max_tiff_count

source_folder = "/hpc/projects/capillary-flow/data"
max_tiff_count = count_max_tiff_files(source_folder)

print("Maximum number of TIFF files in a 'kymo' folder:", max_tiff_count)