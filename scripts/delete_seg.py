"""
Filename: delete_seg.py
------------------------------------------------------
This program deletes all segmentation files.

By: Marcus Forst
"""

import os
import shutil
from src.tools.find_earliest_date_dir import find_earliest_date_dir

def delete_seg_files(source_folder):
    """
    This function searches for segmentation files in the data folder and deletes them.

    Args:
        source_folder (str): the path to the data folder
    Returns:
        None
    """
    for i in range(21,28):
        participant = 'part' + str(i).zfill(2)
        date = find_earliest_date_dir(os.path.join(source_folder, participant))
        for location in os.listdir(os.path.join(source_folder, participant, date)):
            # delete all files and folders in the centerlines
            folder_list = []
            file_list = []
            for file in os.listdir(os.path.join(source_folder, participant, date, location, 'segmented', 'hasty')):
                if os.path.isdir(os.path.join(source_folder, participant, date, location, 'segmented', 'hasty', file)):
                    folder_list.append(file)
                else:
                    file_list.append(file)
            for file in file_list:
                os.remove(os.path.join(source_folder, participant, date, location, 'segmented', 'hasty', file))
            for folder in folder_list:
                shutil.rmtree(os.path.join(source_folder, participant, date, location, 'segmented', 'hasty', folder))            
    return 0

if __name__ == "__main__":
    source_folder = "/hpc/projects/capillary-flow/data"
    delete_seg_files(source_folder)