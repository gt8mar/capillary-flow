"""
Filename: delete_kymo.py
------------------------------------------------------
This program copies all the kymograph TIFF files from the data folder to the results folder.

By: Marcus Forst
"""

import os
import shutil
from src.tools.find_earliest_date_dir import find_earliest_date_dir

def delete_kymo_files(source_folder):
    """
    This function searches for kymograph TIFF files in the data folder and copies them to the results folder.

    Args:
        source_folder (str): the path to the data folder
        destination_folder (str): the path to the results folder
    Returns:
        list: a list of the paths to the TIFF files that have been copied
    """
    for i in range(9,21):
        participant = 'part' + str(i).zfill(2)
        date = find_earliest_date_dir(os.path.join(source_folder, participant))
        for location in os.listdir(os.path.join(source_folder, participant, date)):
            # delete all files and folders in the centerlines
            folder_list = []
            file_list = []
            # check if path is empty:
            if os.path.exists(os.path.join(source_folder, participant, date, location, 'kymographs')) == False:
                continue
            
            # iterate through files and folders until all are deleted
            for file in os.listdir(os.path.join(source_folder, participant, date, location, 'kymographs')):
                if os.path.isdir(os.path.join(source_folder, participant, date, location, 'kymographs', file)):
                    folder_list.append(file)
                else:
                    file_list.append(file)
            for file in file_list:
                os.remove(os.path.join(source_folder, participant, date, location, 'kymographs', file))
            for folder in folder_list:
                shutil.rmtree(os.path.join(source_folder, participant, date, location, 'kymographs', folder))            
    return 0

if __name__ == "__main__":
    source_folder = "/hpc/projects/capillary-flow/data"
    delete_kymo_files(source_folder)