"""
Filename: delete_centerline_folders_gabby.py
------------------------------------------------------
This program deletes all files created by the second half of the size pipeline from each location.
By: Gabby Rincon

"""

import os
import shutil
from src.tools.find_earliest_date_dir import find_earliest_date_dir

def delete_centerlines_folders_files(source_folder):
    """
    This function removes the folders created by the size pipeline in the segmented folder if they exist.

    Args:
        source_folder (str): the path to the data folder
    """
    for i in range(9,21):
        participant = 'part' + str(i).zfill(2)
        date = find_earliest_date_dir(os.path.join(source_folder, participant))
        for location in os.listdir(os.path.join(source_folder, participant, date)):
            translated_centerlines = os.path.join(source_folder, participant, date, location, 'centerlines', 'translated')
            if os.path.exists(translated_centerlines):
                shutil.rmtree(translated_centerlines)
            
            renamed_centerlines = os.path.join(source_folder, participant, date, location, 'centerlines', 'renamed')
            if os.path.exists(renamed_centerlines):
                shutil.rmtree(renamed_centerlines)

if __name__ == "__main__":
    source_folder = "/hpc/projects/capillary-flow/data"
    delete_centerlines_folders_files(source_folder)