"""
Filename: delete_seg_folders.py
------------------------------------------------------
This program deletes all size files in each location folder

"""

import os
import shutil
from src.tools.find_earliest_date_dir import find_earliest_date_dir

def delete_centerlines_files(source_folder):
    """
    This function removes the folders created by the size pipeline in the segmented folder if they exist.

    Args:
        source_folder (str): the path to the data folder
    """
    for i in range(9,21):
        participant = 'part' + str(i).zfill(2)
        date = find_earliest_date_dir(os.path.join(source_folder, participant))
        for location in os.listdir(os.path.join(source_folder, participant, date)):
            registered_fp = os.path.join(source_folder, participant, date, location, 'segmented', 'registered')
            if os.path.exists(registered_fp):
                shutil.rmtree(registered_fp)

            projcaps_fp = os.path.join(source_folder, participant, date, location, 'segmented', 'proj_caps')
            if os.path.exists(projcaps_fp):
                shutil.rmtree(projcaps_fp)

            mocoreg_fp = os.path.join(source_folder, participant, date, location, 'segmented', 'moco_registered')     
            if os.path.exists(mocoreg_fp):
                shutil.rmtree(mocoreg_fp)   

            indicapstrans_fp = os.path.join(source_folder, participant, date, location, 'segmented', 'individual_caps_translated')     
            if os.path.exists(indicapstrans_fp):
                shutil.rmtree(indicapstrans_fp) 

            indicapsorig_fp = os.path.join(source_folder, participant, date, location, 'segmented', 'individual_caps_original')     
            if os.path.exists(indicapsorig_fp):
                shutil.rmtree(indicapsorig_fp)

if __name__ == "__main__":
    source_folder = "/hpc/projects/capillary-flow/data"
    delete_centerlines_files(source_folder)