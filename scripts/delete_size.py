"""
Filename: delete_size.py
------------------------------------------------------
This program deletes all size files in each location folder

"""

import os
import shutil
from src.tools.find_earliest_date_dir import find_earliest_date_dir

def delete_size_files(source_folder):
    """
    This function removes all files and folders within the size folder

    Args:
        source_folder (str): the path to the data folder
    """
    for i in range(9,21):
        participant = 'part' + str(i).zfill(2)
        date = find_earliest_date_dir(os.path.join(source_folder, participant))
        #remove size folder in locations
        for location in os.listdir(os.path.join(source_folder, participant, date)):
            size_fp = os.path.join(source_folder, participant, date, location, 'size')
            if os.path.exists(size_fp):
                shutil.rmtree(size_fp)
        #remove size folder in results
        size_results_fp = '/hpc/projects/capillary-flow/results/size'
        if os.path.exists(size_results_fp):
            shutil.rmtree(size_results_fp)

if __name__ == "__main__":
    source_folder = "/hpc/projects/capillary-flow/data"
    delete_size_files(source_folder)