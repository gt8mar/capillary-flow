"""
Filename: delete_size.py
------------------------------------------------------
This program deletes all size files in each location folder

"""

import os
import shutil
from src.tools.find_earliest_date_dir import find_earliest_date_dir

def delete_centerlines_files(source_folder):
    """
    This function removes all files and folders within the size folder

    Args:
        source_folder (str): the path to the data folder
    """
    for i in range(9,21):
        participant = 'part' + str(i).zfill(2)
        date = find_earliest_date_dir(os.path.join(source_folder, participant))
        for location in os.listdir(os.path.join(source_folder, participant, date)):
            # delete all files and folders in size
            folder_list = []
            file_list = []
            for file in os.listdir(os.path.join(source_folder, participant, date, location, 'size')):
                item_path = os.path.join(source_folder, participant, date, location, 'size', file)
                if os.path.isdir(os.path.join(source_folder, participant, date, location, 'size', file)):
                    folder_list.append(item_path)
                else:
                    file_list.append(item_path)     
            for file in file_list:
                os.remove(os.path.join(source_folder, participant, date, location, 'size', file))
            for folder in folder_list:
                shutil.rmtree(os.path.join(source_folder, participant, date, location, 'size', folder))            
    return 0

if __name__ == "__main__":
    source_folder = "/hpc/projects/capillary-flow/data"
    delete_centerlines_files(source_folder)