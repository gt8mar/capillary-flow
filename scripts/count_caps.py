"""
Filename: count_caps.py
--------------------------------------
This file navigates into the individual_caps_original
directory and counts the number of capillaries. 
It then compares to the number of capillaries in the
results/centerlines directory. 

By: Marcus Forst
"""

import os, platform
import numpy as np
from src.tools.find_earliest_date_dir import find_earliest_date_dir
import shutil

def main():
    """
    This function navigates into the individual_caps_original
    directory and counts the number of capillaries.
    It then compares to the number of capillaries in the
    results/centerlines directory.

    Args:
        None

    Returns:
        0, if the program runs successfully
    """
    # define the path to the data folder
    if platform.system() == 'Windows':
        computer_name = platform.node()
        print(f'running on {computer_name}')
        if computer_name == 'LAPTOP-I5KTBOR3':
            data_folder = "C:\\Users\\gt8ma\\capillary-flow\\data"
        else:
            data_folder = "C:\\Users\\gt8mar\\capillary-flow\\data"
    else:
        data_folder = "/hpc/projects/capillary-flow/data"
    
    for i in range(9,21):
        
        participant = 'part' + str(i).zfill(2)
        date = find_earliest_date_dir(os.path.join(data_folder, participant))
        for location in os.listdir(os.path.join(data_folder, participant, date)):
            print(location)
            if location == "locEx" or location == "locTemp" or location == "locScan":
                continue
            if os.path.exists(os.path.join(data_folder, participant, date, location, 'segmented', 'hasty', 'individual_caps_original')) == False:
                continue
            if os.path.exists(os.path.join(data_folder, participant, date, location, 'centerlines')) == False:
                continue
            individual_caps_file_counter = 0
            centerlines_file_counter = 0
            # Count all files in the centerlines
            for file in os.listdir(os.path.join(data_folder, participant, date, location, 'centerlines','coords')):
                if os.path.isdir(os.path.join(data_folder, participant, date, location, 'centerlines','coords', file)):
                    continue
                else:
                    centerlines_file_counter += 1

            # Count all files in the individual_caps_original
            for file in os.listdir(os.path.join(data_folder, participant, date, location, 'segmented', 'hasty', 'individual_caps_original')):
                if os.path.isdir(os.path.join(data_folder, participant, date, location, 'segmented', 'hasty', 'individual_caps_original', file)):
                    continue
                elif file.endswith('.png'):
                    individual_caps_file_counter += 1
                else:
                    continue 

            # Compare the two numbers
            if individual_caps_file_counter != centerlines_file_counter:
                print(f'{participant} location {location} has {individual_caps_file_counter} individual_caps and {centerlines_file_counter} centerlines')
                print(f'{participant} has {individual_caps_file_counter - centerlines_file_counter} more individual_caps_original than centerlines')
            else:
                print(f'{participant} {location} same; {individual_caps_file_counter}')

    return 0 

if __name__ == "__main__":
    main()