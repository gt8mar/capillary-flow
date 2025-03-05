"""
Filename: extract_individual.py
---------------------------------
This file contains a script that extracts individual capillary images 
from the segmented images of participants.

By: Marcus Forst
"""

import os, platform
import shutil
from src.tools.find_earliest_date_dir import find_earliest_date_dir


# Define the base directories
if platform.system() == 'Windows':
    if 'gt8mar' in os.getcwd():
        base_directory = 'C:\\Users\\gt8mar\\capillary-flow\\data'
        result_directory = 'C:\\Users\\gt8mar\\capillary-flow\\results\\segmented\\renamed_individual_caps_original'
        os.makedirs(result_directory, exist_ok=True)
    else:
        base_directory = 'C:\\Users\\gt8ma\\capillary-flow\\data'
        result_directory = 'C:\\Users\\gt8ma\\capillary-flow\\results\\segmented\\renamed_individual_caps_original'
        os.makedirs(result_directory, exist_ok=True)
else:
    base_directory = '/hpc/projects/capillary-flow/data'
    result_directory = '/hpc/projects/capillary-flow/results/segmented/renamed_individual_caps_original'
    os.makedirs(result_directory, exist_ok=True)

# Loop through the file tree for participants part09 through part81
participant_list = [f'part{str(i).zfill(2)}' for i in range(9, 82)]
for participant in participant_list: #os.listdir(base_directory):
    participant_path = os.path.join(base_directory, participant)
    date = find_earliest_date_dir(participant_path)
    date_path = os.path.join(participant_path, date)
    for location in os.listdir(date_path):
        location_path = os.path.join(date_path, location)
        # Define the source path
        source_path = os.path.join(location_path, 'segmented', 'hasty', 'renamed_individual_caps_original')

        # Check if the source path exists and is a directory
        if os.path.isdir(source_path):
            for file in os.listdir(source_path):
                # Check if the file is a PNG
                if file.endswith('.png'):
                    # Define the source and destination file paths
                    source_file = os.path.join(source_path, file)
                    destination_file = os.path.join(result_directory, file)

                    # if the file isn't already in the destination directory
                    if not os.path.isfile(destination_file):
                        # Copy the file
                        shutil.copy(source_file, destination_file)
                    else:
                        print(f'{file} already exists in {result_directory}')
        else:
            print(f'{source_path} does not exist')


print("Files copied successfully.")