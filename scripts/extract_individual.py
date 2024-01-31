import os, platform
import shutil
from src.tools.find_earliest_date_dir import find_earliest_date_dir


# Define the base directories
if platform.system() == 'Windows':
    if 'gt8mar' in os.getcwd():
        base_directory = 'C:\\Users\\gt8mar\\capillary-flow\\data'
        result_directory = 'C:\\Users\\gt8mar\\capillary-flow\\results\\segmented\\individual_caps_original'
        os.makedirs(result_directory, exist_ok=True)
    else:
        base_directory = 'C:\\Users\\gt8ma\\capillary-flow\\data'
        result_directory = 'C:\\Users\\gt8ma\\capillary-flow\\results\\segmented\\individual_caps_original'
        os.makedirs(result_directory, exist_ok=True)
else:
    base_directory = '/hpc/projects/capillary-flow/data'
    result_directory = '/hpc/projects/capillary-flow/results/segmented/individual_caps_original'
    os.makedirs(result_directory, exist_ok=True)

# Loop through the file tree
for participant in os.listdir(base_directory):
    participant_path = os.path.join(base_directory, participant)
    date = find_earliest_date_dir(participant_path)
    date_path = os.path.join(participant_path, date)
    for location in os.listdir(date_path):
        location_path = os.path.join(date_path, location)
        # Define the source path
        source_path = os.path.join(location_path, 'segmented', 'hasty', 'individual_caps_original')

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


print("Files copied successfully.")