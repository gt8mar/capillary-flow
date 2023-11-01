"""
Filename: sort_hasty_seg.py
---------------------------
This module contains a function that sorts the hasty segmentations into the
correct location folders.
"""

import os, platform
import shutil
from src.tools.parse_filename import parse_filename

def main():
    if platform.system() == 'Windows':
        seg_path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\segmented\\hasty\\231101'
    else:
        seg_path = '/hpc/projects/capillary-flow/results/segmented/hasty/231101'

    # get all files in the directory
    files = os.listdir(seg_path)
    # replace _contrast and _background
    # files = [file.replace('_contrast', '').replace('_background', '').replace('set01_','').replace('set_01_', '') for file in files]

    for file in files:
        participant, date, location, video, file_prefix = parse_filename(file)
        filename = file_prefix + '_seg.png'
        # create the new path
        if platform.system() == 'Windows':
            new_path = f'C:\\Users\\gt8mar\\capillary-flow\\data\\{participant}\\{date}\\{location}\\segmented\\hasty\\' 
            os.makedirs(new_path, exist_ok=True)
        else:
            new_path = f'/hpc/projects/capillary-flow/data/{participant}/{date}/{location}/segmented/hasty/'
            os.makedirs(new_path, exist_ok=True)
        # copy the file to the new path
        shutil.copy(os.path.join(seg_path, file), os.path.join(new_path, filename))

    print('Done!')

if __name__ == "__main__":
    main()
    

