"""
Filename: pipeline.py
------------------------------------------------------
This program runs a sequence of python programs to write a background
file, a corresponding video, segment capillaries, and calculate
flow rates.
By: Marcus Forst
"""

import time
import os, sys, re
from src import write_background_file
# from src import segment
from src.tools.find_earliest_date_dir import find_earliest_date_dir

SET = "set01"

def list_only_folders(path):
    """
    This function returns a list of only folders in a given path.

    Args:
        path (str): the path to the folder to be searched
    Returns:
        list: a list of folders in the given path
    """
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def main():
    """
    Write background file, corresponding video, segment capillaries, and calculate flow rates.

    Args:
        None
    Returns:
        0 if successful
    Saves:
        background file
        video
        segmented file
    """

    """ Write Background """
    i = sys.argv[1]
    print(i)
    ticks_total = time.time()
    participant = 'part' + str(i).zfill(2) 
    participant_path = os.path.join('/hpc/projects/capillary-flow/data', participant)    
    date = find_earliest_date_dir(participant_path)
    
    locations = os.listdir(os.path.join('/hpc/projects/capillary-flow/data', participant, date))
    for location in locations:
        # Omit locScan, locTemp, and locEx from this analysis
        if location == 'locScan' or location == 'locTemp' or location == 'locEx':
            continue
        else:
            ticks = time.time()
            video_folders = os.listdir(os.path.join('/hpc/projects/capillary-flow/data', 
                                                participant, date, location, "vids"))
            for video in video_folders:
                path =  os.path.join('/hpc/projects/capillary-flow/data', participant, date, location, "vids", video)
                write_background_file.main(path, color = True)
                print(f'video {video}')
                print(str(ticks-time.time()))

    print(f'finished {participant} from the {date}')
    print(str(ticks_total-time.time()))    
   
    return 0


"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    print("Run full pipeline")
    print("-------------------------------------")
    ticks_first = time.time()
    ticks = time.time()
    main()  

    print("-------------------------------------")
    print("Total Pipeline Runtime: " + str(time.time() - ticks_first))

