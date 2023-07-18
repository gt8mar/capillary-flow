"""
Filename: pipeline.py
------------------------------------------------------
This program runs a sequence of python programs to find 
centerlines for segmented capillaries and make kymographs.

By: Marcus Forst
"""

import os, sys, gc, time
from src.tools import save_cap_vid
from src.tools import find_earliest_date_dir

SET = "set_01"

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
    Finds centerlines for segmented capillaries and makes kymographs.

    Args:
        None

    Returns:
        0 if successful

    Saves:
        centerline files
        kymograph files
    """

    # Participant number is passed as an argument
    i = sys.argv[1]
    print(f"begin cap_vid_pipeline for participant {i}")
    ticks_total = time.time()
    participant = 'part' + str(i).zfill(2) 

    # Load the date and video numbers
    date = find_earliest_date_dir(os.path.join('/hpc/projects/capillary-flow/data', participant))

    videos = os.listdir(os.path.join('/hpc/projects/capillary-flow/data', participant, date))

    # Find centerlines and make kymographs for each video
    for video in videos:
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"beginning cropped capillary videos for: {video}")
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        ticks = time.time()
        path =  os.path.join('/hpc/projects/capillary-flow/data', participant, date[0], video)
        ran = save_cap_vid.main(path)
        if ran == 1:
            print(f"no capillaries found for video {video}")
        else:
            print(f"completed capillary videos for video {video} in {ticks-time.time()} seconds")

    print(f'finished {participant} from the date {date[0]} in {ticks_total-time.time()} seconds')
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

