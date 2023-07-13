"""
Filename: pipeline.py
------------------------------------------------------
This program runs a sequence of python programs to find 
centerlines for segmented capillaries and make kymographs.

By: Marcus Forst
"""

import os, sys, gc, time
from src import find_centerline
from src import make_kymograph

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
    print(f"begin kymo_pipeline for participant {i}")
    ticks_total = time.time()
    participant = 'part' + str(i).zfill(2) 

    # Load the date and video numbers
    date = list_only_folders(os.path.join('/hpc/projects/capillary-flow/data', participant))
    videos = os.listdir(os.path.join('/hpc/projects/capillary-flow/data', participant, date[0]))

    # Find centerlines and make kymographs for each video
    for video in videos:
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"beginning centerlines and kymographs for video {video}")
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        ticks = time.time()
        path =  os.path.join('/hpc/projects/capillary-flow/data', participant, date[0], video)
        find_centerline.main(path, verbose=False, write=True)
        print(f"completed centerlines for video {video} in {ticks-time.time()} seconds")
        
        # Make kymographs
        make_kymograph.main(path, verbose=False, write=True)
        print(f'completed kymographs for video {video} in {ticks-time.time()} seconds')
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


    print(f'finished {participant} from the date {date[0]} in {ticks_total-time.time()} seconds')

    """ Correlation files """
    # for folder in os.listdir(UMBRELLA_FOLDER_MOCO):
    #     path = os.path.join(UMBRELLA_FOLDER, folder)
    #     pic2vid.main(path, folder, DATE, PARTICIPANT)
    #     segmented_file_name = folder + '0000segmented'
    #     correlation_with_cap_selection.main(path, UMBRELLA_FOLDER_MOCO, segmented_file_name)
    #     auto_corr.main(UMBRELLA_FOLDER_MOCO, CAPILLARY_ROW, CAPILLARY_COL, BKGD_ROW, BKGD_COL)
    #     correlation.main(UMBRELLA_FOLDER_MOCO)

    # print("-------------------------------------")
    # print("Correlation Runtime: " + str(time.time() - ticks))
    # ticks = time.time()
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

