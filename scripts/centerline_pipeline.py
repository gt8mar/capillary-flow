"""
Filename: centerline_pipeline.py
------------------------------------------------------
This program runs a sequence of python programs to find 
centerlines for segmented capillaries 

By: Marcus Forst
"""

import os, sys, gc, time
from src import find_centerline
from src import make_kymograph
from src.tools.find_earliest_date_dir import find_earliest_date_dir

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
    date = find_earliest_date_dir(os.path.join('/hpc/projects/capillary-flow/data', participant))
    locations = os.listdir(os.path.join('/hpc/projects/capillary-flow/data', participant, date))
    for location in locations:
        # Omit locScan, locTemp, and locEx from this analysis
        if location == 'locScan' or location == 'locTemp' or location == 'locEx':
            continue
        else:
            print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(f"beginning centerlines and kymographs for location {location}")
            print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            ticks = time.time()
            location_path =  os.path.join('/hpc/projects/capillary-flow/data', participant, date, location)
            find_centerline.main(location_path, verbose=False, write=True)
            print(f"completed centerlines for location {location} in {ticks-time.time()} seconds")

    print(f'finished {participant} from the date {date} in {ticks_total-time.time()} seconds')    
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

