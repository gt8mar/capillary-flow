from src.tools.align_segmented import align_segmented
from src.tools import group_caps
from src.tools import translate_centerlines
from src.tools import plot_area
from src.tools import plot_size_slopes
from src.tools.find_earliest_date_dir import find_earliest_date_dir
from src.tools.naming_overlay import make_overlays
from src.tools.rename_individual_caps import rename
import os
import sys
import time

def main():
    # Participant number is passed as an argument
    i = sys.argv[1]
    print(f"begin size_pipeline for participant {i}")
    ticks_total = time.time()
    participant = 'part' + str(i).zfill(2) 

    # Load the date and video numbers
    date = find_earliest_date_dir(os.path.join('/hpc/projects/capillary-flow/data', participant))
    locations = os.listdir(os.path.join('/hpc/projects/capillary-flow/data', participant, date))
    for location in locations:
        if location == "locEx" or location == "locTemp" or location == "locScan":
            continue
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"beginning size for location {location}")
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        ticks = time.time()
        location_path =  os.path.join('/hpc/projects/capillary-flow/data', participant, date, location)
        rename(location_path)
        make_overlays(location_path, rename=True)
    

if __name__ == "__main__":
    print("Run full pipeline")
    print("-------------------------------------")
    ticks_first = time.time()
    ticks = time.time()
    main()  

    print("-------------------------------------")
    print("Total Pipeline Runtime: " + str(time.time() - ticks_first))