"""
Filename: analyze.py
------------------------------------------------------
This program runs a sequence of python programs to analyze capillaries.
It requires segmented capillaries to run. 
By: Marcus Forst
"""

import time
import os
from src.analysis import auto_corr
from src.analysis import correlation
from src import make_kymograph
from src import find_centerline

# CAPILLARY_ROW = 565
# CAPILLARY_COL = 590
# BKGD_COL = 669
# BKGD_ROW = 570

SET = "set_01"
sample = "sample_000"
processed_folder = os.path.join('C:\\Users\\ejerison\\capillary-flow\\data\\processed', SET)

def G_correlation():
    # path = os.path.join(UMBRELLA_FOLDER, folder)
    # segmented_file_name = folder + '0000segmented'
    # correlation_with_cap_selection.main(path, UMBRELLA_FOLDER_MOCO, segmented_file_name)
    # auto_corr.main(UMBRELLA_FOLDER_MOCO, CAPILLARY_ROW, CAPILLARY_COL, BKGD_ROW, BKGD_COL)
    # correlation.main(UMBRELLA_FOLDER_MOCO)
    # print(f'finished correlation')
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
    for i in range(12,21):
        sample = 'sample_' + str(i).zfill(3)

        find_centerline.main(SET, sample, write = True)
        print("-------------------------------------")
        print(f"{sample} Centerline Runtime: {time.time() - ticks}")
        ticks = time.time()

        make_kymograph.main(SET, sample, write = True)    
        print("-------------------------------------")
        print(f"{sample} Blood-Flow Runtime: {time.time() - ticks}")
        ticks = time.time()
        # correlation.main(SET, sample, verbose = False, write = True)
    print("-------------------------------------")
    print("Total Pipeline Runtime: " + str(time.time() - ticks_first))

