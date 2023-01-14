"""
Filename: preprocess.py
------------------------------------------------------
This program runs a sequence of python programs to analyze capillaries
By: Marcus Forst
"""

import time
import os
from src import auto_corr
from src import correlation
from src import crop
from src import write_background_file
# import correlation_with_cap_selection

# CAPILLARY_ROW = 565
# CAPILLARY_COL = 590
# BKGD_COL = 669
# BKGD_ROW = 570

SET = "set_01"
processed_folder = os.path.join('C:\\Users\\ejerison\\capillary-flow\\data\\processed', SET)

def a_preprocess():
    for i in range(12,21):
        sample = 'sample_' + str(i).zfill(3)
        crop.main(SET, sample)


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

    """ Step A: Preprocess """
    a_preprocess()
    print("-------------------------------------")
    print(f"Step A: Preprocess Runtime: {time.time() - ticks}")
    ticks = time.time()



    print("-------------------------------------")
    print("Total Preprocess Runtime: " + str(time.time() - ticks_first))

