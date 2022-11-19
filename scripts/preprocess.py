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
from src.tools import pic2vid
from src import crop
from src import write_background_file
# import correlation_with_cap_selection

# CAPILLARY_ROW = 565
# CAPILLARY_COL = 590
# BKGD_COL = 669
# BKGD_ROW = 570

SET = "set_01"
processed_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', SET)

def a_preprocess():
    for i in range(12):
        sample = 'sample_' + str(i+1).zfill(3)
        if sample not in os.listdir(processed_folder):
            os.makedirs(os.path.join(processed_folder, sample, "A_cropped", "vid"))
            os.makedirs(os.path.join(processed_folder, sample, "B_stabilized", "vid")) 
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

