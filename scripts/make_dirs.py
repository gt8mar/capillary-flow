"""
Filename: analyze.py
------------------------------------------------------
This program runs a sequence of python programs to analyze capillaries
By: Marcus Forst
"""

import time
import os
from src import auto_corr
from src import correlation
from src import blood_flow_linear
from src import find_centerline

# CAPILLARY_ROW = 565
# CAPILLARY_COL = 590
# BKGD_COL = 669
# BKGD_ROW = 570

SET = "set_01"
sample = "sample_000"
processed_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', SET)

def make_dirs():
    for i in range(12):
        sample = 'sample_' + str(i).zfill(3)
        os.makedirs(os.path.join(processed_folder, sample, "A_cropped", "vid"))
        os.makedirs(os.path.join(processed_folder, sample, "B_stabilized", "vid")) 
        os.makedirs(os.path.join(processed_folder, sample, "C_background"))
        os.makedirs(os.path.join(processed_folder, sample, "D_stabilized"))
        os.makedirs(os.path.join(processed_folder, sample, "E_centerline"))
        os.makedirs(os.path.join(processed_folder, sample, "F_blood_flow"))
        os.makedirs(os.path.join(processed_folder, sample, "G_correlation"))
        os.makedirs(os.path.join(processed_folder, sample, "H_turbulence"))
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

    """ Make directories """


    print("-------------------------------------")
    print("Correlation Runtime: " + str(time.time() - ticks))
    ticks = time.time()

    """ """





    print("-------------------------------------")
    print("Total Pipeline Runtime: " + str(time.time() - ticks_first))

