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

# CAPILLARY_ROW = 565
# CAPILLARY_COL = 590
# BKGD_COL = 669
# BKGD_ROW = 570

SET = "set_01"
SAMPLE = "sample_000"
processed_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', SET)

def E_centerline():
    for i in range(12):
        sample = 'sample_' + str(i).zfill(3)
        os.makedirs(os.path.join(processed_folder, sample, "E_centerline"))
        print(f'finished sample {i}')

def G_correlation():
    for i in range(12):
        sample = 'sample_' + str(i).zfill(3)
        os.makedirs(os.path.join(processed_folder, sample, "G_correlation"))
    #     path = os.path.join(UMBRELLA_FOLDER, folder)
    #     segmented_file_name = folder + '0000segmented'
    #     correlation_with_cap_selection.main(path, UMBRELLA_FOLDER_MOCO, segmented_file_name)
    #     auto_corr.main(UMBRELLA_FOLDER_MOCO, CAPILLARY_ROW, CAPILLARY_COL, BKGD_ROW, BKGD_COL)
    #     correlation.main(UMBRELLA_FOLDER_MOCO)
        print(f'finished sample {i}')




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
    # done in preprocess.py

    """ Step B: Stabilize using moco in imagej """
    # done in imagej

    """ Step C: Write Background """
    # done in pipeline.py

    """ Step D: Segmentation """
    # done in hasty.ai

    """ Step E: Find centerline """
    E_centerline()
    
    print("-------------------------------------")
    print("Centerline Runtime: " + str(time.time() - ticks))
    ticks = time.time()
 


    """ Step G: Calculate correlation """
    G_correlation()

    print("-------------------------------------")
    print("Correlation Runtime: " + str(time.time() - ticks))
    ticks = time.time()

    """ """





    print("-------------------------------------")
    print("Total Pipeline Runtime: " + str(time.time() - ticks_first))

