"""
Filename: pipeline.py
------------------------------------------------------
This program runs a sequence of python programs to write a background
file and corresponding video.
By: Marcus Forst
"""

import time
import os
from src import write_background_file_training

SET = "set_02"
SAMPLE = "sample_000"
processed_folder = os.path.join('C:\\Users\\ejerison\\capillary-flow\\data\\processed', SET)

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

    """ Write Background """
    # for folder in os.listdir(UMBRELLA_FOLDER):
    #     path = os.path.join(UMBRELLA_FOLDER, folder)
    #     write_background_file.main(folder, path, DATE, PARTICIPANT)
    for i in range(43,57):
        sample = 'sample_' + str(i).zfill(3)
        write_background_file_training.main(SET, sample, color = True)
        print('finished one')
        print(str(ticks-time.time()))    
    print("-------------------------------------")
    print("Background Runtime: " + str(time.time() - ticks))
    ticks = time.time()


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

    print("-------------------------------------")
    print("Total Pipeline Runtime: " + str(time.time() - ticks_first))

