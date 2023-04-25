"""
Filename: pipeline.py
------------------------------------------------------
This program runs a sequence of python programs to write a background
file and corresponding video.
By: Marcus Forst
"""

import time
import os
from src import write_background_file
from src import segment

SET = "set_01"

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
   # TODO: iterate through participants, dates, and vids
    for i in range(43,57):
        sample = # we probably want some participant date and vid info here. original: 'sample_' + str(i).zfill(3)
        # TODO: refactor write_background_file to work with participant, date ,vid folder format
        write_background_file.main(SET, sample, color = True)
        """ Segment capillaries using segment.py """
        # TODO: marcus make this work
        segment.main()
        find_centerline.py
        make_kymograph.py 


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

