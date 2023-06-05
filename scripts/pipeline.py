"""
Filename: pipeline.py
------------------------------------------------------
This program runs a sequence of python programs to write a background
file, a corresponding video, segment capillaries, and calculate
flow rates.
By: Marcus Forst
"""

import time
import os, sys, re
from src import write_background_file
# from src import segment


SET = "set_01"

def clean_string(string="(21', '22', '23', '24', '25)\r"):
    """
    This function takes a string of numbers and returns a list of numbers.

    Args:
        string (str): a string of numbers

    Returns:
        list: a list of numbers    
    """
    input_string = string
    cleaned_string = input_string.replace("(", "").replace(")", "").replace("'", "").replace("\r", "")
    number_list = [int(num) for num in cleaned_string.split(", ")]
    return number_list

def extract_numbers_from_string(string):
    """
    This function takes a string of numbers and returns a list of numbers.

    Args:
        string (str): a string of numbers

    Returns:
        list: a list of numbers    
    """
    numbers = re.findall(r'\d+', string)
    numbers = [int(num) for num in numbers]
    return numbers

def main():
    """
    Write background file, corresponding video, segment capillaries, and calculate flow rates.

    Args:
        None
    Returns:
        0 if successful
    Saves:
        background file
        video
        segmented file
    """

    """ Write Background """
    i = sys.argv[1]
    print(i)
    ticks_total = time.time()
    participant = 'part' + str(i).zfill(2) 
    # date = os.listdir(os.path.join('/hpc/projects/capillary-flow/data', participant))
    # videos = os.listdir(os.path.join('/hpc/projects/capillary-flow/data', participant, date[0]))
    # for video in videos:
    #     ticks = time.time()
    #     path =  os.path.join('/hpc/projects/capillary-flow/data', participant, date[0], video)
    #     write_background_file.main(path, color = True)
    #     print(f'video {video}')
    #     print(str(ticks-time.time()))
    # """ Segment capillaries using segment.py """
    # # TODO: make this work
    # # segment.main()
    # # find_centerline.py
    # # make_kymograph.py 

    print(f'finished {participant} from the {date[0]}')
    print(str(ticks_total-time.time()))    



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

