"""
Filename: frog_sd_pipeline.py
------------------------------------------------------
This program runs a sequence of python programs to find
the standard deviation of a series of images of frogs

By: Marcus Forst
"""

import os, sys, gc, time
from src import find_centerline
from src import make_kymograph
from src.analysis import make_velocities
from src.tools.find_earliest_date_dir import find_earliest_date_dir
from src.tools import frog_sd

def list_only_folders(path):
    """
    This function returns a list of only folders in a given path.

    Args:
        path (str): the path to the folder to be searched
    Returns:
        list: a list of folders in the given path
    """
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def main(test=False):
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
    testing_folder = 'C:\\Users\\ejerison\\capillary-flow\\tests\\24-4-19_WkSl'

    # # Participant number is passed as an argument
    # i = sys.argv[1]
    # print(f"begin kymo_pipeline for participant {i}")
    ticks_total = time.time()

    if test:
        # Testing
        for folder2 in os.listdir(testing_folder):
            if folder2.startswith('Frog'):
                for folder3 in os.listdir(os.path.join(testing_folder, folder2)):
                    if folder3.startswith('Left'):
                        left_folder = os.path.join(testing_folder, folder2, folder3)    
                        print(left_folder)
                        frog_sd.main(left_folder)
                    elif folder3.startswith('Right'):
                        right_folder = os.path.join(testing_folder, folder2, folder3)
                        # frog_sd.main(os.path.join(right_folder, folder2, folder3))
                        print(folder3)
                    else:
                        continue
            else:
                continue
    # Load the date and video numbers
    else:
        for folder in os.listdir('/hpc/projects/capillary-flow/frog'):
            # check if folder is a folder and if it starts with '24'
            if os.path.isdir(os.path.join('/hpc/projects/capillary-flow/frog', folder)) and folder.startswith('27-7-29'):
                for folder2 in os.listdir(os.path.join('/hpc/projects/capillary-flow/frog', folder)):
                    if folder2.startswith('Frog'):
                        for folder3 in os.listdir(os.path.join('/hpc/projects/capillary-flow/frog', folder, folder2)):
                            if folder3.startswith('Left'):
                                left_folder = os.path.join('/hpc/projects/capillary-flow/frog', folder, folder2, folder3) 
                                print(left_folder)   
                                frog_sd.main(left_folder)
                            elif folder3.startswith('Right'):
                                right_folder = os.path.join('/hpc/projects/capillary-flow/frog', folder, folder2, folder3)
                                print(right_folder)
                                frog_sd.main(right_folder)
                            else:
                                continue                    
                    else:
                        continue
        
            else:
                continue
                
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
    main(test=False)  

    print("-------------------------------------")
    print("Total Pipeline Runtime: " + str(time.time() - ticks_first))

