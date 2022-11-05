"""
Filename: crop.py
-------------------------------------------------------------
This file turns a group of files into a group of files that are slightly smaller
by: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
"""


import os
import glob
import re
import time
import numpy as np
import cv2
from src.tools import get_images

def main(set, sample):
    input_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\raw', str(set), str(sample), 'vid')              # 'C:\\Users\\gt8mar\\Desktop\\data\\221010'
    output_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(set), str(sample), 'A_cropped\\vid')
    print(input_folder)
    print(output_folder)
    os.mkdir(output_folder)
    images = get_images.main(input_folder)
    for i in range(len(images)):
        image = np.array(cv2.imread(os.path.join(input_folder, images[i])))
        # # This chops the image into smaller pieces (important if there has been motion correction)
        cropped_image = image[15:]
        cv2.imwrite(os.path.join(output_folder, images[i]), cropped_image)
    print(f"finished cropping {set} {sample}")
    return 0

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main(set, sample)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))