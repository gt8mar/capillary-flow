"""
Filename: crop.py
-------------------------------------------------------------
This file turns a group of files into a single tiff movie file. It also crops the files by 15 rows at the top. 
by: Marcus Forst
"""

import os
import glob
import re
import time
import numpy as np
import tifffile
from src.tools import get_images

def main(SET='set_01', sample = 'sample_000'):
    input_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\raw', str(SET), str(sample), 'vid')              # 'C:\\Users\\gt8mar\\Desktop\\data\\221010'
    processed_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample))
    output_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'A_cropped')
    print(f'the input folder is {input_folder}')
    print(f'the output folder is {output_folder}')
    if 'A_cropped' not in os.listdir(processed_folder):
        os.mkdir(output_folder)
    images = get_images.main(input_folder)
    image_stack = []
    for i in range(len(images)):
        image = tifffile.imread(os.path.join(input_folder, images[i]))
        # # This chops the image into smaller pieces (important if there has been motion correction)
        cropped_image = image[15:]
        image_stack.append(cropped_image)
    image_stack = np.array(image_stack)
    print(image_stack.shape)
    stack_name = f'{SET}_{sample}_stack.tif'
    tifffile.imwrite(os.path.join(output_folder, stack_name), image_stack, photometric='minisblack')  # minisblack means grayscale with 0 as black
    print(f"finished cropping {SET} {sample}")
    return 0

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))