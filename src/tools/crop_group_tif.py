"""
Filename: crop.py
-------------------------------------------------------------
This file turns a group of files into a single tiff movie file. It also crops the files by 15 rows at the top. 
by: Marcus Forst
"""

import os
import time
import numpy as np
import tifffile
from src.tools import get_images
from skimage.transform import resize, rescale, downscale_local_mean     # can use any of these to downscale image

def add_dust(image, dust_coords):
    """
    # TODO: write this
    """
    for coord in dust_coords:
        image[coord[0]][coord[1]] = 119
        image[coord[0]+1][coord[1]] = 119
        image[coord[0]+1][coord[1]+1] = 119
        image[coord[0]][coord[1]+1] = 119
    return image

def main(SET='set_01', sample = 'sample_000', downsample = False):
    tif_path = os.path.join('C:\\Users\\gt8mar\\caiman_data\\example_movies\\Sue_2x_3000_40_-46.tif')              # 'C:\\Users\\gt8mar\\Desktop\\data\\221010'
    output_folder = os.path.join('C:\\Users\\gt8mar\\caiman_data\\example_movies')
    dust_coords = [[60, 43], [62, 14], [53, 35], [2, 112]]

    with tifffile.TiffFile(tif_path) as tffl:
        image_stack = tffl.asarray(out = 'np.ndarray')
        image_stack = image_stack.astype('int_')
        print(image_stack)
        new_stack = []
        for i in range(len(image_stack)):
            image = image_stack[i]
            # # This chops the image into smaller pieces (important if there has been motion correction)
            # cropped_image = image[14:]
            cropped_image = add_dust(image, dust_coords)
            if downsample:
                cropped_image = downscale_local_mean(cropped_image, (7, 7))
            new_stack.append(cropped_image)
        new_stack = np.array(new_stack).astype('int_')
        stack_name = f'Sue_2x_3000_40_-46_dust.tif'
        tifffile.imwrite(os.path.join(output_folder, stack_name), new_stack, photometric='minisblack')  # minisblack means grayscale with 0 as black
        print(f"finished cropping {stack_name}")
    return 0

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main(downsample=False)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))