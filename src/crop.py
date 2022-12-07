"""
Filename: crop.py
-------------------------------------------------------------
This file turns a group of files into a single tiff movie file. It also crops the files by 15 rows at the top. 
by: Marcus Forst
"""

import os
import time
import numpy as np
import cv2
from src.tools.get_images import get_images
from skimage.transform import resize, rescale, downscale_local_mean     # can use any of these to downscale image

def main(SET='set_01', sample = 'sample_000', downsample = False):
    input_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\raw', str(SET), str(sample), 'vid')              # 'C:\\Users\\gt8mar\\Desktop\\data\\221010'
    processed_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample))
    output_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'A_cropped\\vid')
    if 'A_cropped' not in os.listdir(processed_folder):
        os.makedirs(os.path.join(processed_folder, "A_cropped", "vid"))
    images = get_images(input_folder)
    for i in range(len(images)):
        image = np.array(cv2.imread(os.path.join(input_folder, images[i]), cv2.IMREAD_GRAYSCALE))
        # This chops the image into smaller pieces (important if there has been motion correction)
        cropped_image = image[14:]
        if downsample:
            cropped_image = downscale_local_mean(cropped_image, (2,2))
            print(cropped_image)
            cropped_image = cropped_image.astype(int)
            print(cropped_image)
        cv2.imwrite(os.path.join(output_folder, images[i]), cropped_image.astype('uint8'))    # minisblack means grayscale with 0 as black
    print(f"finished cropping {SET} {sample}")
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