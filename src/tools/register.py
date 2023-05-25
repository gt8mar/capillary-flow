"""
Filename: register.py
-------------------------------------------------------------
This file 
by: Gabby Rincon
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage.registration import phase_cross_correlation
from skimage import exposure, util
from skimage.filters import gaussian


def main():
    image1 = cv2.imread("D:\\Marcus\\gabby test data\\part09\\230414\\vid1\\moco\\vid1_moco_0000.tif", cv2.COLOR_BGR2GRAY)
    image2 = cv2.imread("D:\\Marcus\\gabby test data\\part09\\230414\\vid2\\moco\\vid2_moco_0000.tif", cv2.COLOR_BGR2GRAY)

    #test with same image but shifted- does work
    '''empty1 = np.empty_like(image1)
    offset1 = empty1 + np.mean(image1)
    offset_x = 10
    offset_y = 50
    height, width = image1.shape
    for x in range(height):
        for y in range(width):
            if 0 <= x + offset_x < width and 0 <= y + offset_y < height:
                offset1[y,x] = image1[y+offset_y,x+offset_x]'''

    #test with increased contrast- doesn't work
    '''image1norm = util.img_as_float(image1)
    image2norm = util.img_as_float(image2)
    image1contrast = exposure.rescale_intensity(image1norm)
    image2contrast = exposure.rescale_intensity(image2norm)'''
    
    #gaussians
    image1hp = image1 - gaussian(image1)
    image2hp = image2 - gaussian(image2)
    #plt.imshow(image1hp)
    #plt.show()
    #cv2.imshow("2", image2hp)

    #shift = phase_cross_correlation(image1contrast, image2contrast)[0]
    shift = phase_cross_correlation(image1hp, image2hp)[0]
    #shift = cv2.phaseCorrelate(np.float32(image1), np.float32(image2))
    print(shift)

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