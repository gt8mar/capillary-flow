"""
Filename: enumerate_capillaries.py
----------------------------
This program takes a 2D numpy array of a capillary image and returns a 3D numpy array of enumerated capillaries.

By: Marcus Forst
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import cv2

def enumerate_capillaries(image, test = False, verbose = False, write = False, write_path = None):
    """
    This function finds the number of capillaries and returns an array of images with one
    capillary per image.
    :param image: 2D numpy array
    :param short: boolian, if you want to test using one capillary. Default is false.
    :return: 3D numpy array: [capillary index, row, col]
    """
    row, col = image.shape
    print(row, col)
    contours = measure.find_contours(image, 0.8)
    print("The number of contours is: " + str(len(contours)))
    if test:
        contour_array = np.zeros((1, row, col))
        for i in range(1):
            grid = np.array(measure.grid_points_in_poly((row, col), contours[i]))
            contour_array[i] = grid
            # plt.plot(contours[i][:, 1], contours[i][:, 0], linewidth=2)   # this shows all of the enumerated capillaries
        # plt.show()
        return contour_array
    else:
        contour_array = np.zeros((len(contours), row, col))
        if  verbose or write:
            fig = plt.figure(figsize = (12,10))
            ax = fig.add_subplot(111)
        for i in range(len(contours)):
            grid = np.array(measure.grid_points_in_poly((row, col), contours[i]))
            contour_array[i] = grid
            if verbose or write:
                ax.plot(contours[i][:, 1], contours[i][:, 0], linewidth=2, label = "capillary " + str(i)) #plt.imshow(contour_array[i])   # plt.plot(contours[i][:, 1], contours[i][:, 0], linewidth=2) this shows all of the enumerated capillaries
                # plt.show()
        if verbose or write:
            ax.invert_yaxis()
            ax.legend(loc = 'center left')
            if write:
                fig.savefig(write_path)
            if verbose:
                plt.show()
        # check each contour to see if it fits inside another contour
        # if it does, subtract the smaller contour from the larger contour
        for i in range(contour_array.shape[0]):
            for j in range(contour_array.shape[0]):
                if i != j:
                    if np.all(np.isin(contour_array[i], contour_array[j])):
                        print('ya')
                        contour_array[j] = contour_array[j] - contour_array[i]
                        # remove contour_array[i] from contour_array
                        contour_array[i] = np.zeros((row, col))
        # remove empty contours
        contour_array = contour_array[~np.all(contour_array == 0, axis=(1, 2))]
        print(f'the new number of capillaries is {contour_array.shape[0]}')
        return contour_array

if __name__ == "__main__":
    image_path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\segmented\\set01_part09_230414_loc02_vid22_seg.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contour = enumerate_capillaries(image, test = False, verbose = False)
    plt.imshow(contour[1])
    plt.show()