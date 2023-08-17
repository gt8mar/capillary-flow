"""
Filename: enumerate_capillaries.py
----------------------------
This program takes a 2D numpy array of a capillary image and returns a 3D numpy array of enumerated capillaries.

By: Marcus Forst
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

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
        for i in range(len(contours)):
            for j in range(len(contours)):
                if i != j:
                    if np.all(np.isin(contours[i], contours[j])):
                        contour_array[j] = contour_array[j] - contour_array[i]
                        # remove contour_array[i] from contour_array
                        contour_array[i] = np.zeros((row, col))
        # remove empty contours
        # Calculate the sum of absolute values for each image
        image_sums = np.sum(np.abs(contour_array), axis=(1, 2))
        # Find the indices of non-blank images
        non_blank_indices = np.nonzero(image_sums)
        # Select only the non-blank images
        contour_array = contour_array[non_blank_indices]
        print(f'the new number of capillaries is {contour_array.shape[0]}')
        return contour_array

if __name__ == "__main__":
    enumerate_capillaries()