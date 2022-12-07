"""
Filename: blood_flow_linear.py
------------------------------------------------------
This file calculates the blood flow rate statistically using the centerline.

By: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
average_in_circle credit: Nicolas Gervais (https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle)
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
from src.tools.get_images import get_images
from src.tools.load_image_array import load_image_array
from src.tools.load_csv_list import load_csv_list

FILEFOLDER = 'C:\\Users\\gt8mar\\Desktop\\data\\221010\\vid4_moco'
SKELETON_FILE = 'vid4_test_skeleton_coords_7.csv'
# SKELETON_FILE_LIST = ['test_skeleton_coords_2.csv','test_skeleton_coords_3.csv','test_skeleton_coords_4.csv','test_skeleton_coords_5.csv','test_skeleton_coords_7.csv']
PIXELS_PER_UM = 2

def average_array(array):
    """
    This returns an averaged array with half length of the input 1d array
    :param array: 1d numpy array length 2n
    :return: 1d array length n
    """
    if np.mod(len(array), 2) == 0:
        return (array[::2] + array[1::2]) // 2
    else:
        return (array[:-1:2] + array[1::2]) // 2
def build_centerline_vs_time(image, skeleton_coords, long = False, offset = False):
    """
    This function takes an image and text file (default: csv) of the coordinates of a
    skeleton and outputs an image of the centerline pixel values vs time.
    :param image: 3D numpy array (time, row, col)
    :param skeleton_txt: 2D text file to be read into the function
    :return: centerline_array: 2D numpy array that shows the pixels of the centerline vs time.
    """
    # if offset:
    #     row_offset = np.full((skeleton_coords.shape[0], 1), 0)
    #     col_offset = np.full((skeleton_coords.shape[0], 1), -100)
    #     offset_array = np.hstack((row_offset, col_offset))
    #     skeleton_coords = skeleton_coords + offset_array
    centerline_array = np.zeros((skeleton_coords.shape[0], image.shape[0]))
    if long == False:
        for i in range(skeleton_coords.shape[0]):
            for j in range(image.shape[0]):
                centerline_array[i][j] = image[j][skeleton_coords[i][0]][skeleton_coords[i][1]]
    if long == True:
        for i in range(skeleton_coords.shape[0]):
            for j in range(image.shape[0]):
                centerline_array[i][j] = average_in_circle(image[j], skeleton_coords[i][0], skeleton_coords[i][1], radius=5)
    return centerline_array
def average_in_circle(image, row, col, radius = 5):
    """
    This function inputs an image and a coordinate and outputs the average of a circle of
    pixels surrounding the coordinate with specified radius.
    :param image: 2D numpy array
    :param row: integer, the row coordinate you want to average around
    :param col: integer, the column coordinate you want to average around
    :param radius: the radius you want to average over
    :return averaged_value: the averaged pixel value.
    """
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])

    mask = (x[np.newaxis, :] - col) ** 2 + (y[:, np.newaxis] - row) ** 2 < radius ** 2
    circle_values = image[mask].reshape(1,-1)
    return np.mean(circle_values)
def normalize_image(image):
    image = image - np.min(image)
    image = image / np.max(image)
    image *= 255
    image = np.rint(image)
    return image.astype('uint8')
def normalize_rows(image):
    """ this function normalizes the rows of an image """
    # TODO: this is not clearly the best way to normalize
    average_col = np.mean(image, axis = 1) # averages along the rows to give one big column
    std_col = np.std(image, axis = 1)
    big_average = np.tile(average_col, (image.shape[1], 1)).transpose()
    big_std = np.tile(std_col, (image.shape[1], 1)).transpose()
    subtracted_image = (image - big_average)/big_std
    new_image = normalize_image(subtracted_image)
    return new_image
def test(row, col, radius = 5):
    x = np.arange(0, 32)
    y = np.arange(0, 32)
    arr = np.zeros((y.size, x.size))

    col = 12.
    row = 16.
    r = radius

    # The two lines below could be merged, but I stored the mask
    # for code clarity.
    mask = (x[np.newaxis, :] - col) ** 2 + (y[:, np.newaxis] - row) ** 2 < r ** 2
    print(mask)
    arr[mask] = 123

    # This plot shows that only within the circle the value is set to 123.
    plt.figure(figsize=(6, 6))
    plt.pcolormesh(x, y, arr)
    plt.colorbar()
    plt.show()
def test2():
    image = np.loadtxt('C:\\Users\\gt8mar\\capillary-flow\\tests\\vid4_centerline_array_long_7.csv', delimiter=',', dtype = int)
    # image = np.random.randint(size = (100,100), low=0, high = 255)
    print(image)
    new_image = normalize_rows(image)
    plt.imshow(image)
    plt.show()
    plt.imshow(new_image)
    plt.show()

def main(SET = 'set_01', sample = 'sample_009', write = False):
    input_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'B_stabilized')
    skeleton_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'E_centerline')
    output_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'F_blood_flow')
    shifts = pd.read_csv(os.path.join(input_folder, 'Results.csv'))
    gap_left = shifts['x'].max()
    gap_right = shifts['x'].min()
    gap_bottom = shifts['y'].min()
    gap_top = shifts['y'].max()
    # Import images
    images = get_images(os.path.join(input_folder,'vid'))
    image_array = load_image_array(images, input_folder)      # this has the shape (frames, row, col)
    # Crop array based on shifts
    image_array = image_array[:, gap_top:gap_bottom, gap_left:gap_right] 
    skeletons = load_csv_list(skeleton_folder)
    max = np.max(image_array)
    print("The size of the array is " + str(image_array.shape))

    if write:
        for i in range(len(skeletons)): 
            centerline_array = build_centerline_vs_time(image_array, skeletons[i], long=True, offset=False)
            # centerline_array = normalize_rows(centerline_array)
            centerline_array = normalize_image(centerline_array)
            np.savetxt(os.path.join(output_folder, f'{SET}_{sample}_blood_flow_{str(i).zfill(2)}.csv'), 
                    centerline_array, delimiter=',')
            im = Image.fromarray(centerline_array)
            im.save(os.path.join(output_folder, f'{SET}_{sample}_blood_flow_{str(i).zfill(2)}.tiff'))


    
    # # Plot pixels vs time:
    # plt.imshow(centerline_array)
    # plt.title('centerline pixel values per time')
    # plt.xlabel('frame')
    # plt.ylabel('centerline pixel')
    # plt.show()

    # TODO: calculate flow rate
    return 0


# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main(write=False)
    # test2()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))

