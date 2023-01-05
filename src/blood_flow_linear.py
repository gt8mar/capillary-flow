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
import pandas as pd
from PIL import Image
from src.tools.get_images import get_images
from src.tools.load_image_array import load_image_array
from src.tools.load_csv_list import load_csv_list
from src.tools.get_shifts import get_shifts

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
def build_centerline_vs_time(image, skeleton_coords, radii, long = False, offset = False):
    """
    This function takes an image and text file (default: csv) of the coordinates of a
    skeleton and outputs an image of the centerline pixel values vs time.
    :param image: 3D numpy array (time, row, col)
    :param skeleton_txt: 2D text file to be read into the function
    :return: centerline_array: 2D numpy array that shows the pixels of the centerline vs time.
    """
    centerline_array = np.zeros((skeleton_coords.shape[0], image.shape[0]))
    if long == False:
        for i in range(skeleton_coords.shape[0]):
            row = skeleton_coords[i][0]         # skeleton coords is a list of (row, col) objects
            col = skeleton_coords[i][1]
            centerline_array[i] = image[:, row, col]
    if long == True:
        for i in range(skeleton_coords.shape[0]):
            row = skeleton_coords[i][0]         # skeleton coords is a list of (row, col) objects
            col = skeleton_coords[i][1]
            radius = int(radii[i])
            centerline_array[i] = average_in_circle(image, row, col, radius)
    return centerline_array
def average_in_circle(image, row, col, radius = 5):
    """
    This function inputs an image and a coordinate and outputs the average of a circle of
    pixels surrounding the coordinate with specified radius.
    :param image: 3D numpy array
    :param row: integer, the row coordinate you want to average around
    :param col: integer, the column coordinate you want to average around
    :param radius: the radius you want to average over
    :return circle_values_list: a numpy array of the averaged values of a (row, col) time-slice.
    """
    x = np.arange(0, image.shape[2])
    y = np.arange(0, image.shape[1])

    mask = (x[np.newaxis, :] - col) ** 2 + (y[:, np.newaxis] - row) ** 2 < radius ** 2
    mask = np.dstack([mask]*image.shape[0])
    mask = mask.transpose(2, 0, 1)
    circle_values_list = image[mask].reshape(image.shape[0],-1)
    circle_values_list = circle_values_list.mean(axis = 1)
    return circle_values_list
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
def test(row = 16, col = 12, radius = 5):
    x = np.arange(0, 32)
    y = np.arange(0, 32)
    z = np.arange(0, 40)
    arr = np.zeros((z.size, y.size, x.size))

    col = 12.
    row = 16.
    r = radius

    # The two lines below could be merged, but I stored the mask
    # for code clarity.
    mask = (x[np.newaxis, :] - col) ** 2 + (y[:, np.newaxis] - row) ** 2 < r ** 2
    mask_array = np.dstack([mask]*arr.shape[0])
    mask_array = mask_array.transpose(2, 0, 1)
    arr[mask_array] = 123
    print(np.sum(mask))
    print(mask.shape)
    print(mask_array.shape)
    print(arr[mask_array].shape)
    new = arr[mask_array].reshape(arr.shape[0],-1)
    print(new.shape)
    print(new[0])
    print(new.mean(axis=1))
    # # This plot shows that only within the circle the value is set to 123.
    # plt.figure(figsize=(6, 6))
    # plt.pcolormesh(x, y, arr[0])
    # plt.colorbar()
    # plt.show()
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
    gap_left, gap_right, gap_bottom, gap_top = get_shifts(input_folder)
    # Import images
    images = get_images(os.path.join(input_folder,'vid'))
    image_array = load_image_array(images, input_folder)      # this has the shape (frames, row, col)
    # Crop array based on shifts
    image_array = image_array[:, gap_top:gap_bottom, gap_left:gap_right] 
    skeletons = load_csv_list(skeleton_folder)
    path = str(os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'D_segmented'))
    centerline_radii = load_csv_list(path, float)
    centerline_radii = centerline_radii[0]          # this has the same length as  
    print("The size of the array is " + str(image_array.shape))

    if write:
        for j in range(1): #len(skeletons)
            i = 14
            centerline_array = build_centerline_vs_time(image_array, skeletons[i], centerline_radii, long=True, offset=False)
            # centerline_array = normalize_rows(centerline_array)
            centerline_array = normalize_image(centerline_array)
            np.savetxt(os.path.join(output_folder, f'{SET}_{sample}_blood_flow_{str(i).zfill(2)}_v2.csv'), 
                    centerline_array, delimiter=',')
            im = Image.fromarray(centerline_array)
            im.save(os.path.join(output_folder, f'{SET}_{sample}_blood_flow_{str(i).zfill(2)}_v2.tiff'))


    
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
    main(write=True)
    # test2()
    # test()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))

