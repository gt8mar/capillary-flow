"""
Filename: make_kymograph.py
------------------------------------------------------
This file creates kymographs (centerline vs time graphs) of each capillary.

By: Marcus Forst
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
from scipy.ndimage import gaussian_filter
from src.tools.parse_vid_path import parse_vid_path


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
def build_centerline_vs_time(image, centerline_coords, long = True):
    """
    This function takes an image and text file (default: csv) of the coordinates of a
    skeleton and outputs an image of the centerline pixel values vs time.
    :param image: 3D numpy array (time, row, col)
    :param skeleton_txt: 2D text file to be read into the function
    :return: centerline_array: 2D numpy array that shows the pixels of the centerline vs time.
    """
    centerline_array = np.zeros((centerline_coords.shape[0], image.shape[0]))
    if long == False:
        for i in range(centerline_coords.shape[0]):
            row = centerline_coords[i][0]         # skeleton coords is a list of (row, col) objects
            col = centerline_coords[i][1]
            centerline_array[i] = image[:, row, col]
    if long == True:
        for i in range(centerline_coords.shape[0]):
            row = centerline_coords[i][0]         # skeleton coords is a list of (row, col) objects
            col = centerline_coords[i][1]
            radius = 5
            centerline_array[i] = average_in_circle(image, row, col, radius)
    return centerline_array
def build_centerline_vs_time_variable_radii(image, centerline_coords, radii, long = False, offset = False):
    """
    This function takes an image and text file (default: csv) of the coordinates of a
    skeleton and outputs an image of the centerline pixel values vs time.
    :param image: 3D numpy array (time, row, col)
    :param centerline_coords: 2D array of coordinates for the centerline of the capillary
    :param radii: 1D numpy array of the radii of the capillary
    :return: centerline_array: 2D numpy array that shows the pixels of the centerline vs time.
    """
    centerline_array = np.zeros((centerline_coords.shape[0], image.shape[0]))
    if long == False:
        for i in range(centerline_coords.shape[0]):
            row = centerline_coords[i][0]         # skeleton coords is a list of (row, col) objects
            col = centerline_coords[i][1]
            centerline_array[i] = image[:, row, col]
    if long == True:
        for i in range(centerline_coords.shape[0]):
            row = centerline_coords[i][0]         # skeleton coords is a list of (row, col) objects
            col = centerline_coords[i][1]
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
def normalize_row_and_col(image):    
    # Normalize rows
    norms = np.linalg.norm(image, axis=1)
    normalized_rows = image / norms[:, np.newaxis]
    # normalized_rows = gaussian_filter(normalized_rows, sigma = 2)

    # Normalize columns
    norms = np.linalg.norm(image, axis=0)
    normalized_cols = image / norms
    # normalized_cols = gaussian_filter(normalized_cols, sigma = 2)


    # Plot original image
    plt.subplot(3, 1, 1)
    plt.imshow(image)
    plt.title("Original image")

    # Plot normalized rows
    plt.subplot(3, 1, 2)
    plt.imshow(normalized_rows)
    plt.title("Normalized rows")

    # Plot normalized columns
    plt.subplot(3, 1, 3)
    plt.imshow(normalized_cols)
    plt.title("Normalized columns")

    plt.show()
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
    # This plot shows that only within the circle the value is set to 123.
    # plt.figure(figsize=(6, 6))
    # plt.pcolormesh(x, y, arr[0])
    # plt.colorbar()
    # plt.show()
    return 0
def test2():
    image = np.loadtxt('C:\\Users\\ejerison\\capillary-flow\\tests\\set_01_sample_003_blood_flow_00.csv', delimiter=',', dtype = int)
    # image = np.random.randint(size = (100,100), low=0, high = 255)
    print(image)
    new_image = normalize_rows(image)
    plt.imshow(image)
    plt.show()
    plt.imshow(new_image)
    plt.show()
    new_new_image = normalize_row_and_col(image)
    return 0

def main(path = 'C:\\Users\\gt8mar\\capillary-flow\\data\\part11\\230427\\vid01', 
         write = True, variable_radii = False):
    """
    TODO: this
    """
    input_folder = os.path.join(path, 'moco')
    metadata_folder = os.path.join(path, 'metadata')
    centerline_folder = os.path.join(path, 'E_centerline')
    
    # Create output folders
    os.makedirs(os.path.join(path, 'F_blood_flow', 'kymo'), exist_ok=True)
    os.makedirs(os.path.join(path, 'F_blood_flow', 'velocities'), exist_ok=True)
    output_folder = os.path.join(path, 'F_blood_flow')
    
    # Get metadata
    participant, date, video = parse_vid_path(path)
    SET = 'set_01'
    file_prefix = f'{SET}_{participant}_{date}_{video}'
    gap_left, gap_right, gap_bottom, gap_top = get_shifts(metadata_folder) # get gaps from the metadata
    
    # Import images
    images = get_images(os.path.join(input_folder,'vid'))
    image_array = load_image_array(images, input_folder)      # this has the shape (frames, row, col)
    
    # Crop array based on shifts
    image_array = image_array[:, gap_top:gap_bottom, gap_left:gap_right] 
    skeleton_data = load_csv_list(os.path.join(centerline_folder, 'coords'))
    centerline_coords = [array[:, :2] for array in skeleton_data] # note that the centerline_coords will be row vectors
    centerline_radii = [array[:, 2] for array in skeleton_data] # note that the radii will be row vectors
    print("The size of the array is " + str(image_array.shape))

    if write:
        # iterate over the capillaries
        for i in range(len(skeleton_data)):
            if variable_radii: 
                kymograph = build_centerline_vs_time_variable_radii(image_array, centerline_coords[i], centerline_radii[i], long=True, offset=False)
            else:
                kymograph = build_centerline_vs_time(image_array, centerline_coords[i], long = True)
            # centerline_array = normalize_rows(centerline_array)
            kymograph = normalize_image(kymograph)
            np.savetxt(os.path.join(output_folder, 'kymo', file_prefix + f'_blood_flow_{str(i).zfill(2)}.csv'), 
                    kymograph, delimiter=',', fmt = '%s')
            im = Image.fromarray(kymograph)
            im.save(os.path.join(output_folder, 'kymo', file_prefix + f'_blood_flow_{str(i).zfill(2)}.tiff'))


    
    # # Plot pixels vs time:
    # plt.imshow(centerline_array)
    # plt.title('centerline pixel values per time')
    # plt.xlabel('frame')
    # plt.ylabel('centerline pixel')
    # plt.show()
    return 0


# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    # main(write=True)
    test2()
    # test()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))

