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
from cv2 import filter2D


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
    :return: kymograph: 2D numpy array that shows the pixels of the centerline vs time.
    """
    kymograph = np.zeros((centerline_coords.shape[0], image.shape[0]))
    if long == False:
        for i in range(centerline_coords.shape[0]):
            row = centerline_coords[i][0]         # skeleton coords is a list of (row, col) objects
            col = centerline_coords[i][1]
            kymograph[i] = image[:, row, col]
    if long == True:
        for i in range(centerline_coords.shape[0]):
            row = centerline_coords[i][0]         # skeleton coords is a list of (row, col) objects
            col = centerline_coords[i][1]
            radius = 5
            kymograph[i] = average_in_circle(image, row, col, radius)
    return kymograph
def build_centerline_vs_time_kernal(image, centerline_coords, long = True):
    """
    This function takes an image and text file (default: csv) of the coordinates of a
    skeleton and outputs an image of the centerline pixel values vs time.
    :param image: 3D numpy array (time, row, col)
    :param skeleton_txt: 2D text file to be read into the function
    :return: centerline_array: 2D numpy array that shows the pixels of the centerline vs time.
    """
    averaged_array = compute_average_surrounding_pixels(image)
    kymograph = np.zeros((centerline_coords.shape[0], image.shape[0]))
    if long == False:
        for i in range(centerline_coords.shape[0]):
            row = centerline_coords[i][0]         # skeleton coords is a list of (row, col) objects
            col = centerline_coords[i][1]
            kymograph[i] = image[:, row, col]
    if long == True:
        for i in range(centerline_coords.shape[0]):
            row = centerline_coords[i][0]         # skeleton coords is a list of (row, col) objects
            col = centerline_coords[i][1]
            radius = 5
            kymograph[i] = averaged_array[:, row, col]
    return kymograph
def build_centerline_vs_time_variable_radii(image, skeleton_data, long = False, offset = False):
    """
    This function takes an image and text file (default: csv) of the coordinates of a
    skeleton and outputs an image of the centerline pixel values vs time.
    :param image: 3D numpy array (time, row, col)
    :param centerline_coords: 2D array of coordinates for the centerline of the capillary
    :param radii: 1D numpy array of the radii of the capillary
    :return: centerline_array: 2D numpy array that shows the pixels of the centerline vs time.
    """
    centerline_array = np.zeros((skeleton_data.shape[0], image.shape[0]))
    if long == False:
        for i in range(skeleton_data.shape[0]):
            row = skeleton_data[i][0]         # skeleton coords is a list of (row, col) objects
            col = skeleton_data[i][1]
            centerline_array[i] = image[:, row, col]
    if long == True:
        for i in range(skeleton_data.shape[0]):
            row = skeleton_data[i][0]         # skeleton coords is a list of (row, col) objects
            col = skeleton_data[i][1]
            radius = int(skeleton_data[i][2])
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
def test2_normalize_row_and_col():
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
def compute_average_surrounding_pixels(image_stack):
    # Convert the image stack to float32 type for accurate calculations
    image_stack = np.float32(image_stack)

    # Create a kernel of ones with a size of 3x3
    kernel = np.ones((3, 3), np.float32) / 9

    # Reshape the image stack to a 4D tensor to represent individual windows
    image_windows = image_stack.reshape(image_stack.shape[0], image_stack.shape[1], -1, 3)

    # Perform 2D convolution to compute the average of the surrounding pixels
    averaged_windows = filter2D(image_windows, -1, kernel)

    # Reshape the averaged windows back to the original stack shape
    averaged_stack = averaged_windows.reshape(image_stack.shape)

    # Convert the averaged stack back to the original data type (e.g., uint8)
    averaged_stack = np.uint8(averaged_stack)

    return averaged_stack



def main(path = 'C:\\Users\\gt8mar\\capillary-flow\\data\\part11\\230427\\vid01', 
         write = True, variable_radii = False, verbose = False):
    """
    This function takes a path to a video and calculates the blood flow.

    Args:
        path (str): path to the video
        write (bool): whether to write the blood flow to a csv file
        variable_radii (bool): whether to use variable radii
        verbose (bool): whether to print the progress

    Returns:
        blood_flow (np.array): blood flow

    Saves:
        kymograph (np.array): kymograph of the blood flow
        kymograph (png file): kymograph of the blood flow
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
    print(gap_left, gap_right, gap_bottom, gap_top)

    # Import images
    start = time.time()
    images = get_images(input_folder)
    image_array = load_image_array(images, input_folder)      # this has the shape (frames, row, col)
    example_image = image_array[0]
    print("The time to load the images is " + str(time.time() - start) + " seconds")
    print("The size of the array is " + str(image_array.shape))

    # Crop array based on shifts
    image_array = image_array[:, gap_top:example_image.shape[0] + gap_bottom, gap_left:example_image.shape[1] + gap_right] 
    start_time = time.time()
    skeleton_data = load_csv_list(os.path.join(centerline_folder, 'coords'))
    print(f"the time to load the skeleton csv is {time.time() - start_time} seconds")
    print("The size of the array after trimming is " + str(image_array.shape))
    # iterate over the capillaries
    for i in range(len(skeleton_data)):
        if variable_radii: 
            kymograph = build_centerline_vs_time_variable_radii(image_array, 
                            skeleton_data[i], long=True, offset=False)
        else:
            start_time = time.time()
            kymograph = build_centerline_vs_time(image_array, skeleton_data[i], long = True)
            print(f"the old way took {time.time() - start_time} seconds")
            start_time = time.time()
            kymograph_new = build_centerline_vs_time_kernal(image_array, skeleton_data[i], long = True)
            print(f"the new way took {time.time() - start_time} seconds")
        # centerline_array = normalize_rows(centerline_array)
        start_time = time.time()
        kymograph = normalize_image(kymograph)
        print(f"the time to normalize the image is {time.time() - start_time} seconds")
        kymograph_new = normalize_image(kymograph_new)
        if write:
                np.savetxt(os.path.join(output_folder, 'kymo', 
                                        file_prefix + f'_blood_flow_{str(i).zfill(2)}.csv'), 
                                        kymograph, delimiter=',', fmt = '%s')
                im = Image.fromarray(kymograph)
                im.save(os.path.join(output_folder, 'kymo', 
                                    file_prefix + f'_blood_flow_{str(i).zfill(2)}.tiff'))
                im2 = Image.fromarray(kymograph_new)
                im2.save(os.path.join(output_folder, 'kymo', 
                                    file_prefix + f'_blood_flow_{str(i).zfill(2)}_new.tiff'))

        if verbose:
            # Plot pixels vs time:
            plt.imshow(kymograph)
            plt.title('centerline pixel values per time')
            plt.xlabel('frame')
            plt.ylabel('centerline pixel')
            plt.show()
    return 0


# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main(path ='/hpc/projects/capillary-flow/data/part11/230427/vid01',
          write=True)
    # test2_normalize_row_and_col()
    # test()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))

