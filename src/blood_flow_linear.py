"""
Filename: blood_flow_linear.py
------------------------------------------------------
This file calculates the blood flow rate statistically using the centerline.

By: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
average_in_circle credit: Nicolas Gervais (https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import time
from PIL import Image

FILEFOLDER = 'C:\\Users\\gt8mar\\Desktop\\data\\221010\\vid4_moco'
SKELETON_FILE = 'vid4_test_skeleton_coords_7.csv'
# SKELETON_FILE_LIST = ['test_skeleton_coords_2.csv','test_skeleton_coords_3.csv','test_skeleton_coords_4.csv','test_skeleton_coords_5.csv','test_skeleton_coords_7.csv']
PIXELS_PER_UM = 2
FRAME_PAD = 25

# Sort images first
def tryint(s):
    try:
        return int(s)
    except:
        return s
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]
def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
def get_images(FILEFOLDER):
    """
    this function grabs image names, sorts them, and puts them in a list.
    :param FILEFOLDER: string
    :return: images: list of images
    """
    images = [img for img in os.listdir(FILEFOLDER) if img.endswith(".tif") or img.endswith(
        ".tiff")]  # if this came out of moco the file suffix is .tif otherwise it's tiff
    sort_nicely(images)
    return images
def load_image_array(image_list):
    """
    This function loads images into a numpy array.
    :param image_list: List of images
    :return: image_array: 3D numpy array
    """
    # Initialize array for images
    z_time = len(image_list)
    image_example = cv2.imread(os.path.join(FILEFOLDER, image_list[0]))
    rows, cols, layers = image_example.shape
    image_array = np.zeros((z_time, rows-FRAME_PAD-FRAME_PAD, cols-FRAME_PAD-FRAME_PAD),
                            dtype='uint16')
    # loop to populate array
    for i in range(z_time):
        image = cv2.imread(os.path.join(FILEFOLDER, image_list[i]))
        image_2D = np.mean(image, axis=2)
        image_minus_border = image_2D[FRAME_PAD:-FRAME_PAD, FRAME_PAD:-FRAME_PAD]
        image_array[i] = image_minus_border
    return image_array
def generate_operator(vector):
    """
    Use a vector to generate an operator of a series of row vectors stacked next to each other.
    This operator, when multiplied by a matrix with diagonals equal to the original vector, (an eigenvector)
    can be used to find diagonals and off-diagonals.
    :param vector: 1D numpy array
    :return: operator: 2D numpy array
    """
    operator = np.array(vector)
    A = np.array(vector)
    for n in range(len(vector) - 1):
        operator = np.vstack((operator, A))
    return np.transpose(operator)
def diagonalize(operator, vector):
    """
    This multiplies the operator and the diagonalized eigenvector to get the diagonalized matrix.
    :param operator: 2D array. This operator is a series of the same vector, vertically stacked.
    :param vector: 1D array. This is the eigenvector
    :return: diag_matrix: 2D array
    """
    eigenvector = np.diag(vector)
    return np.matmul(operator, eigenvector)
def vector_to_diag(vector):
    """
    this takes a vector and turns it into a diagonal similarity matrix.
    :param vector: 1D numpy array
    :return: diag_matrix: 2D numpy array
    """
    vector = np.array(vector)
    operator = generate_operator(vector)
    return diagonalize(operator, vector)
def cycle_rows(array):
    """
    Cycle arrays by taking the top row and putting it on the bottom.
    :param array: 2D numpy array
    :return: cycled: 2D array
    """
    return np.vstack((array[1:], array[0]))
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
def build_centerline_vs_time(image, skeleton_txt, delimiter = ',', long = False, offset = False):
    """
    This function takes an image and text file (default: csv) of the coordinates of a
    skeleton and outputs an image of the centerline pixel values vs time.
    :param image: 3D numpy array (time, row, col)
    :param skeleton_txt: 2D text file to be read into the function
    :return: centerline_array: 2D numpy array that shows the pixels of the centerline vs time.
    """
    skeleton_coords = np.genfromtxt(skeleton_txt, delimiter=delimiter, dtype=int)
    if offset:
        row_offset = np.full((skeleton_coords.shape[0], 1), 0)
        col_offset = np.full((skeleton_coords.shape[0], 1), -100)
        offset_array = np.hstack((row_offset, col_offset))
        skeleton_coords = skeleton_coords + offset_array
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
    return image
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
def offset_skeleton(skeleton_txt, OFFSET):
    pass

def main():
    # test(16, 12, 5)
    # Import images
    images = get_images(FILEFOLDER)
    image_array = load_image_array(images)      # this has the shape (frames, row, col)
    # write background
    background = np.mean(image_array, axis=0)
    max = np.max(image_array)
    print("The size of the array is " + str(image_array.shape))

    # save one file
    centerline_array = build_centerline_vs_time(image_array, SKELETON_FILE, long=True, offset=False)
    np.savetxt("vid4_centerline_array_long_7.csv", centerline_array, delimiter=',')   # str(SKELETON_FILE) + for loop

    # for file in SKELETON_FILE_LIST:
    #     print(str(file))
    #     centerline_array = build_centerline_vs_time(image_array, file, long= True)
    #     np.savetxt(str(file) + "centerline_array_long.csv", centerline_array, delimiter=',')
    #     normalized_array = normalize_image(centerline_array)
    #     im = Image.fromarray(normalized_array)
    #     im.save(str(file) + "centerline_array_long.tiff")

    # Plot pixels vs time:
    plt.imshow(centerline_array)
    plt.title('centerline pixel values per time')
    plt.xlabel('frame')
    plt.ylabel('centerline pixel')
    plt.show()

    # TODO: calculate flow rate
    return 0



# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))

