"""
Filename: make_kymograph.py
------------------------------------------------------
This file creates kymographs (centerline vs time graphs) of each capillary.

By: Marcus Forst
average_in_circle credit: Nicolas Gervais (https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle)
"""

import os, time, gc, platform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from src.tools.get_images import get_images
from src.tools.load_image_array import load_image_array
from src.tools.load_name_map import load_name_map
from src.tools.load_csv_list import load_csv_list
from src.tools.get_shifts import get_shifts
from src.tools.parse_filename import parse_filename
from scipy.ndimage import convolve
from skimage import exposure

PIXELS_PER_UM = 2

def create_circular_kernel(radius):
    """
    Create a circular kernel of a given radius.
    
    Args:
        radius (int): radius of the circular kernel
    Returns:
        kernel (np.ndarray): circular kernel of size (2*radius+1, 2*radius+1)
    """
    diameter = 2 * radius + 1
    center = (radius, radius)
    kernel = np.zeros((diameter, diameter), dtype=np.float32)

    for i in range(diameter):
        for j in range(diameter):
            if np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2) <= radius:
                kernel[i, j] = 1

    return kernel / np.sum(kernel)
def compute_average_surrounding_pixels(image_stack, radius=4, circle = True):
    """
    Compute the average of the surrounding pixels for each pixel in the image stack.

    Args:
        image_stack (np.ndarray): 3D image stack of shape (time, row, col)
        radius (int): radius of the circular kernel

    Returns:
        averaged_stack (np.ndarray): 3D image stack of shape (time, row, col)
    """
    # Convert the image stack to float32 type for accurate calculations
    image_stack = np.float32(image_stack)

    if circle:
        # Create a circular kernel of a given radius
        kernel = create_circular_kernel(radius)
        
    else:
        # Create a kernel of ones with a size of radius x radius
        kernel = np.ones((radius, radius), np.float32) / radius**2

    # Perform 3D convolution to compute the average of the surrounding pixels
    averaged_stack = convolve(image_stack, kernel[np.newaxis, :, :])

    # Convert the averaged stack back to the original data type (e.g., uint8)
    averaged_stack = np.uint8(averaged_stack)

    return averaged_stack
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
def row_wise_normalize(image):
    """" 
    Normalizes the rows of an image by dividing each row by the average of that row

    Args:
        image (np.ndarray): 2D image of shape (row, col)
    
    Returns:
        image (np.ndarray): 2D image of shape (row, col)
    """
    # Compute the average intensity of each row
    row_averages = np.mean(image, axis=1)

    # Calculate the mean average intensity across all rows
    mean_average = np.mean(row_averages)

    # Compute the scaling factors for each row
    scaling_factors = mean_average / row_averages

    # Apply row-wise normalization
    normalized_image = image * scaling_factors[:, np.newaxis]

    # Convert the normalized image to 8-bit unsigned integer
    normalized_image = normalized_image.astype(np.uint8)
    
    return image
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

def main(path = 'F:\\Marcus\\data\\part09\\230414\\loc01', 
         write = True, variable_radii = False, verbose = False, plot = False, test = False):
    """
    This function takes a path to a video and calculates the blood flow.

    Args:
        path (str): path to the video
        write (bool): whether to write the blood flow to a csv file
        variable_radii (bool): whether to use variable radii
        verbose (bool): whether to print the progress
        plot (bool): whether to plot the kymographs

    Returns:
        blood_flow (np.array): blood flow

    Saves:
        kymograph (np.array): kymograph of the blood flow
        kymograph (png file): kymograph of the blood flow
    """
    
    # Create output folders
    centerline_folder = os.path.join(path, 'centerlines')

    os.makedirs(os.path.join(path, 'kymographs'), exist_ok=True)
    output_folder = os.path.join(path, 'kymographs')
    
    if platform.system() == 'Windows':
        results_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results'
    else:
        results_folder = '/hpc/projects/capillary-flow/results'
    
    centerline_dict = {}
    # make dictionary of centerline files with same video number
    for centerline_file in os.listdir(os.path.join(centerline_folder, 'coords')):
        if centerline_file.endswith(".csv"):
            participant, date, location, video, file_prefix = parse_filename(centerline_file)
            if video.endswith('bp'):
                video = video.replace('bp', '')
            # check if video ends with "scan"
            elif video.endswith('scan'):
                continue # skip scan video for kymographs, the frame is moving
            
            # check if video is in dictionary, if not add it
            if video not in centerline_dict.keys():
                centerline_dict[video] = [centerline_file]
            else:
                centerline_dict[video].append(centerline_file)
        else:
            continue
    if verbose:
        # check dictionary to see if videos match up with the key
        for video in centerline_dict.keys():
            print(video)
            for test_file in centerline_dict[video]:
                __, __, __, video_parsed, __ = parse_filename(test_file)
                if video_parsed.endswith('bp'):
                    video_parsed = video_parsed.replace('bp', '')
                if video != video_parsed:
                    print(f'Video name mismatch: {video} vs {video_parsed}')
   
    # loop through videos
    for video_key in centerline_dict.keys():
        number_of_capillaries = len(centerline_dict[video_key])
        if video_key.endswith('bp'):
            video_key = video_key.replace('bp', '')
        
        # check to see if mocoslice or mocosplit folder exists
        if os.path.exists(os.path.join(path, 'vids', video_key, 'mocoslice')):
            video_folder = os.path.join(path, 'vids', video_key, 'mocoslice')
        elif os.path.exists(os.path.join(path, 'vids', video_key, 'mocosplit')):
            video_folder = os.path.join(path, 'vids', video_key, 'mocosplit')
        else:
            video_folder = os.path.join(path, 'vids', video_key, 'moco')

        # check to see if moco folder exists
        if os.path.exists(video_folder) == False:
            print(f'No moco folder for {file_prefix} and {video_folder}') 
        metadata_folder = os.path.join(path, 'vids', video_key, 'metadata')

        # Get metadata
        gap_left, gap_right, gap_bottom, gap_top = get_shifts(metadata_folder) # get gaps from the metadata

        # Check to make sure that the shifts are not negative
        if gap_left < 0:
            gap_left = 0
        if gap_top < 0:
            gap_top = 0
        if gap_right > 0:
            gap_right = 0
        if gap_bottom > 0:
            gap_bottom = 0

        if verbose:
            print(gap_left, gap_right, gap_bottom, gap_top)

        # Import images
        start = time.time()
        images = get_images(video_folder)
        image_array = load_image_array(images, video_folder)      # this has the shape (frames, row, col)
        example_image = image_array[0]
        print(f"Loading images for {file_prefix} {video_key} took {time.time() - start} seconds")
        print("The size of the array is " + str(image_array.shape))

        # Crop array based on shifts
        image_array = image_array[:, gap_top:example_image.shape[0] + gap_bottom, gap_left:example_image.shape[1] + gap_right] 
        start_time = time.time()

        print('right before the capillary loop')
        # loop through capillaries
        for file in centerline_dict[video_key]:
            participant, date, location, video_parsed, file_prefix = parse_filename(file)
            capillary_number = file.split('_')[-1].split('.')[0]
            kymograph_filename = file.replace('centerline', 'kymograph').replace('csv', 'tiff')
           
            print(f'Processing {video_key} capillary {capillary_number}')            

            # load centerline file:
            skeleton = np.loadtxt(os.path.join(centerline_folder, 'coords', file), delimiter=',').astype(int)

            # build the kymograph
            start_time = time.time()
            kymograph = build_centerline_vs_time_kernal(image_array, skeleton, long = True)
            print(f"capillary {capillary_number} took {time.time() - start_time} seconds")
            
            # normalize the kymograph 
            start_time = time.time()
            # normalize intensity of the kymograph
            kymograph = exposure.rescale_intensity(kymograph, in_range = 'image', out_range = np.uint8)
            if verbose:
                print(f"the time to normalize the image is {time.time() - start_time} seconds")

            # save the kymograph
            if write:
                if video_parsed == video_key:
                    # save to output folder
                    print(f'saving {kymograph_filename}')
                    im = Image.fromarray(kymograph)
                    im.save(os.path.join(output_folder, 
                                        kymograph_filename))
                    if not test:
                        # save to results folder
                        im.save(os.path.join(results_folder, 'kymographs',
                                        kymograph_filename))
                else:
                    print(f'video {video_parsed} does not match video key {video_key}')
            if plot:
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
    if platform.system() == 'Windows':
        # path = 'F:\\Marcus\\data\\part09\\230414\\loc01'
        # path = 'C:\\Users\\gt8mar\\capillary-flow\\tests\\part22\\230530\\loc02'
        path = 'F:\\Marcus\\data\\2022\\test_220513\\part00\\220513\\loc01'
        main(path, write=True, verbose=False)
    else:
        path = '/hpc/projects/capillary-flow/data/part09/230414/loc01'
        main(path, write = True)
    # test2_normalize_row_and_col()
    # test()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))

