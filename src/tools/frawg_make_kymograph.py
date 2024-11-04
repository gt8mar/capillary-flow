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
from get_images import get_images
from load_image_array import load_image_array
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

def main(path, write = True, variable_radii = False, verbose = False, plot = False, hasty = False):
    """
    This function takes a path to a video and calculates the blood flow.

    Args:
        path (str): path to the video
        write (bool): whether to write the blood flow to a csv file
        variable_radii (bool): whether to use variable radii
        verbose (bool): whether to print the progress
        plot (bool): whether to plot the kymographs
        hasty (bool): whether to use the hasty segmentation files

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

    
    centerline_dict = {}
    # make dictionary of centerline files with same video number
    for centerline_file in os.listdir(os.path.join(centerline_folder, 'coords')):
        if centerline_file.endswith(".csv"):
            date = centerline_file.split('_')[0]
            video = centerline_file.split('_')[1]
            
            # check if video is in dictionary, if not add it
            if video not in centerline_dict.keys():
                centerline_dict[video] = [centerline_file]
            else:
                centerline_dict[video].append(centerline_file)
    
    # loop through videos
    for video_key in centerline_dict.keys():
        #if video_key in ['WkSlAlertFrog7Lankle2', 'WkSlExaustedFrog7Lankle2']: # TO DELETE
            #continue
        print(video_key)
        number_of_capillaries = len(centerline_dict[video_key])

        video_name = date + '_' + video_key
        video_folder = os.path.join(path, 'vids', video_name)

        # Get images
        # Import images
        start = time.time()
        images = get_images(video_folder)
        image_array = load_image_array(images, video_folder)      # this has the shape (frames, row, col)
        example_image = image_array[0]
        #print(f"Loading images for {file_prefix} {video_key} took {time.time() - start} seconds")
        print("The size of the array is " + str(image_array.shape))

        start_time = time.time()

        print('right before the capillary loop')
        # loop through capillaries
        for file in centerline_dict[video_key]:
            # participant, date, location, video_parsed, file_prefix = parse_filename(file)
            print(file)

            capillary_number = file.split('_')[4][:-4]

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
                if capillary_number.isalpha():
                    capillary_name = capillary_number
                else:
                    capillary_name = str(int(capillary_number)).zfill(2)
                # save to output folder
                print(f'saving {video_key}_kymograph_{str(capillary_number).zfill(2)}.csv')
                np.savetxt(os.path.join(output_folder, 
                                        f'{video_key}_kymograph_{capillary_name}.csv'), 
                                        kymograph, delimiter=',', fmt = '%s')
                im = Image.fromarray(kymograph)
                im.save(os.path.join(output_folder, 
                                    f'{video_key}_kymograph_{capillary_name}.tiff'))

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
    # umbrella_folder = 'J:\\frog\\data'
    umbrella_folder = '/hpc/projects/capillary-flow/frog/'
    for date in os.listdir(umbrella_folder):
        if not date.startswith('240729'):
            continue
        if date == 'archive' or date in ['240213', '240214', '240229', '240402', '240404', '240411', '240419']: # TO DELETE
            continue
        if date.endswith('alb'):
            continue
        for frog in os.listdir(os.path.join(umbrella_folder, date)):
            if frog.startswith('STD'):
                continue
            if not frog.startswith('Frog4'):
                continue   
            for side in os.listdir(os.path.join(umbrella_folder, date, frog)):
                if side.startswith('STD'):
                    continue
                if side == 'archive':
                    continue
                if not side.startswith('Left'): # only process the left side for now
                    continue
                print('Processing: ' + date + ' ' + frog + ' ' + side)
                path = os.path.join(umbrella_folder, date, frog, side)
                main(path)
    #main(path, write=True, hasty=True, verbose=False)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))

