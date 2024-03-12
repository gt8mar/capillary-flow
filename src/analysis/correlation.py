"""
Filename: correlation.py
------------------------------------------------------
Calculates the correlation between pixels and their nearest neighbors. 

By: Marcus Forst
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import time
from PIL import Image
import pandas as pd
from skimage.measure import block_reduce
from src.tools.get_images import get_images
from src.tools.load_image_array import load_image_array
from src.tools.get_shifts import get_shifts
from src.analysis.correlation_edge import find_edge_points

BIN_FACTOR = 4

# SECTION_START = 138
# SECTION_END = 984

def make_correlation_matrix(image_array_binned):
    """
    Calculates the correlation between pixels and their nearest neighbors.
    """
    # Initialize correlation matrix
    up_left = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, :-2, :-2]
    up_mid = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, :-2, 1:-1]
    up_right = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, :-2, 2:]
    center_right = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 1:-1, 2:]
    center_left = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 1:-1, :-2]
    down_right = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 2:, 2:]
    down_left = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 2:, :-2]
    down_mid = image_array_binned[:-1, 1:-1, 1:-1] * image_array_binned[1:, 2:, 1:-1]

    """-----------------------------------------------------------------------------------------------------------------"""

    # total correlations
    corr_total_right = 707 * (up_right + down_right) // 1000 + center_right
    corr_total_left = 707 * (up_left + down_left) // 1000 + center_left
    corr_total_up = 707 * (up_left + up_right) // 1000 + up_mid
    corr_total_down = 707 * (down_left + down_right) // 1000 + down_mid

    # Now we need to make vectors for the correllations:
    corr_total_right_2D = np.mean(corr_total_right, axis=0)
    corr_total_left_2D = np.mean(corr_total_left, axis=0)
    corr_total_up_2D = np.mean(corr_total_up, axis=0)
    corr_total_down_2D = np.mean(corr_total_down, axis=0)
    corr_x = corr_total_right_2D - corr_total_left_2D
    # corr_x /= 5000
    corr_y = corr_total_up_2D - corr_total_down_2D
    # corr_y /= 5000
    return corr_x, corr_y

def main(path, mask = False, verbose = False, plot = True, write = False, bin_bool = False, bin_factor = 4):
    """
    Calculates the correlation between pixels and their nearest neighbors.

    Args:   
        path (str): The path to the location folder for the videos
    """
    participant = 'part26'
    video_key = 'vid09'
    test_mask = 'set01_part26_230613_loc01_vid09_seg_cap_01a.png'
    test_centerline = 'set01_part26_230613_loc01_vid09_centerline_01a.csv'

    if os.path.exists(os.path.join(path, 'vids', video_key, 'mocoslice')):
        video_folder = os.path.join(path, 'vids', video_key, 'mocoslice')
    elif os.path.exists(os.path.join(path, 'vids', video_key, 'mocosplit')):
        video_folder = os.path.join(path, 'vids', video_key, 'mocosplit')
    else:
        video_folder = os.path.join(path, 'vids', video_key, 'moco')

    # check to see if moco folder exists
    if os.path.exists(video_folder) == False:
        print(f'No moco folder for {path} and {video_folder}') 
    metadata_folder = os.path.join(path, 'vids', video_key, 'metadata')

    mask_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\results\\segmented\\individual_caps_original')
    centerline_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\results\\centerlines')
    output_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\results\\correlation')
    os.makedirs(output_folder, exist_ok=True)


    images = get_images(video_folder)
    image_array = load_image_array(images, video_folder)
    example_image = image_array[0]


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

    # Crop array based on shifts
    image_array = image_array[:, gap_top:example_image.shape[0] + gap_bottom, gap_left:example_image.shape[1] + gap_right] 

    # Read in the masks
    segmented = cv2.imread(os.path.join(mask_folder, test_mask), cv2.IMREAD_GRAYSCALE)
    # Make mask either 1 or 0
    segmented[segmented != 0] = 1
    if mask:
        image_array = segmented * image_array
    print(f'image_array.shape: {image_array.shape}')

    if bin_bool:
        # """Bin images to conserve memory and improve resolution"""
        image_array_binned = block_reduce(image_array, (2,2,2), func= np.mean)
        mask_binned = block_reduce(segmented, (2, 2), func=np.mean)

        print(image_array_binned.shape)
        print(mask_binned.shape)
        # plt.imshow(np.mean(image_array_binned, axis = 0))
        # plt.show()

        corr_x, corr_y = make_correlation_matrix(image_array_binned)
        # Plot a subset of this array
        corr_x_slice = corr_x[::bin_factor, ::bin_factor]
        corr_y_slice = corr_y[::bin_factor, ::bin_factor]
        # mask_slice = mask_binned[::bin_factor, ::bin_factor] 
        # mask_slice = mask_slice[:,:-1]  # this -1 is a result of the make_corr_matrix shift
        mask_slice = 1
        print(f'corr_y_slice.shape: {corr_y_slice.shape}')

        # y, x = np.meshgrid(np.arange(0, rows // (BIN_FACTOR*2), 1), np.arange(0, cols // (BIN_FACTOR*2), 1))
        # plt.quiver(x, y, corr_x_slice, corr_y_slice)
    
    else:
        corr_x, corr_y = make_correlation_matrix(image_array)
        print(f'corr_x shape: {corr_x.shape}')
        # Plot a subset of this array
        corr_x_slice = corr_x[::bin_factor, ::bin_factor]
        corr_y_slice = corr_y[::bin_factor, ::bin_factor]
        mask_slice = 1
        # print(corr_y_slice.shape)


    plt.quiver(corr_y_slice*mask_slice, corr_x_slice*mask_slice, angles = 'xy')
    plt.gca().invert_yaxis()
    if write:
        plt.imsave(os.path.join(output_folder, "correlation.png"))
    if plot:
        plt.show()
    else:
        plt.close()
    

    # now check centerline and edge points:
    # load in centerline
    centerline = np.loadtxt(os.path.join(centerline_folder, test_centerline), delimiter=",")
    # remove 3rd column
    centerline = centerline[:, :2]

    # find edge points
    edge_points = find_edge_points(segmented, centerline)

    # plot edge points and centerline points with correlation
    print(f'corr_y_slice.shape: {corr_y_slice.shape}')

    edge_points_corr = pd.DataFrame(edge_points, columns = ['Centerline', 'Left', 'Right'])
    # Identify the corr_x and corr_y values at the centerline point and the edge points
    for point in edge_points:
        centerline_point = point[0]
        edge_left = point[1]
        edge_right = point[2]

        # find the corr_x and corr_y values at the centerline point
        corr_x_centerline = corr_x[int(centerline_point[0]), int(centerline_point[1])]
        corr_y_centerline = corr_y[int(centerline_point[0]), int(centerline_point[1])]
        magnitude_centerline = np.sqrt(corr_x_centerline**2 + corr_y_centerline**2)
        # find the corr_x and corr_y values at the edge points
        corr_x_left = corr_x[int(edge_left[0]), int(edge_left[1])]
        corr_y_left = corr_y[int(edge_left[0]), int(edge_left[1])]
        magnitude_left = np.sqrt(corr_x_left**2 + corr_y_left**2)

        corr_x_right = corr_x[int(edge_right[0]), int(edge_right[1])]
        corr_y_right = corr_y[int(edge_right[0]), int(edge_right[1])]
        magnitude_right = np.sqrt(corr_x_right**2 + corr_y_right**2)

        edge_points_corr.loc[edge_points_corr['Centerline'] == centerline_point, 'Centerline_corr_x'] = corr_x_centerline
        edge_points_corr.loc[edge_points_corr['Centerline'] == centerline_point, 'Centerline_corr_y'] = corr_y_centerline
        edge_points_corr.loc[edge_points_corr['Centerline'] == centerline_point, 'Centerline_magnitude'] = magnitude_centerline

        edge_points_corr.loc[edge_points_corr['Left'] == edge_left, 'Left_corr_x'] = corr_x_left
        edge_points_corr.loc[edge_points_corr['Left'] == edge_left, 'Left_corr_y'] = corr_y_left
        edge_points_corr.loc[edge_points_corr['Left'] == edge_left, 'Left_magnitude'] = magnitude_left

        edge_points_corr.loc[edge_points_corr['Right'] == edge_right, 'Right_corr_x'] = corr_x_right
        edge_points_corr.loc[edge_points_corr['Right'] == edge_right, 'Right_corr_y'] = corr_y_right
        edge_points_corr.loc[edge_points_corr['Right'] == edge_right, 'Right_magnitude'] = magnitude_right
    
    # plot histogram of the magnitude of the correlation for the centerline and edge points
    plt.hist(edge_points_corr['Centerline_magnitude'], bins = 20, alpha = 0.75, label = 'Centerline')
    plt.hist(edge_points_corr['Left_magnitude'], bins = 20, alpha = 0.35, label = 'Left')
    plt.hist(edge_points_corr['Right_magnitude'], bins = 20, alpha = 0.35, label = 'Right')
    plt.legend()
    plt.show()




    # print(edge_points)

    for point in edge_points:
        # plot using matplotlib
        plt.plot(point[0][1], point[0][0], 'ro')
        plt.plot(point[1][1], point[1][0], 'go')
        plt.plot(point[2][1], point[2][0], 'bo')
    plt.show()


    return 0

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    test_path = 'F:\\Marcus\\data\\part26\\230613\\loc01'
    main(path= test_path, mask = False, plot = False, bin_bool=False)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))