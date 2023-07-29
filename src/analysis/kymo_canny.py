"""
Filename: kymo_canny.py
-------------------------------------------------
This file uses canny edge detection to call average velocities from
kymographs. 

By: Marcus Forst
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import os
import seaborn as sns
import time
from src.tools.get_images import get_images
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from sklearn.linear_model import Lasso

FPS = 169.3
PIX_UM = 1.74
CANNY_THRESH_1 = 20
CANNY_THRESH_2 = 50

# TODO: play around with these parameters

# TODO: improve plotting
# TODO: zero speed capillaries handling
# TODO: metadata lookup
# TODO: data aggregation

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
def select_outliers(data, lower_percentile = 10, upper_percentile = 90):
    """
    This function removes outliers by percentile and returns
    both the clipped data and the outlier points. 
    Args:
        data (list of list): List of lists with data to be plotted.
        lower_percentile (int): lower percentile to clip
        upper_percentile (int): upper percentile to clip
    """
    data_clipped = []
    outlier_points = []
    for column in data:
        q1 = np.percentile(column, lower_percentile)
        q3 = np.percentile(column, upper_percentile)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data_clipped.append(column[(column >= lower_bound) & (column <= upper_bound)]) 
        outlier_points.append(column[(column < lower_bound) | (column > upper_bound)])
    return data_clipped, outlier_points
def plot_box_swarm(data, x_labels, y_axis_label,  plot_title, figure_name, 
                   verbose = True, write = False, remove_outliers = False):
    """Plot box-plot and swarm plot for data list.
 
    Args:
        data (list of list): List of lists with data to be plotted.
        y_axis_label (str): Y- axis label.
        x_labels (list of str): List with labels of x-axis.
        plot_title (str): Plot title.
        figure_name (str): Path to output figure.
        verbose (bool): If True, show plot.
        write (bool): If True, write plot to file.
        remove_outliers (bool): If True, remove outliers from data.

    Returns:
        int: 0 if successful.
         
    """
    if remove_outliers:
        # Remove outliers using the clip function
        data_clipped, outliers = select_outliers(data)
        # initialize plot information:
        sns.set(color_codes=True)
        plt.figure(1, figsize=(9, 6))
        plt.title(plot_title)

        # Create box and swarm plot with clipped data
        ax = sns.boxplot(data=data_clipped)
        sns.swarmplot(data=data_clipped, color=".25")
        sns.swarmplot(data = outliers, color = "red")
        if verbose: plt.show()
    else:
        # initialize plot information:
        sns.set(color_codes=True)
        plt.figure(1, figsize=(9, 6))
        plt.title(plot_title)
    
        # plot data on swarmplot and boxplot
        ax = sns.boxplot(data=data)
        sns.swarmplot(data=data, color=".25")
        
        # y-axis label
        ax.set(ylabel=y_axis_label)
    
        # write labels with number of elements
        ax.set_xticks(np.arange(len(data)), labels = x_labels)
        ax.legend()
        
        if write:
            # write figure file with quality 400 dpi
            plt.savefig(figure_name, bbox_inches='tight', dpi=400)
            if verbose: plt.show()
            else: plt.close()
        if verbose: plt.show()
    return 0
def find_slopes(image, output_folder=None, method = 'ridge', verbose = False, write = False, plot_title = "Kymograph", filename = "kymograph_1", remove_outliers = False):
    edges = cv2.Canny(image, CANNY_THRESH_1, CANNY_THRESH_2)
    # To save edge images:
    # Create a 1x3 grid for subplots
    fig = plt.figure(figsize= (16,4)) #figsize = (5,20)
    # gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 2])
    # gs.update(height_ratios=[1])


    # # Add plots to the grid
    # ax1 = plt.subplot(gs[0])
    # ax2 = plt.subplot(gs[1])
    # ax3 = plt.subplot(gs[2]) 
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=1, rowspan=2)
    ax2 = plt.subplot2grid((2,4), (0, 1), colspan=1, rowspan=2)
    ax3 = plt.subplot2grid((1,4), (0, 2), colspan=2, rowspan=1)
    ax1.imshow(edges)
    ax1.set_title("Canny Edge Detection")
        
    print(f"the shape of the file is {edges.shape}")
    # Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # remove contours with less than 10 points
    contours = [contour for contour in contours if len(contour) > 20]

    # Iterate through the contours
    slopes = []
    lengths = []
    for contour in contours:
        if method == 'ridge':
            # Fit a line to the contour using least squares regression
            [vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Compute the start and end points of the line
            lefty = int((-x*vy/vx) + y)
            righty = int(((image.shape[1]-x)*vy/vx)+y)
            print(f"lefty is {lefty}")
            print(f"righty is {righty}")
            # Draw the line on the original image
            # cv2.line(image, (image.shape[1]-1,righty), (0,lefty), (0,255,0), 2)
        if method == 'lasso':
            # Extract the x and y coordinates of the contour points
            x, y = contour[:, 0, 0], contour[:, 0, 1]
            
            # Fit a Lasso regression model to the contour
            lasso = Lasso(alpha=0.1, tol = 0.0001)
            X = x.reshape(-1, 1) # Reshape the x array into a 2D array
            lasso.fit(X, y)
            
            # Compute the start and end points of the line
            start_x, end_x = 0, image.shape[1]-1
            start_y, end_y = lasso.predict([[start_x]]), lasso.predict([[end_x]])
            slope = (end_y-start_y)/(end_x-start_x)
            if np.absolute(slope) <= 1:
                # cv2.line(image, (int((end_x-start_x)//4), int((end_y-start_y)//4)), (int(3*(end_x-start_x)//4), int(3*(end_y-start_y)//4)), (0,255,0), 2)
                pass
            else:
                # Draw the line on the original image
                # cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0,255,0), 2)
                pass
            # weight the slope by the length of the contour
            length = len(contour)
            lengths.append(length)

            # Add line to list of lines
            slopes.append(slope[0])

    # Plot the image with the lines
    if remove_outliers:
        slopes_clipped, outliers = select_outliers(slopes)
        if len(slopes_clipped) == 0:
            average_slope = 0
            weighted_average_slope = 0
        else:
            average_slope = np.absolute(np.mean(np.array(slopes_clipped, dtype = float)))
            weighted_average_slope = np.absolute(np.average(np.array(slopes_clipped, dtype = float), weights = np.array(lengths, dtype = float)))
    else:
        if len(slopes) == 0:
            average_slope = 0
            weighted_average_slope = 0
        else:
            weighted_average_slope = np.average(np.array(slopes, dtype = float), weights = np.array(lengths, dtype = float))
            average_slope = np.mean(np.array(slopes, dtype = float))
    
    # Display the original image with lines drawn on it
    print(weighted_average_slope)
    if weighted_average_slope == 0:
        cv2.line(image, (int(image.shape[1]/2), 0), (int(image.shape[1]/2), image.shape[0]-1), (255,255,0), 2)
    else:
        cv2.line(image, (int(image.shape[1]/2), 0), (int((image.shape[0]-1)/average_slope) + int(image.shape[1]/2), image.shape[0]-1), (255,255,0), 2)
        cv2.line(image, (int(image.shape[1]/2), 0), (int((image.shape[0]-1)/weighted_average_slope) + int(image.shape[1]/2), image.shape[0]-1), (0,255,0), 2)
    ax2.imshow(image, cmap='gray')
    ax2.set_title("Line Fitting")
    ax3.hist(slopes, bins = 100)
    ax3.set_title("Slope Histogram")
    plot_title = f"Average Slope: {weighted_average_slope:.3f}"
    plt.tight_layout()
    # cv2.imshow(plot_title, image)
    # cv2.waitKey(0)
    if verbose:
        plt.show()
    if write:
        # make plot title
        # plot and save
        # plt.figure(1, figsize=(9, 6))
        plt.savefig(os.path.join(output_folder, filename+".png"), bbox_inches='tight', dpi=400)
        plt.close()
    # cv2.destroyAllWindows()
    return weighted_average_slope

def main(path='C:\\Users\\gt8mar\\capillary-flow\\tests\\kymo_test', verbose = False, write = False):
    # Set up paths
    input_folder = os.path.join(path, 'F_blood_flow', 'kymo')
    output_folder = os.path.join(path, 'F_blood_flow', 'velocities')
    metadata_folder = os.path.join(path, 'part_metadata')                           # This is for the test data
    # metadata_folder = os.path.join(os.path.dirname(path), 'part_metadata')        # This is for the real data
    
    # Read in the metadata
    metadata = pd.read_excel(os.path.join(metadata_folder,os.listdir(metadata_folder)[0]), sheet_name = 'Sheet1')
    print(metadata.head())

    # Read in the kymographs
    images = get_images(input_folder, "tiff")
    data = []
    
    for image in images: 
        kymo_raw = cv2.imread(os.path.join(input_folder, image), cv2.IMREAD_GRAYSCALE)

        # # Normalize rows of image (note this made our estimates worse)
        # norms = np.linalg.norm(kymo_raw, axis=1)
        # normalized_rows = (kymo_raw / norms[:, np.newaxis])*255
        # norm_blur = gaussian_filter(normalized_rows, sigma=2)
        # plt.imshow(norm_blur)
        # plt.show()
        # print(np.mean(normalized_rows))
        kymo_slice = kymo_raw[::5,:]
        kymo_blur = gaussian_filter(kymo_raw, sigma = 2)
        kymo_slice_blur = gaussian_filter(kymo_slice, sigma = 2)
        if write:
            base_name, extension = os.path.splitext(image)
            filename = base_name + "kymo_new_line"
            weighted_average_slope = find_slopes(kymo_blur, method = 'lasso', verbose = False, write=True, filename=filename)
        else:
            weighted_average_slope = find_slopes(kymo_blur, method = 'lasso', verbose = verbose, write=False)
        # transform slope from pixels/frames into um/s:
        um_slope = np.absolute(weighted_average_slope) *FPS/PIX_UM

        data.append(um_slope)
        print(f"Average slope: {weighted_average_slope:.3f} um/s")

    # plot_box_swarm(data, ["0.2 psi", "0.4 psi", "0.6 psi", "0.8 psi"], 
    #                "velocity (um/s)", "Participant_4 cap_4", "figure 1")
    # plot_box_swarm(data_slice, ["0.2 psi", "0.4 psi", "0.6 psi", "0.8 psi"],
    #                  "velocity (um/s)", "slice", "figure 2")
    return 0



"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main(write = False, verbose=True)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
