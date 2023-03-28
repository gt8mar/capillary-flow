"""
Filename: kymo_canny.py
-------------------------------------------------
This file uses canny edge detection to call average velocities from
kymographs. 

By: Marcus Forst
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import seaborn as sns
import time
from src.tools.get_images import get_images
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from sklearn.linear_model import Lasso

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
def plot_box_swarm(data, x_labels, y_axis_label,  plot_title, figure_name, verbose = True, write = False, remove_outliers = False):
    """Plot box-plot and swarm plot for data list.
 
    Args:
        data (list of list): List of lists with data to be plotted.
        y_axis_label (str): Y- axis label.
        x_labels (list of str): List with labels of x-axis.
        plot_title (str): Plot title.
        figure_name (str): Path to output figure.
         
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
def find_slopes(image, method = 'ridge', verbose = False, write = False, plot_title = "Kymograph", filename = "kymograph_1.png", remove_outliers = False):
    edges = cv2.Canny(image, 50, 110)
    print(f"the shape of the file is {edges.shape}")
    # Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Iterate through the contours
    slopes = []
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
            cv2.line(image, (image.shape[1]-1,righty), (0,lefty), (0,255,0), 2)
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

            # Draw the line on the original image
            cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0,255,0), 2)

            # Add line to list of lines
            slopes.append(slope[0])
    if remove_outliers:
        slopes_clipped, outliers = select_outliers(slopes)
        average_slope = np.absolute(np.mean(np.array(slopes_clipped, dtype = float)))
    else:
        average_slope = np.mean(np.array(slopes, dtype = float))
    
    # Display the original image with lines drawn on it
    if verbose:
        cv2.line(image, (int(image.shape[1]/2), 0), (int((image.shape[0]-1)/average_slope) + int(image.shape[1]/2), image.shape[0]-1), (255,255,0), 2)
        cv2.imshow(plot_title, image)
        cv2.waitKey(0)
    if write:
        input_folder = os.path.join('C:\\Users\\ejerison\\capillary-flow\\data\\processed', "set_01", 'participant_04_cap_04', "blood_flow")
        cv2.line(image, (int(image.shape[1]/2), 0), (int((image.shape[0]-1)/average_slope) + int(image.shape[1]/2), image.shape[0]-1), (255,255,0), 2)
        plt.figure(1, figsize=(9, 6))
        plt.title(plot_title)
        plt.imshow(image)
        plt.savefig(os.path.join(input_folder, filename), bbox_inches='tight', dpi=400)
        plt.close()
    cv2.destroyAllWindows()
    return slopes

def main(SET='set_01', sample = 'sample_000', verbose = False, write = False):
    input_folder = os.path.join('C:\\Users\\ejerison\\capillary-flow\\data\\processed', str(SET), 'participant_04_cap_04', "blood_flow")
    output_folder = os.path.join(input_folder, "centerlines")
    # Read in the mask
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

        kymo_blur = gaussian_filter(kymo_raw, sigma = 2)

        if write:
            base_name, extension = os.path.splitext(image)
            filename = base_name + "kymo_new_line.png"
            slopes = find_slopes(kymo_blur, method = 'lasso', verbose = False, write=True, filename=filename)
        else:
            slopes = find_slopes(kymo_blur, method = 'lasso', verbose = False, write=False)
        data.append(np.absolute(slopes))
        print(slopes)
        print(f"The average slope for {image} is {np.mean(np.array(slopes, dtype = float))}")
    plot_box_swarm(data, ["0.2 psi", "0.4 psi", "0.6 psi", "0.8 psi"], 
                   "flow (um^3/s)", "Flow vs pressure cap_4", "figure 1")

    return 0



"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main("set_01", "sample_009", write = False, verbose=True)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
