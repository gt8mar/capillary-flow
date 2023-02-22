"""
Filename: find_centerline_v3.py
-------------------------------------------------
This file segments an image using ____ technique from scikit image

By: Marcus Forst

png to polygon credit: Stephan Hügel (https://gist.github.com/urschrei/a391f6e18a551f8cbfec377903920eca)
find skeletons: (https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html#sphx-glr-auto-examples-edges-plot-skeleton-py)
sort_continuous credit: Imanol Luengo (https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
import seaborn as sns
from skimage import measure
from skimage.morphology import medial_axis
from fil_finder import FilFinder2D
import astropy.units as u
import time
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from src.tools.get_images import get_images
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from skimage.segmentation import watershed
from skimage import filters
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.linear_model import Lasso


BRANCH_THRESH = 40
MIN_CAP_LEN = 50

def test():
    a = np.arange(6).reshape((2, 3))
    b = a.transpose()
    print(a)
    print(b)
    return 0
def enumerate_capillaries(image, test = False, verbose = False, write = False, write_path = None):
    """
    This function finds the number of capillaries and returns an array of images with one
    capillary per image.
    :param image: 2D numpy array
    :param short: boolian, if you want to test using one capillary. Default is false.
    :return: 3D numpy array: [capillary index, row, col]
    """
    row, col = image.shape
    print(row, col)
    contours = measure.find_contours(image, 0.8)
    print("The number of capillaries is: " + str(len(contours)))
    if test:
        contour_array = np.zeros((1, row, col))
        for i in range(1):
            grid = np.array(measure.grid_points_in_poly((row, col), contours[i]))
            contour_array[i] = grid
            # plt.plot(contours[i][:, 1], contours[i][:, 0], linewidth=2)   # this shows all of the enumerated capillaries
        # plt.show()
        return contour_array
    else:
        contour_array = np.zeros((len(contours), row, col))
        if  verbose or write:
            fig = plt.figure(figsize = (12,10))
            ax = fig.add_subplot(111)
        for i in range(len(contours)):
            grid = np.array(measure.grid_points_in_poly((row, col), contours[i]))
            contour_array[i] = grid
            if verbose or write:
                ax.plot(contours[i][:, 1], contours[i][:, 0], linewidth=2, label = "capillary " + str(i)) #plt.imshow(contour_array[i])   # plt.plot(contours[i][:, 1], contours[i][:, 0], linewidth=2) this shows all of the enumerated capillaries
                # plt.show()
        if verbose or write:
            ax.invert_yaxis()
            ax.legend(loc = 'center left')
            if write:
                fig.savefig(write_path)
            if verbose:
                plt.show()
        return contour_array
def make_skeletons(image, verbose = True, write = False, write_path = None):
    """
    This function uses the FilFinder package to find and prune skeletons of images.
    :param image: 2D numpy array or list of points that make up polygon mask
    :return fil.skeleton: 2D numpy array with skeletons
    :return distances: 1D numpy array that is a list of distances (which correspond to the skeleton coordinates)
    """
    # Load in skeleton class for skeleton pruning
    fil = FilFinder2D(image, beamwidth=0 * u.pix, mask=image)
    # Use separate method to get distances
    skeleton, distance = medial_axis(image, return_distance=True)
    # This is a necessary step for the fil object. It does nothing.
    fil.preprocess_image(skip_flatten=True)
    # This makes the skeleton
    fil.medskel()
    # This prunes the skeleton
    fil.analyze_skeletons(branch_thresh=BRANCH_THRESH * u.pix, prune_criteria='length',
                          skel_thresh=BRANCH_THRESH * u.pix)
    # Multiply the distances by the skeleton, selects out the distances we care about.
    distance_on_skeleton = distance * fil.skeleton
    distances = distance[fil.skeleton.astype(bool)]
    overlay = distance_on_skeleton + image
    # This plots the histogram of the capillary and the capillary with distance values.
    if verbose:
        plt.hist(distances)
        plt.show()
        plt.imshow(distance_on_skeleton, cmap='magma')
        plt.show()
    if write:
        if verbose:
            plt.imshow(overlay)
            plt.show()
        plt.imsave(write_path, overlay)
    return fil.skeleton, distances
def add_radii_value(distance_array):
    """
    This function creates a list of distances for the skeleton of an image
    :param distance_array: array of skeleton distance values
    :return: list of distances
    """
    skeleton_coordinates = np.transpose(np.nonzero(distance_array))
    distances = []
    for i in range(len(skeleton_coordinates)):
        row = skeleton_coordinates[i][0]
        col = skeleton_coordinates[i][1]
        distances.append(distance_array[row][col])
    return distances
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
def sort_continuous(array_2D, verbose = False):
    """
    This function takes a 2D array of shape (2, length) and sorts it in order of continuous points
    :param array_2D: 2D numpy array
    :param verbose: bool, shows plots if true.
    :return sorted_array: 2D numpy array
    :return opt_order: something that slices into the correct order when given a 1D array
    """
    if isinstance(array_2D, (list, np.ndarray)):
        points = np.c_[array_2D[0], array_2D[1]]
        neighbors = NearestNeighbors(n_neighbors=2).fit(points)
        graph = neighbors.kneighbors_graph()
        graph_connections = nx.from_scipy_sparse_array(graph)
        paths = [list(nx.dfs_preorder_nodes(graph_connections, i)) for i in range(len(points))]
        min_dist = np.inf
        min_idx = 0

        for i in range(len(points)):
            order = paths[i]  # order of nodes
            ordered = points[order]  # ordered nodes
            # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
            cost = (((ordered[:-1] - ordered[1:]) ** 2).sum(1)).sum()
            if cost < min_dist:
                min_dist = cost
                min_idx = i
        opt_order = paths[min_idx]
        row = array_2D[0][opt_order]
        col = array_2D[1][opt_order]
        sorted_array = np.c_[row, col]
        if verbose == True:
            plt.plot(col, row)
            plt.show()
            print(sorted_array)
            print(opt_order)
        return sorted_array, opt_order
    else:
        raise Exception('wrong type')
def watershed_seg(image, verbose = False):
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()
    return 0
def remove_outliers(data, lower_percentile = 10, upper_percentile = 90):
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
def plot_box_swarm(data, x_labels, y_axis_label,  plot_title, figure_name, verbose = True, write = False, remove_outliers = True):
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
        data_clipped, outliers = remove_outliers(data)
        # initialize plot information:
        sns.set(color_codes=True)
        plt.figure(1, figsize=(9, 6))
        plt.title(plot_title)

        # Create box and swarm plot with clipped data
        ax = sns.boxplot(data=data_clipped)
        sns.swarmplot(data=data_clipped, color=".25")
        sns.swarmplot(data = outliers, color = "red")
        if verbose:
            plt.show()

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
        ax.set_xticks(np.arange(4), labels = x_labels)
        ax.legend()
        
        if write:
            # write figure file with quality 400 dpi
            plt.savefig(figure_name, bbox_inches='tight', dpi=400)
        if verbose:
            plt.show()
    return 0
def find_slopes(image, method = 'ridge', verbose = False, write = False, plot_title = "Kymograph", filename = "kymograph_1.png", remove_outliers = True):
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
        slopes_clipped, outliers = remove_outliers(slopes)
        average_slope = np.absolute(np.mean(np.array(slopes_clipped, dtype = float)))
    else:
        average_slope = np.mean(np.array(slopes, dtype = float))
    
    # Display the original image with lines drawn on it
    if verbose:
        cv2.line(image, (int(image.shape[1]/2), 0), (int((image.shape[0]-1)/average_slope) + int(image.shape[1]/2), image.shape[0]-1), (255,255,0), 2)
        cv2.imshow(plot_title, image)
        cv2.waitKey(0)
    if write:
        input_folder = os.path.join('C:\\Users\\ejerison\\capillary-flow\\data\\processed', str(set_01), 'participant_04_cap_04', "blood_flow")
        cv2.line(image, (int(image.shape[1]/2), 0), (int((image.shape[0]-1)/average_slope) + int(image.shape[1]/2), image.shape[0]-1), (255,255,0), 2)
        plt.figure(1, figsize=(9, 6))
        plt.title(plot_title)
        plt.imshow(image)
        plt.savefig(os.path.join(input_folder, filename), bbox_inches='tight', dpi=400)
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

        # # Normalize rows of image
        # norms = np.linalg.norm(kymo_raw, axis=1)
        # normalized_rows = (kymo_raw / norms[:, np.newaxis])*255
        # norm_blur = gaussian_filter(normalized_rows, sigma=2)
        # plt.imshow(norm_blur)
        # plt.show()
        # print(np.mean(normalized_rows))

        kymo_blur = gaussian_filter(kymo_raw, sigma = 2)
        kymo_high_pass = kymo_raw - kymo_blur
        kymo_hp_blur = gaussian_filter(kymo_high_pass, sigma = 1) 
        # fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        # ax1.imshow(kymo_raw)
        # ax2.imshow(kymo_blur)
        # ax3.imshow(kymo_high_pass)
        # plt.show()
        # plt.imshow(kymo_hp_blur)
        # plt.show()
        val = filters.threshold_otsu(kymo_hp_blur)
        kymo_otsu = kymo_hp_blur<val
        # plt.imshow(kymo_otsu)
        # plt.show()
        kymo_despeckle = median_filter(kymo_otsu, size = 3)
        # plt.imshow(kymo_despeckle)
        # plt.show()
        # kymo_watershed = watershed_seg(kymo_despeckle, verbose = True)
        if write:
            base_name, extension = os.path.splitext(image)
            filename = base_name + "kymo_new_line.png"
            slopes = find_slopes(kymo_blur, method = 'lasso', verbose = False, write=True, filename=filename)
        else:
            slopes = find_slopes(kymo_blur, method = 'lasso', verbose = False, write=False)

        # slopes = find_slopes(norm_blur, method = 'lasso', verbose = False)
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
