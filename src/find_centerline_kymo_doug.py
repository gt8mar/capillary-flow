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
def find_lines(image, method = 'ridge'):
    edges = cv2.Canny(image, 50, 110)
    print(edges.shape)
    # Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Iterate through the contours
    for contour in contours:
        if method == 'ridge':
            # Fit a line to the contour using least squares regression
            [vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Compute the start and end points of the line
            lefty = int((-x*vy/vx) + y)
            righty = int(((image.shape[1]-x)*vy/vx)+y)
            print(lefty)
            print(righty)
            # Draw the line on the original image
            cv2.line(image, (image.shape[1]-1,righty), (0,lefty), (0,255,0), 2)
        if method == 'lasso':
            # Extract the x and y coordinates of the contour points
            x, y = contour[:, 0, 0], contour[:, 0, 1]
            
            # Fit a Lasso regression model to the contour
            lasso = Lasso(alpha=0.1)
            X = x.reshape(-1, 1) # Reshape the x array into a 2D array
            lasso.fit(X, y)
            
            # Compute the start and end points of the line
            start_x, end_x = 0, image.shape[1]-1
            start_y, end_y = lasso.predict([[start_x]]), lasso.predict([[end_x]])
            
            # Draw the line on the original image
            cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0,255,0), 2)



    # Display the original image with lines drawn on it
    cv2.imshow('Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cedges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    # print(lines)
    # if lines is not None:
    #     for i in range(len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv2.line(cedges, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    # cv2.imshow("title", cedges)
    # cv2.waitKey()
    return 0

def main(SET='set_01', sample = 'sample_000', verbose = False, write = False):
    input_folder = os.path.join('C:\\Users\\ejerison\\capillary-flow\\data\\processed', str(SET), 'participant_04_cap_04', "blood_flow")
    output_folder = os.path.join(input_folder, "centerlines")
    # Read in the mask
    images = get_images(input_folder, "tiff")
    for image in images: 
        kymo_raw = cv2.imread(os.path.join(input_folder, image), cv2.IMREAD_GRAYSCALE)
        print(np.mean(kymo_raw))
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
        find_lines(kymo_blur, method = 'lasso')
        
        
        


        



        # Make mask either 1 or 0
        


    #     # save to results
    #     total_skeleton, total_distances = make_skeletons(segmented, verbose = False, write = write, 
    #                                                     write_path=os.path.join(output_folder, f'{image}_background_skeletons.png'))

    #     # Make a numpy array of images with isolated capillaries. The mean/sum of this is segmented_2D.
    #     contours = enumerate_capillaries(segmented, verbose=False, write=True, write_path = os.path.join(output_folder, f"{image}_cap_map.png"))
    #     capillary_distances = {}
    #     skeleton_coords = {}
    #     flattened_distances = []
    #     used_capillaries = []
    #     j = 0
    #     for i in range(contours.shape[0]):
    #         skeleton, distances = make_skeletons(contours[i], verbose=False)     # Skeletons come out in the shape
    #         skeleton_nums = np.asarray(np.nonzero(skeleton))
    #         # omit small capillaries
    #         if skeleton_nums.shape[1] <= MIN_CAP_LEN:
    #             used_capillaries.append([i, "small", str(skeleton_nums.shape[1])])
    #             pass
    #         else:
    #             used_capillaries.append([i, f"new_capillary_{j}", str(skeleton_nums.shape[1])])
    #             j += 1
    #             sorted_skeleton_coords, optimal_order = sort_continuous(skeleton_nums, verbose=False)
    #             ordered_distances = distances[optimal_order]
    #             capillary_distances[str(i)] = ordered_distances
    #             flattened_distances += list(distances)
    #             skeleton_coords[str(i)] = sorted_skeleton_coords
    #     print(f"{len(skeleton_coords.keys())}/{contours.shape[0]} capillaries used")
    #     if verbose:
    #         plt.show()
    #         # Plot all capillaries together      
    #             # plt.plot(capillary_distances[i])
    #             # plt.title(f'Capillary {i} radii')
    #             # plt.show()

    #     if write:
    #         np.savetxt(os.path.join(output_folder, f'{image}_cap_cut.csv'),
    #                             np.array(used_capillaries), delimiter = ',',
    #                             fmt = '%s')
    #         for key in skeleton_coords:
    #             np.savetxt(os.path.join(input_folder, "coords", f'{image}_skeleton_coords_{str(key).zfill(2)}.csv'), 
    #                     skeleton_coords[key], delimiter=',', fmt = '%s')
    #             np.savetxt(os.path.join(input_folder, "distances", f'{image}_capillary_distances_{str(key).zfill(2)}.csv'), 
    #                     capillary_distances[key], delimiter=',', fmt = '%s')


    #     # # Make overall histogram
    #     # # plt.hist(flattened_distances)
    #     # # plt.show()

    #     # # TODO: Write program to register radii maps with each other 
    #     # # TODO: Abnormal capillaries, how do.

    return 0



"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main("set_01", "sample_009", write = True, verbose=False)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
