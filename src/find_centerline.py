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
from skimage import measure
from skimage.morphology import medial_axis
from fil_finder import FilFinder2D
import astropy.units as u
import time
from sklearn.neighbors import NearestNeighbors
import networkx as nx

BRANCH_THRESH = 40
MIN_CAP_LEN = 150

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
def make_skeletons(image, verbose = True, histograms = False, write = False, write_path = None):
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
        if histograms:
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

def main(SET='set_01', sample = 'sample_000', verbose = False, write = False):
    input_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'D_segmented')
    output_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'E_centerline')
    # Read in the mask
    segmented = cv2.imread(os.path.join(input_folder, f'{SET}_{sample}_background.png'), cv2.IMREAD_GRAYSCALE)
    # Make mask either 1 or 0
    segmented[segmented != 0] = 1

    # save to results
    total_skeleton, total_distances = make_skeletons(segmented, verbose = verbose, write = write, 
                                                     write_path=os.path.join(output_folder, f'{SET}_{sample}_background_skeletons.png'))

    # Make a numpy array of images with isolated capillaries. The mean/sum of this is segmented_2D.
    contours = enumerate_capillaries(segmented, verbose=False, write=write, write_path = os.path.join(input_folder, f"{SET}_{sample}_cap_map.png"))
    capillary_distances = []
    skeleton_coords = []
    flattened_distances = []
    used_capillaries = []
    j = 0
    for i in range(contours.shape[0]):
        skeleton, distances = make_skeletons(contours[i], verbose=False, histograms = False)     # Skeletons come out in the shape
        skeleton_nums = np.asarray(np.nonzero(skeleton))
        # omit small capillaries
        if skeleton_nums.shape[1] <= MIN_CAP_LEN:
            used_capillaries.append(["small", str(skeleton_nums.shape[1])])
            pass
        else:
            used_capillaries.append([f"new_capillary_{j}", str(skeleton_nums.shape[1])])
            j += 1
            sorted_skeleton_coords, optimal_order = sort_continuous(skeleton_nums, verbose=False)
            ordered_distances = distances[optimal_order]
            capillary_distances.append(ordered_distances)
            flattened_distances += list(distances)
            skeleton_coords.append(sorted_skeleton_coords)
    print(f"{len(skeleton_coords)}/{contours.shape[0]} capillaries used")
    if verbose:
        plt.show()
        # Plot all capillaries together      
            # plt.plot(capillary_distances[i])
            # plt.title(f'Capillary {i} radii')
            # plt.show()

    if write:
        np.savetxt(os.path.join(input_folder, f'{SET}_{sample}_cap_cut.csv'),
                            np.array(used_capillaries), delimiter = ',',
                            fmt = '%s')
        for i in range(len(skeleton_coords)):
            np.savetxt(os.path.join(output_folder, "coords", f'{SET}_{sample}_skeleton_coords_{str(i).zfill(2)}.csv'), 
                    skeleton_coords[i], delimiter=',', fmt = "%s")
            np.savetxt(os.path.join(output_folder, "distances", f'{SET}_{sample}_capillary_distances_{str(i).zfill(2)}.csv'), 
                    capillary_distances[i], delimiter=',', fmt = '%s')


    # # Make overall histogram
    # # plt.hist(flattened_distances)
    # # plt.show()

    # # TODO: Write program to register radii maps with each other 
    # # TODO: Abnormal capillaries, how do.

    return 0



"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    for i in range(1,6):
        main("set_01", "sample_00"+ str(i), write = False, verbose=True)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
