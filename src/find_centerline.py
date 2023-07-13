"""
Filename: find_centerline.py
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
from src.tools.parse_vid_path import parse_vid_path
import warnings


BRANCH_THRESH = 40
MIN_CAP_LEN = 50

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
    :return radii: 1D numpy array that is a list of radii (which correspond to the skeleton coordinates)
    """
    # Load in skeleton class for skeleton pruning
    fil = FilFinder2D(image, beamwidth=0 * u.pix, mask=image)
    # Use separate method to get radii
    skeleton, distance = medial_axis(image, return_distance=True)
    # This is a necessary step for the fil object. It does nothing.
    fil.preprocess_image(skip_flatten=True)
    # This makes the skeleton
    fil.medskel()
    # This prunes the skeleton
    fil.analyze_skeletons(branch_thresh=BRANCH_THRESH * u.pix, prune_criteria='length',
                          skel_thresh=BRANCH_THRESH * u.pix)
    # Multiply the radii by the skeleton, selects out the radii we care about.
    distance_on_skeleton = distance * fil.skeleton
    radii = distance[fil.skeleton.astype(bool)]
    overlay = distance_on_skeleton + image
    # This plots the histogram of the capillary and the capillary with distance values.
    if verbose:
        if histograms:
            plt.hist(radii)
            plt.show()
        plt.imshow(distance_on_skeleton, cmap='magma')
        plt.show()
    if write:
        if verbose:
            plt.imshow(overlay)
            plt.show()
        plt.imsave(write_path, overlay)
    return fil.skeleton, radii
def add_radii_value(distance_array):
    """
    This function creates a list of radii for the skeleton of an image
    :param distance_array: array of skeleton distance values
    :return: list of radii
    """
    skeleton_coordinates = np.transpose(np.nonzero(distance_array))
    radii = []
    for i in range(len(skeleton_coordinates)):
        row = skeleton_coordinates[i][0]
        col = skeleton_coordinates[i][1]
        radii.append(distance_array[row][col])
    return radii
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

def main(path = "F:\\Marcus\\data\\part13\\230428\\vid25",
        verbose = False, write = False):
    """ Isolates capillaries from segmented image and finds their centerlines and radii. 

    Args: 
        path (str): Path to the umbrella video folder.
        verbose: bool, shows plots if true
        write: bool, saves plots if true

    Returns: 0 if successful

    Saves: Centerlines, radii, which capillaries are too small
    """
    
    # Ignore FilFinder warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="fil_finder.filament")
    
    # Set up paths
    input_folder = os.path.join(path, 'D_segmented')
    os.makedirs(os.path.join(path, 'E_centerline', 'coords'), exist_ok=True)
    os.makedirs(os.path.join(path, 'E_centerline', 'images'), exist_ok=True)
    output_folder = os.path.join(path, 'E_centerline')

    # extract metadata from path
    participant, date, video = parse_vid_path(path)
    SET = 'set_01'
    file_prefix = f'{SET}_{participant}_{date}_{video}'
    segmented_filename = file_prefix + '_background_seg.png'
    skeleton_filename = file_prefix + '_background_skeletons.png'
    cap_map_filename = file_prefix + '_cap_map.png'

    # Read in the mask
    segmented = cv2.imread(os.path.join(input_folder, segmented_filename), 
                           cv2.IMREAD_GRAYSCALE)
    # Make mask either 1 or 0
    segmented[segmented != 0] = 1

    # # save to results
    # total_skeleton, total_radii = make_skeletons(segmented, verbose = verbose, write = write, 
    #                                                  write_path=os.path.join(output_folder,'images', skeleton_filename))

    # Make a numpy array of images with isolated capillaries. The mean/sum of this is segmented_2D.
    contours = enumerate_capillaries(segmented, verbose=False, write=write, write_path = os.path.join(input_folder, cap_map_filename))
    
    if write:
        # save segmented_2D
        segmented_2D = np.sum(contours, axis=0).astype(bool)
        # segmented_2D[segmented_2D != 0] = 1
        # save to results
        np.savetxt(os.path.join(output_folder, 'centerline_mask.txt'), segmented_2D, delimiter=',')

    capillary_radii = []
    skeleton_coords = []
    flattened_radii = []
    used_capillaries = []
    skeleton_data = []
    j = 0
    for i in range(contours.shape[0]):
        # make skeleton
        print(f"Making skeleton for capillary {i}")
        skeleton, radii = make_skeletons(contours[i], verbose=False, histograms = False)     # Skeletons come out in the shape
        skeleton_nums = np.asarray(np.nonzero(skeleton))
        # omit small capillaries
        if skeleton_nums.shape[1] <= MIN_CAP_LEN:
            used_capillaries.append(["small", str(skeleton_nums.shape[1])])
            pass
        else:
            used_capillaries.append([f"new_capillary_{j}", str(skeleton_nums.shape[1])])
            j += 1

            # Sort skeleton points in order of continuous points
            sorted_skeleton_coords, optimal_order = sort_continuous(skeleton_nums, verbose=False)
            ordered_radii = radii[optimal_order]
            skeleton_coords_with_radii = np.column_stack((sorted_skeleton_coords, ordered_radii))
            capillary_radii.append(ordered_radii)
            flattened_radii += list(radii)
            # Attach capillary_radii to skeleton_coords
            skeleton_coords.append(sorted_skeleton_coords)
            skeleton_data.append(skeleton_coords_with_radii)
    print(f"{len(skeleton_coords)}/{contours.shape[0]} capillaries used")
    if verbose:
        plt.show()
        # Plot all capillaries together      
            # plt.plot(capillary_radii[i])
            # plt.title(f'Capillary {i} radii')
            # plt.show()

    if write:
        # Save which capillaries were dropped out
        np.savetxt(os.path.join(input_folder, file_prefix + '_cap_cut.csv'), 
                            np.array(used_capillaries), delimiter = ',',
                            fmt = '%s')
        # Save centerline and radii information
        for i in range(len(skeleton_coords)):
            np.savetxt(os.path.join(output_folder, "coords", file_prefix + f'_centerline_coords_{str(i).zfill(2)}.csv'), 
                    skeleton_data[i], delimiter=',', fmt = "%s")
            # np.savetxt(os.path.join(output_folder, "radii", file_prefix + f'_capillary_radii_{str(i).zfill(2)}.csv'), 
            #         capillary_radii[i], delimiter=',', fmt = '%s')


    # # Make overall histogram
    # # plt.hist(flattened_radii)
    # # plt.show()

    # # TODO: Abnormal capillaries

    return 0



"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main(path = '/hpc/projects/capillary-flow/data/part13/230428/vid25', verbose = False, write = True)
    # main(verbose = True, write = True)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
