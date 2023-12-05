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
import os, glob, platform
from skimage import measure
from skimage.morphology import medial_axis
from fil_finder import FilFinder2D
import astropy.units as u
import time
from sklearn.neighbors import NearestNeighbors
import networkx as nx
# from src.tools.parse_vid_path import parse_vid_path
from src.tools.parse_filename import parse_filename
from src.tools.enumerate_capillaries import enumerate_capillaries
from src.tools.enumerate_capillaries2 import find_connected_components
import warnings
import pandas as pd
from scipy.ndimage import convolve



BRANCH_THRESH = 40
MIN_CAP_LEN = 50

def find_junctions(skel):
    """Finds pixels with exactly three neighbors."""
    kernel = np.array([
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ])

    neighbors_count = convolve(skel.astype(int), kernel, mode='constant', cval=0)
    return (neighbors_count - 10 == 3) & skel
def find_endpoints(skel):
    # Define a kernel that counts the number of neighbors
    kernel = np.array([
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ])

    neighbors_count = convolve(skel.astype(int), kernel, mode='constant', cval=0)
    # Endpoints are skeleton points with only one neighbor
    return (neighbors_count == 11) & skel

def make_skeletons(binary_image, plot = False):
    """
    This function uses the FilFinder package to find and prune skeletons of images.
    Args:
        image: 2D numpy array or list of points that make up polygon mask
    Returns:
        skeleton: 2D numpy array with skeletons
        skeleton_longpath: 2D numpy array with skeletons that have been pruned to a single line
        radii: 1D numpy array that is a list of radii (which correspond to the skeleton coordinates)
    """
   
    # Load in skeleton class for skeleton pruning
    fil = FilFinder2D(binary_image, beamwidth=0 * u.pix, mask=binary_image)
    # Use separate method to get radii
    __, distance = medial_axis(binary_image, return_distance=True)
    # This is a necessary step for the fil object. It does nothing.
    fil.preprocess_image(skip_flatten=True)
    # This makes the skeleton
    fil.medskel(verbose=False)
    # find highest point in the skeleton (lowest row)
    junctions = np.asarray(np.nonzero(find_junctions(fil.skeleton))).shape[1]
    endpoints = np.asarray(np.nonzero(find_endpoints(fil.skeleton))).shape[1]
    print(f'Number of junctions is {junctions}')
    print(f'Number of endpoints is {endpoints}')
    print('-----------------------------------------------')
    if junctions == 0 and endpoints == 0:
        # Note: it's unclear if it is necessary to cut the loop. I think it makes sense for kymographs but it could work without.
        print('This is a loop')
        # POINTS_TO_CUT = 5
        # rows, cols = np.nonzero(fil.skeleton)
        # highest_point = (rows[0], cols[0])
        # sorted_skeleton_coords, optimal_order = sort_continuous(np.asarray(np.nonzero(fil.skeleton)))
        # top_points = np.concatenate((sorted_skeleton_coords[:POINTS_TO_CUT], sorted_skeleton_coords[-POINTS_TO_CUT:]), axis=1)
        # # blast a hole at the top of the loop
        # for point in top_points:
        #     row = point[0]
        #     col = point[1]
        #     print(f'row: {row}, col: {col}')
        #     fil.skeleton[row, col] = 0     

        #  # Multiply the radii by the skeleton, selects out the radii we care about.
        distance_on_skeleton = distance * fil.skeleton
        radii = distance[fil.skeleton.astype(bool)]
        # This makes an overlay of the skeleton and the distance values
        overlay = distance_on_skeleton + binary_image

        if plot:
            plt.imshow(overlay)
            plt.show()
        return fil.skeleton, fil.skeleton, radii
    else:    
        # This prunes the skeleton
        fil.analyze_skeletons(branch_thresh=BRANCH_THRESH * u.pix, prune_criteria='length',
                            skel_thresh=MIN_CAP_LEN * u.pix)
        # Multiply the radii by the skeleton, selects out the radii we care about.
        distance_on_skeleton = distance * fil.skeleton_longpath
        radii = distance[fil.skeleton_longpath.astype(bool)]
        # This makes an overlay of the skeleton and the distance values
        overlay = distance_on_skeleton + binary_image

        # plot the skeleton and the pruned skeleton
        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True)
            ax = axes.ravel()
            ax[0].imshow(binary_image, cmap=plt.cm.gray)
            ax[0].axis('off')
            ax[0].set_title('original', fontsize=20)
            ax[1].imshow(fil.skeleton, cmap=plt.cm.gray)
            ax[1].axis('off')
            ax[1].set_title('skeleton', fontsize=20)
            ax[2].imshow(fil.skeleton_longpath, cmap=plt.cm.gray)
            ax[2].axis('off')
            ax[2].set_title('cut', fontsize=20)
            fig.tight_layout()        
            plt.show()
        if plot:
            plt.imshow(overlay)
            plt.show()
        return fil.skeleton, fil.skeleton_longpath, radii
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
def load_image_with_prefix(input_folder, segmented_filename):
    # Define the varying parts of the filename
    varying_parts = ["", "bp", "scan"]

    # Loop through the possible filenames and load the image if it exists
    found_image = False
    for varying_part in varying_parts:
        full_filename = f"{segmented_filename.replace('_background', varying_part + '_background')}"
        matching_files = glob.glob(os.path.join(input_folder, full_filename))
        
        if matching_files:
            found_image = True
            image_path = matching_files[0]
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            break

    if found_image:
        # Image successfully loaded
        print("Image loaded:", image_path)
        return image
    else:
        print("Image not found.")
        return None

#  "F:\\Marcus\\data\\part09\\230414\\loc01" 'C:\\Users\\gt8mar\\capillary-flow\\tests\\part09\\230414\\loc02'


def main(path ='F:\\Marcus\\data\\hasty_seg\\230626\\part10\\230425\\loc01', # 'C:\\Users\\gt8mar\\capillary-flow\\tests\\part09\\230414\\loc02', #
        verbose = False, write = False, plot=False, hasty = True):
    """ Isolates capillaries from segmented image and finds their centerlines and radii. 

    Args: 
        path (str): Path to the umbrella location folder.
        verbose: bool, shows plots if true
        write: bool, saves plots if true
        plot: bool, shows plots if true
        hasty: bool, if true, uses hasty segmentation. If false, uses normal segmentation.

    Returns: 0 if successful

    Saves: Centerlines, radii, which capillaries are too small
    """
    
    # Ignore FilFinder warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="fil_finder.filament")
    
    # segmented folder
    if hasty:
        segmented_folder = os.path.join(path, "segmented", "hasty")
    else:
        segmented_folder = os.path.join(path, "segmented")
    os.makedirs(os.path.join(path, 'centerlines', 'coords'), exist_ok=True)
    os.makedirs(os.path.join(path, 'centerlines', 'images'), exist_ok=True)
    output_folder = os.path.join(path, 'centerlines')
    
    for file in os.listdir(segmented_folder):
        if file.endswith(".png"):
            participant, date, location, video, file_prefix = parse_filename(file)

            # Define the varying parts of the filename
            segmented_filename = file
            skeleton_filename = file_prefix + f"_{video}_skeletons.png"
            cap_map_filename = file_prefix + f"_{video}_cap_map.png"
         
            # Read in the mask
            segmented = load_image_with_prefix(segmented_folder, segmented_filename)
            # Make mask either 1 or 0
            segmented[segmented != 0] = 1

            # Make a numpy array of images with isolated capillaries. The mean/sum of this is segmented_2D.
            contours = find_connected_components(segmented)    # , verbose=False, write=write, write_path = os.path.join(output_folder, 'images', cap_map_filename)
            
            # if plot:
            #     for i in range(contours.shape[0]):
            #         plt.imshow(contours[i])
            #         plt.show()    
            # print(contours.shape)

            capillary_radii = []
            skeleton_coords = []
            flattened_radii = []
            used_capillaries = []
            skeleton_data = []
            j = 0
            for i in range(contours.shape[0]):
                # check to see if contours is zero
                if contours[i].shape[0] == 0:
                    pass
                elif np.nonzero(contours[i])[0].shape[0] <= MIN_CAP_LEN:
                    pass
                else:
                    # make skeleton
                    print(f"Making skeleton for capillary {i}")
                    skeleton, skeleton_longpath, radii = make_skeletons(contours[i], plot=False)     # Skeletons come out in the shape of the image
                    if plot:
                        fig, axes = plt.subplots(1,3, figsize=(10, 8), sharex=True, sharey=True)
                        ax = axes.ravel()
                        ax[0].imshow(skeleton, cmap=plt.cm.gray)
                        ax[0].axis('off')
                        ax[0].set_title('skeleton', fontsize=20)
                        ax[1].imshow(skeleton_longpath, cmap=plt.cm.gray)
                        ax[1].axis('off')
                        ax[1].set_title('cut', fontsize=20)
                        ax[2].imshow(contours[i], cmap=plt.cm.gray)
                        ax[2].axis('off')
                        ax[2].set_title('original', fontsize=20)
                        fig.tight_layout()
                        plt.show()
                    
                    skeleton_nums = np.asarray(np.nonzero(skeleton_longpath))
                    # omit small capillaries
                    print(f"Capillary {i} has {skeleton_nums.shape[1]} points")
                    if skeleton_nums.shape[1] <= MIN_CAP_LEN:
                        pass
                    else:
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
                            # Save centerline and radii information
                            for i in range(len(skeleton_coords)):
                                np.savetxt(os.path.join(output_folder, "coords", file_prefix + f'_{video}_centerline_coords_{str(i).zfill(2)}.csv'), 
                                        skeleton_data[i], delimiter=',', fmt = "%s")
                                if platform.system() == 'Windows':
                                    pass
                                else:
                                    os.makedirs('/hpc/projects/capillary-flow/results/centerlines', exist_ok=True)
                                    np.savetxt(os.path.join('/hpc/projects/capillary-flow/results/centerlines', file_prefix + f'_{video}_centerline_coords_{str(i).zfill(2)}.csv'), 
                                                skeleton_data[i], delimiter=',', fmt = "%s")
                                

            # # # Make overall histogram
            # # # plt.hist(flattened_radii)
            # # # plt.show()

            # # # TODO: Abnormal capillaries

    return 0



"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    test_path = 'F:\\Marcus\\data\\part12\\230428\\loc02'
    # main(path = '/hpc/projects/capillary-flow/data/part09/230414/loc01', verbose = False, write = True)
    main(path = test_path, verbose = True, write = False, plot=True)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
