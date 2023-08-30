"""
Filename: make_skeletons_tools.py
------------------------------------------------------
This program skeletonizes a binary mask and prunes it to a single line.

By: Marcus Forst
"""

import os
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label
from skimage import io
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage import convolve
from collections import deque
from scipy.ndimage import distance_transform_edt
from fil_finder import FilFinder2D
import astropy.units as u
from skimage.morphology import medial_axis


# Skeletonize the binary mask
def perform_skeletonization(binary_image):
    return skeletonize(binary_image)

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
def find_junctions(skel):
    """Finds pixels with exactly three neighbors."""
    kernel = np.array([
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ])

    neighbors_count = convolve(skel.astype(int), kernel, mode='constant', cval=0)
    return (neighbors_count - 10 == 3) & skel



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
   
    BRANCH_THRESH = 20
    MIN_CAP_LEN = 5
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

if __name__ == "__main__":
    image_folder = 'C:\\Users\\gt8mar\\capillary-flow\\tests\\part09\\230414\\loc02\\segmented\\individual_caps_original'
    # image_names = ['set01_part09_230414_loc02_vid19_seg_cap_06a.png','set01_part09_230414_loc02_vid22_seg_cap_06a.png']
    image_names = ['set01_part09_230414_loc02_vid27_seg_cap_03b.png']
    for image_name in image_names:
        image_path = os.path.join(image_folder, image_name)
        import cv2
        binary_mask = (io.imread(image_path, as_gray=True) > 0.5).astype(int)
        print(binary_mask.shape)
        skeleton, skeleton_longest, radii = make_skeletons(binary_mask, plot=True)
        
        
        


        # plt.hist(distances.values())
        # plt.show()
    # skeleton = make_skeletons(binary_mask, plot=True)
    # # Assuming you have binary_mask and handled_skeleton ready
    # distances = distance_to_edge(binary_mask, skeleton)
    # print(distances)


