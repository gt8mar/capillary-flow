"""
Filename: make_skeletons.py
------------------------------------------------------
This program skeletonizes a binary mask and prunes it to a single line.

By: Marcus Forst
"""

import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label
from skimage import io
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage import convolve
from collections import deque
from scipy.ndimage import distance_transform_edt



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

# Prune the skeleton to produce a single line
def prune_skeleton(skel):
    """
    This function prunes the skeleton to produce a single line by iteratively removing endpoints.
    """
    pruned = skel.copy()
    endpoints = find_endpoints(pruned)
    while np.sum(endpoints) > 2:  # More than two endpoints mean it's not a single line
        for i in range(pruned.shape[0]):
            for j in range(pruned.shape[1]):
                if endpoints[i, j]:
                    pruned[i, j] = 0
                    break
            if np.sum(endpoints) > 2:
                endpoints = find_endpoints(pruned)
    return pruned

# Handle loops in the skeleton
def handle_loops(skel):
    """
    This function handles loops in the skeleton by removing the smaller loop.
    """
    labels, num = label(skel, return_num=True)
    if num > 1:
        # Keep only the largest connected component
        sizes = [np.sum(labels == i) for i in range(1, num + 1)]
        largest_label = np.argmax(sizes) + 1
        return labels == largest_label
    return skel
def neighbors(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1), (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]

def find_junctions(skel):
    """Finds pixels with exactly three neighbors."""
    kernel = np.array([
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ])

    neighbors_count = convolve(skel.astype(int), kernel, mode='constant', cval=0)
    return (neighbors_count - 10 == 3) & skel

def dfs_iterative(start, skel, stop_at_junction=False):
    """Iterative Depth-First Search to traverse and record path."""
    stack = [start]
    path = []
    visited = set()

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            path.append(vertex)

            for neighbor in neighbors(vertex):
                if skel[neighbor] and neighbor not in visited:
                    if stop_at_junction and find_junctions(skel)[neighbor]:
                        return path
                    stack.append(neighbor)

    return path

def cut_loops(skel):
    junctions = find_junctions(skel)
    for junction in np.argwhere(junctions):
        junction = tuple(junction)
        paths = []
        
        # Traverse in each direction from the junction
        for neighbor in neighbors(junction):
            if skel[neighbor]:
                path = dfs_iterative(neighbor, skel, stop_at_junction=True)
                paths.append(path)
        
        # Sort paths by length and cut the shortest
        paths.sort(key=len)
        if len(paths) > 1:
            for p in paths[0]:
                skel[p] = 0
    return skel



def make_skeletons(binary_image, plot=False):
    """
    This function skeletonizes a binary mask and prunes it to a single line.

    Args:
        binary_image (numpy.ndarray): the binary mask
        plot (bool): whether to plot the result
    Returns:
        numpy.ndarray: the skeletonized binary mask
    """

    skeleton = perform_skeletonization(binary_image)
    pruned_skeleton = prune_skeleton(skeleton)
    handled_skeleton = handle_loops(pruned_skeleton)
    copy_of_handled_skeleton = handled_skeleton.copy()
    cut_skeleton = cut_loops(copy_of_handled_skeleton)

    if plot:
        # Display the result
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(binary_image, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('original', fontsize=20)
        ax[1].imshow(skeleton, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('skeleton', fontsize=20)
        ax[2].imshow(cut_skeleton, cmap=plt.cm.gray)
        ax[2].axis('off')
        ax[2].set_title('cut', fontsize=20)
        fig.tight_layout()
        plt.show()
    return cut_skeleton


def distance_to_edge(binary_mask, skeleton):
    # Inverse of binary mask, as we want distances to the zeroes (boundary)
    dist_transform = distance_transform_edt(~binary_mask)
    
    skeleton_distances = {}
    for point in np.argwhere(skeleton):
        tuple_point = tuple(point)
        skeleton_distances[tuple_point] = dist_transform[tuple_point]
        
    return skeleton_distances


if __name__ == "__main__":
    image_path = 'C:\\Users\\gt8mar\\capillary-flow\\tests\\part09\\230414\\loc02\\segmented\\individual_caps_original\\set01_part09_230414_loc02_vid24_seg_cap_01a.png'
    import cv2
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = (io.imread(image_path, as_gray=True) > 0.5).astype(int)
    print(binary_mask.shape)
    skeleton = make_skeletons(binary_mask, plot=True)
    # Assuming you have binary_mask and handled_skeleton ready
    distances = distance_to_edge(binary_mask, skeleton)
    print(distances)


