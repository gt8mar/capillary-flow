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

def neighbors(i, j, visited, skeleton):
    height, width = skeleton.shape
    deltas = [-1, 0, 1]
    for dx in deltas:
        for dy in deltas:
            x, y = i + dx, j + dy
            if 0 <= x < height and 0 <= y < width and skeleton[x][y] == 1 and (x, y) not in visited:
                yield (x, y)
def dfs(i, j, parent, visited, skeleton):
    stack = [(i, j)]
    path = [(i, j)]
    while stack:
        current = stack.pop()
        visited.add(current)
        for neighbor in neighbors(*current, visited, skeleton):
            if neighbor != parent:
                if neighbor in path:
                    return path[path.index(neighbor):]  # Loop detected
                stack.append(neighbor)
                path.append(neighbor)
    return []
def find_largest_loop(skeleton):
    height, width = skeleton.shape
    visited = set()
    largest_loop = []

    for i in range(height):
        for j in range(width):
            if skeleton[i][j] == 1 and (i, j) not in visited:
                loop = dfs(i, j, None, visited, skeleton)
                if len(loop) > len(largest_loop):
                    largest_loop = loop

    return largest_loop

def find_loop_contours(skeleton):
    skeleton_copy = skeleton.np.astype(np.uint8)
    
    # Thicken the skeleton using dilation
    kernel = np.ones((3,3), np.uint8)
    thickened_skeleton = skeleton_copy #cv2.dilate(skeleton_copy, kernel, iterations=1)

    contours = cv2.findContours(thickened_skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return fil.skeleton, fil.skeleton_longpath, np.array([])
    else:
        hierarchy = contours[1] if len(contours) == 2 else contours[2]
        contours = contours[0] if len(contours) == 2 else contours[1]
        hierarchy = hierarchy[0]

    count = 0
    result = cv2.merge([skeleton_copy,skeleton_copy,skeleton_copy])
    for component in zip(contours, hierarchy):
        cntr = component[0]
        hier = component[1]
        # discard outermost no parent contours and keep innermost no child contours
        # hier = indices for next, previous, child, parent
        # no parent or no child indicated by negative values
        if (hier[3] > -1) & (hier[2] < 0):
            count = count + 1
            cv2.drawContours(result, [cntr], 0, (0,0,255), 2)
        # contour_x, contour_y = contours[0][:,0,0], contours[0][:,0,1]
        # plt.scatter(contour_x, contour_y)
        # plt.show()
        # cv2.drawContours(skeleton_copy, [contours[0]], -1, (255, 255, 0), 3, hierarchy=hierarchy)
        cv2.imshow('contour', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return contours

def make_skeletons(binary_image, plot = False):
    """
    This function uses the FilFinder package to find and prune skeletons of images.
    :param image: 2D numpy array or list of points that make up polygon mask
    :return fil.skeleton: 2D numpy array with skeletons
    :return radii: 1D numpy array that is a list of radii (which correspond to the skeleton coordinates)
    """
    BRANCH_THRESH = 20  # Branches must be at least this many pixels long
    MIN_CAP_LEN = 5  # Caps must be at least this many pixels long
    # Load in skeleton class for skeleton pruning
    fil = FilFinder2D(binary_image, beamwidth=0 * u.pix, mask=binary_image)
    # Use separate method to get radii
    __, distance = medial_axis(binary_image, return_distance=True)
    # This is a necessary step for the fil object. It does nothing.
    fil.preprocess_image(skip_flatten=True)
    # This makes the skeleton
    fil.medskel()
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


