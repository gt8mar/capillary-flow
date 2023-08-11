"""
Filename: simvascular.py
-------------------------
This file reads csv files of centerline 
coordinates and respective radii and creates
simvascular paths and segmentations. 
"""

import os
import numpy as np
import sv
import time

POINTS = 40

home_dir = "H:\\Marcus\\Data\\sample_000"
def select_points(coord_path, distance_path, points = 10):
    coords = np.loadtxt(coord_path, delimiter = ',', dtype = float).astype(int)
    distances = np.loadtxt(distance_path, delimiter = ',', dtype = float)
    skip = distances.shape[0]//points
    selected_coords = coords[::skip]
    selected_dist = distances[::skip]
    x_col = selected_coords[:,0]
    y_col = selected_coords[:,1]
    xmax = np.max(x_col)
    ymax = np.max(y_col)
    xmin = np.min(x_col)
    ymin = np.min(y_col)
    xshifts = np.repeat(-(xmin + (xmax-xmin)//2),x_col.shape[0])
    yshifts = np.repeat(-(ymin + (ymax-ymin)//2),y_col.shape[0])
    offset_file = np.transpose(np.vstack((xshifts, yshifts)))
    shifted_coords = selected_coords + offset_file
    return shifted_coords, selected_dist

def make_path(SET, sample, capillary):
    coord_path = os.path.join(home_dir, "E_centerline", "set_01_sample_000_skeleton_coords_07.csv")
    distance_path = os.path.join(home_dir, "D_segmented", "set_01_sample_000_capillary_distances_07.csv")
    name = "set_01_sample_000_path_07"
    path = sv.pathplanning.Path()
    coords, distances = select_points(coord_path, distance_path, points = POINTS)
    for point in coords:
        x_coord = point[0]
        y_coord = point[1]
        path.add_control_point([float(x_coord), float(y_coord), 0.0])
    return path

def make_segmentations(SET, sample, capillary, path):
    coord_path = os.path.join(home_dir, "E_centerline", "set_01_sample_000_skeleton_coords_07.csv")
    distance_path = os.path.join(home_dir, "D_segmented", "set_01_sample_000_capillary_distances_07.csv")
    coords, distances = select_points(coord_path, distance_path, points = POINTS)
    segmentation_list = []
    for i in range(distances.shape[0]-2):
        radius = distances[i]
        center = path.get_curve_point(i*3)
        normal = path.get_curve_tangent(i*3)
        segmentation = sv.segmentation.Circle(radius = radius, center = center, normal = normal)
        segmentation_list.append(segmentation)
    return segmentation_list
    

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    SET = 'set_01'
    sample = 'sample_000'
    capillary = '7'
    path = make_path(SET, sample, capillary)
    sv.dmg.add_path("test2", path)
    segmentation_list = make_segmentations(SET, sample, capillary, path)
    sv.dmg.add_segmentation("test2", "test2", segmentation_list)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))