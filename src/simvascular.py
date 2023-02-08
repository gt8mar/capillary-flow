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

home_dir = "H:\\Marcus\\Data\\sample_000"
def select_points(coord_path, distance_path, skip = 100):
    coords = np.loadtxt(coord_path, delimiter = ',', dtype = float).astype(int)
    distances = np.loadtxt(distance_path, delimiter = ',', dtype = float)
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
    banana = sv.pathplanning.Path()
    coords, distances = select_points(coord_path, distance_path, skip = 80)
    for point in coords:
        x_coord = point[1]
        y_coord = point[0]
        banana.add_control_point([float(x_coord), float(y_coord), 0.0])
    return banana

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    SET = 'set_01'
    sample = 'sample_000'
    capillary = '7'
    banana = make_path(SET, sample, capillary)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))