"""
Filename: simvascular.py
-------------------------
This file reads csv files of centerline 
coordinates and respective radii and creates
simvascular paths and segmentations. 
"""

import os
import numpy as np
import sv.pathplanning as pp
import time

home_dir = "C:\\ejerison\\capillary-flow\\data\\processed"
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
    coord_path = os.path.join(home_dir, str(SET), str(sample), "E_centerline", f"{SET}_{sample}_skeleton_coords_", str(capillary).zfill(2),".csv")
    distance_path = os.path.join(home_dir, str(SET), str(sample), "D_segmented", f"{SET}_{sample}_capillary_distances_", str(capillary).zfill(2),".csv")
    name = f"{SET}_{sample}_path_{capillary}"
    path = pp.Path()
    coords, distances = select_points(coord_path, distance_path, skip = 80)
    for point in coords:
        x_coord = point[1]
        y_coord = point[0]
        path.add_control_point([x_coord, y_coord, 0])
    return 0

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    SET = 'set_01'
    sample = 'sample_001'
    capillary = '7'
    make_path(SET, sample, capillary)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))