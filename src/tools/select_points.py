import os
import time
import numpy as np

path = "H:\\Marcus\\Data\\sample_000\\E_centerline\\set_01_sample_000_skeleton_coords_07.csv"
distance_path = "H:\\Marcus\\Data\\sample_000\\D_segmented\\set_01_sample_000_capillary_distances_07.csv"

def select_points(coord_path, distance_path, skip = 52):
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

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    shifted_coords, selected_dist = select_points(path, distance_path)
    print(selected_dist)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))