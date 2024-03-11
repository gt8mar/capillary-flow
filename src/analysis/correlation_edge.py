import os, platform
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import find_contours
from scipy.spatial.distance import cdist

def find_edge_points(mask, centerline_points):
    # Find contours of the mask, assuming the mask is binary
    contours = find_contours(mask, 0.5)
    contour_points = np.vstack(contours) # Combine all contour points
    
    # For each centerline point, find the closest two points on the mask edge
    edge_points = []
    for point in centerline_points:
        distances = cdist([point], contour_points, 'euclidean')
        sorted_distances_indices = np.argsort(distances[0])
        closest_point_idx = sorted_distances_indices[0]
        second_closest_point_idx = sorted_distances_indices[1]
        
        edge1 = tuple(contour_points[closest_point_idx])
        edge2 = tuple(contour_points[second_closest_point_idx])
        
        edge_points.append([tuple(point), edge1, edge2])
    
    return edge_points

def main():
    # load in masks
    masks_path = "C:\\Users\\gt8mar\\capillary-flow\\results\\segmented\\individual_caps_original"
    centerline_path = "C:\\Users\\gt8mar\\capillary-flow\\results\\centerlines"
    masks  = os.listdir(masks_path)
    for mask_name in masks:
        if mask_name.endswith(".png"):
            mask = cv2.imread(os.path.join(masks_path, mask_name), cv2.IMREAD_GRAYSCALE)
        else:
            continue
        identifier = mask_name.replace(".png", "").replace("seg_cap_", "")
        capillary_id = identifier.split("_")[-1]
        # remove capillary id from identifier
        video_identifier = identifier.replace("_" + capillary_id, "")
        # load in centerline with same identifier
        centerline_name = video_identifier + "_centerline_" + capillary_id + ".csv"
        
        # load in centerline
        centerline = np.loadtxt(os.path.join(centerline_path, centerline_name), delimiter=",")
        # remove 3rd column
        centerline = centerline[:, :2]

        # print(centerline.shape)

        # find edge points
        edge_points = find_edge_points(mask, centerline)
        # # # save edge points
        # # edge_points_path = "C:\\Users\\gt8mar\\capillary-flow\\results\\edge_points\\individual_caps_original"
        # # for edge_point in edge_points_path:
        # #     # np.save(edge_point, edge_points)
        # print(f'edge opints for {identifier}')
        # print(edge_points)

        # plot edge points and centerline points
        for point in edge_points:
            # plot using matplotlib
            plt.plot(point[0][1], point[0][0], 'ro')
            plt.plot(point[1][1], point[1][0], 'go')
            plt.plot(point[2][1], point[2][0], 'bo')
        plt.show()
        #     # plot first tuple: centerline point
        #     cv2.circle(mask, (int(point[0][1]), int(point[0][0])), 2, (0, 0, 100), -1)
        #     # plot second tuple: closest edge point
        #     cv2.circle(mask, (int(point[1][1]), int(point[1][0])), 2, (0, 150, 0), -1)
        #     # plot third tuple: second closest edge point
        #     cv2.circle(mask, (int(point[2][1]), int(point[2][0])), 2, (225, 0, 0), -1)
        # plt.imshow(mask)
        # plt.show()
            
            


    return 0


if __name__ == "__main__":
    main()