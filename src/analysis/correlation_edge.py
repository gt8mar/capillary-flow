import os, platform
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import find_contours
from scipy.spatial.distance import cdist

def find_closest_contour_points_on_column(contour, point):
    """
    Find the closest contour points on the specified column.
    """
    column = point[1]
    contour_columns = contour[:, 1]
    column_diff = np.abs(contour_columns - column)
    closest_point_idx = np.argmin(column_diff)
    closest_point = contour[closest_point_idx]
    print(f'closest point: {closest_point}')
    # Round the coordinates to the nearest integer
    return np.round(closest_point).astype(int)

def split_contour_at_points(contour, split_points):
    """
    Split the contour into two parts at the closest points to the given split points, using a tolerance.
    """
    # Find indices of the closest points in the contour to the split points
    indices = []
    for point in split_points:
        # Calculate the Euclidean distance from each contour point to the split point
        distances = np.sqrt(((contour - point) ** 2).sum(axis=1))
        closest_point_idx = np.argmin(distances)
        indices.append(closest_point_idx)
    
    indices.sort()  # Ensure indices are in ascending order

    # Split the contour into two parts. This logic assumes contour is a simple loop
    # and might need adjustment for more complex shapes or multiple contours
    if indices[0] < indices[1]:
        contour1 = np.vstack([contour[:indices[0]+1], contour[indices[1]:]])
        contour2 = contour[indices[0]:indices[1]+1]
    else:  # Handle the case where indices might not be in expected order
        contour1 = contour[indices[1]:indices[0]+1]
        contour2 = np.vstack([contour[:indices[1]+1], contour[indices[0]:]])

    return contour1, contour2

def find_edge_points(mask, centerline_points):
    contours = find_contours(mask, 0.5)
    contour = max(contours, key=len)  # Assuming the longest contour
    
    # Use the columns of the first and last centerline points to find closest contour points vertically
    first_point = centerline_points[0]
    last_point = centerline_points[-1]
    closest_start = find_closest_contour_points_on_column(contour, first_point)
    closest_end = find_closest_contour_points_on_column(contour, last_point)
    print(f'closest start: {closest_start}')
    print(f'closest end: {closest_end}')
    # Split the contour into two parts
    contour1, contour2 = split_contour_at_points(contour, [closest_start, closest_end])
    
    # For each centerline point, find the closest points on each of the two contours
    edge_points = []
    for point in centerline_points:
        distances1 = cdist([point], contour1, 'euclidean')
        distances2 = cdist([point], contour2, 'euclidean')
        closest_point_idx1 = np.argmin(distances1)
        closest_point_idx2 = np.argmin(distances2)
        edge_left = tuple(contour1[closest_point_idx1])
        edge_right = tuple(contour2[closest_point_idx2])
        
        edge_points.append([tuple(point), edge_left, edge_right])
    
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