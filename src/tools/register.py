"""
Filename: register.py
-------------------------------------------------------------
This file 
by: Gabby Rincon & ChatGPT
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    source_img = cv2.imread("D:\\data_gabby\\230201\\vid7\\moco\\part_7_230201_7_0000.tif", cv2.COLOR_BGR2GRAY)
    target_img = cv2.imread("D:\\data_gabby\\230201\\vid3\\moco\\part_7_230201_3_0000.tif", cv2.COLOR_BGR2GRAY)

    # Initialize the SIFT feature detector and descriptor
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for the source and target images
    keypoints1, descriptors1 = sift.detectAndCompute(source_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(target_img, None)

    # Create a brute-force matcher
    matcher = cv2.BFMatcher()

    # Match descriptors between the source and target images
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    # Calculate the translation parameters
    dx_sum = 0.0
    dy_sum = 0.0
    num_matches = len(good_matches)

    for match in good_matches:
        print(match)
        # Get the keypoints for the matched pair
        kp1 = keypoints1[match.queryIdx]
        kp2 = keypoints2[match.trainIdx]

        # Calculate the translation between keypoints
        dx = kp2.pt[0] - kp1.pt[0]
        dy = kp2.pt[1] - kp1.pt[1]

        dx_sum += dx
        dy_sum += dy

    # Calculate the average translation
    dx_avg = dx_sum / num_matches
    dy_avg = dy_sum / num_matches

    # Calculate transformation matrix
    transformation_matrix = np.array([[1, 0, dx_avg], [0, 1, dy_avg]], dtype=np.float32)

    # Apply transformation
    registered_img = cv2.warpAffine(source_img, transformation_matrix, (target_img.shape[1], target_img.shape[0]))

    # Save registered image
    dest_path = "D:\\data_gabby\\230201\\vid7\\moco registered\\part_7_230201_7_0000_reg2.tif"
    cv2.imwrite(dest_path, registered_img)
    print("saved")
"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))