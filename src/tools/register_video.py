"""
Filename: register_video.py
-------------------------------------------------------------
This file registers all frames of a video to each other (i.e. stabilizes).
The registered frames are saved in a new folder 'stabilized' with a csv 
file of the x and y translations per frame.
by: Gabby Rincon & ChatGPT
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv

# This function registers the target image to the source image.
# This only accounts for linear translations in x and y.
# SIFT feature detection algorithm is used. 
# If matches are not found between target & source image, the target image is translated by prev_dx and prev_dy if provided, and not translated otherwise.
def register_img(source_img_fp, target_img_fp, prev_dx=0, prev_dy=0):
    #get images from filepath
    source_img = cv2.imread(source_img_fp)
    target_img = cv2.imread(target_img_fp)
    
    #find keypoints & descriptors
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(source_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(target_img, None)

    #match descriptors
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    #filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance: #free parameter
            good_matches.append(m)

    """# Draw the matches between the source and target images
    matched_img = cv2.drawMatches(source_img, keypoints1, target_img, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    matched_img = cv2.resize(matched_img, (int(0.5*matched_img.shape[1]), int(0.5*matched_img.shape[0])))
    cv2.imshow('Matches', matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    if len(good_matches) > 0:
        #translation
        dx_sum = 0.0
        dy_sum = 0.0
        num_matches = len(good_matches)

        for match in good_matches:
            kp1 = keypoints1[match.queryIdx]
            kp2 = keypoints2[match.trainIdx]

            dx = kp2.pt[0] - kp1.pt[0]
            dy = kp2.pt[1] - kp1.pt[1]

            dx_sum += dx
            dy_sum += dy

        dx_avg = dx_sum / num_matches
        dy_avg = dy_sum / num_matches

        #transform
        transformation_matrix = np.array([[1, 0, dx_avg], [0, 1, dy_avg]], dtype=np.float32)
        registered_img = cv2.warpAffine(source_img, transformation_matrix, (target_img.shape[1], target_img.shape[0]))

        return registered_img, dx_avg, dy_avg
    else: 
        print("0 matches")
        #transform by values of previous frame
        transformation_matrix = np.array([[1, 0, prev_dx], [0, 1, prev_dy]], dtype=np.float32)
        registered_img = cv2.warpAffine(source_img, transformation_matrix, (target_img.shape[1], target_img.shape[0]))

        return registered_img, prev_dx, prev_dy

def main():
    folder_path = "D:\\data_gabby\\230201\\contrast_vid19"

    #add all image file paths to list
    img_file_paths = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and (file_name.endswith(".tiff") or file_name.endswith(".tif")):
            img_file_paths.append(file_path)

    #iterate through all images, stabilize, and save in new folder "stabilized"
    source_img_fp = img_file_paths[0]
    new_folder_path = os.path.join(folder_path, "stabilized")
    os.makedirs(new_folder_path, exist_ok=True)
    translations = []
    xval = yval = 0
    for x in range(1, 50): 
        registered_img, xval, yval = register_img(source_img_fp, img_file_paths[x], xval, yval)
        translations.append((xval, yval))
        dest_path = os.path.join(new_folder_path, os.path.basename(img_file_paths[x]))
        cv2.imwrite(dest_path, registered_img)   
        source_img_fp = img_file_paths[x] #update source img
    
    #save translations
    translations_file_path = os.path.join(new_folder_path, "translations.csv")
    with open(translations_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(translations)
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