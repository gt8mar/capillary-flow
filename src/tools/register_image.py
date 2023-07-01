"""
Filename: register_image.py
-------------------------------------------------------------
This file registers the first frame of one video to that of another video.
by: Gabby Rincon & ChatGPT
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import shutil


def register_img(reference_img, input_img, prev_dx=0, prev_dy=0):
    #find keypoints & descriptors
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(reference_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(input_img, None)

    #match descriptors
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    #filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance: #free parameter
            good_matches.append(m)

    """# Draw the matches between the reference and input images
    matched_img = cv2.drawMatches(reference_img, keypoints1, input_img, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

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
        registered_img = cv2.warpAffine(input_img, transformation_matrix, (reference_img.shape[1], reference_img.shape[0]))

        return registered_img, dx_avg, dy_avg
    else: 
        print("0 matches")
        #transform by values of previous frame
        transformation_matrix = np.array([[1, 0, prev_dx], [0, 1, prev_dy]], dtype=np.float32)
        registered_img = cv2.warpAffine(input_img, transformation_matrix, (reference_img.shape[1], reference_img.shape[0]))

        return registered_img, prev_dx, prev_dy

# This function registers the first frame of input video to first frame of reference video.
# It saves a new folder in the input folder containing:
#   reference image, input image, registered input image, translations.csv
# If feature detection does not work, it does not translate the image and saves 0 as translation values.
# Args: 
#   folder_path_reference: file path for reference video
#   folder_path_inputt: file path for input video
# Returns:
#   xval: translation value in x
#   yval: translation value in y
def register(folder_path_reference, folder_path_input):
    """folder_path_reference = "D:\\data_gabby\\230201\\vid27\\moco"
    folder_path_input = "D:\\data_gabby\\230201\\vid28\\moco"
    """
    #get reference image (first tiff file in folder)
    file_path_reference = ""
    for file_name in os.listdir(folder_path_reference):
        file_path = os.path.join(folder_path_reference, file_name)
        if os.path.isfile(file_path) and (file_name.endswith(".tiff") or file_name.endswith(".tif")):
            file_path_reference = file_path
            break
    #get input image (first tiff file in folder)
    file_path_input = ""
    for file_name in os.listdir(folder_path_input):
        file_path = os.path.join(folder_path_input, file_name)
        if os.path.isfile(file_path) and (file_name.endswith(".tiff") or file_name.endswith(".tif")):
            file_path_input = file_path
            break
    
    #register
    registered_img, xval, yval = register_img(cv2.imread(file_path_reference), cv2.imread(file_path_input))

    #save reference, input, registered img in new folder "registered"
    #make new registered folder
    new_folder_path = os.path.join(os.path.split(folder_path_input)[0], "registered")
    os.makedirs(new_folder_path, exist_ok=True)
    print(new_folder_path)
    #save reference img
    shutil.copy2(file_path_reference, os.path.join(new_folder_path, os.path.basename(file_path_reference)))
    #save input img
    shutil.copy2(file_path_input, os.path.join(new_folder_path, os.path.basename(file_path_input)))
    #save registered img
    cv2.imwrite(os.path.join(new_folder_path, "reg_" + os.path.basename(file_path_input)), registered_img)

    #save translations in new folder "registered"
    translations = []
    translations.append((xval, yval))
    translations_file_path = os.path.join(new_folder_path, "translations.csv")
    with open(translations_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(translations)

    return xval, yval

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    print("Registering videos...")
    register()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))