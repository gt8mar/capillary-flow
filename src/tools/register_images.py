"""
Filename: register_image.py
By: Gabby Rincon & ChatGPT
"""

import os
import time
import numpy as np
import cv2

"""
This function takes 2 images (reference and target) and returns the target translated to the reference as well as the x and y translation values.
"""
def register_images(reference_img, target_img):
    #grayscale
    equalized_reference_img = cv2.equalizeHist(cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY))
    equalized_target_img = cv2.equalizeHist(cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY))

    #use SIFT to extract keypoints & descriptors
    detector = cv2.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(equalized_reference_img, None)
    keypoints2, descriptors2 = detector.detectAndCompute(equalized_target_img, None)

    #match keypoints & descriptors between images
    matcher = cv2.BFMatcher()
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    #filter for good matches
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    #remove outliers by their slopes when drawing matches between images side by side
    slopes = []
    for match in good_matches:
        m = match
        slope = (keypoints2[m.trainIdx].pt[1] - keypoints1[m.queryIdx].pt[1]) / (keypoints2[m.trainIdx].pt[0] - keypoints1[m.queryIdx].pt[0])
        slopes.append(slope)
    median_slope = np.median(slopes)
    mad = np.median(np.abs(slopes - median_slope))
    filtered_matches = []
    outlier_threshold = 1.2 * mad  #free paramenter
    for match, slope in zip(good_matches, slopes):
        if np.abs(slope - median_slope) < outlier_threshold:
            filtered_matches.append(match)

    #draw matches
    """matched_img = cv2.drawMatches(reference_img, keypoints1, target_img, keypoints2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matched_img = cv2.resize(matched_img, (int(0.5 * matched_img.shape[1]), int(0.5 * matched_img.shape[0])))
    cv2.imshow('Matches', matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    #if no matches found, return (0,0) & untranslated image
    if len(good_matches) == 0:
        print("not translated")
        return (0, 0), equalized_target_img

    #average translations across all matches
    dx_sum = dy_sum = 0.0
    for match in good_matches:
        kp1 = keypoints1[match.queryIdx]
        kp2 = keypoints2[match.trainIdx]

        dx = kp2.pt[0] - kp1.pt[0]
        dy = kp2.pt[1] - kp1.pt[1]

        dx_sum += dx
        dy_sum += dy
    dx_avg = dx_sum / len(good_matches)
    dy_avg = dy_sum / len(good_matches)

    #transform
    transformation_matrix = np.array([[1, 0, -dx_avg], [0, 1, -dy_avg]], dtype=np.float32)
    shifted_image = cv2.warpAffine(target_img, transformation_matrix, (reference_img.shape[1], reference_img.shape[0]))

    return (dx_avg, dy_avg), shifted_image

def main():
    reference_img = cv2.imread("E:\\Marcus\\gabby test data\\part14\\230428\\vid06\\moco\\vid06_moco_0000.tif")
    target_img = cv2.imread("E:\\Marcus\\gabby test data\\part14\\230428\\vid07\\moco\\vid07_moco_0000.tif")

    translation, shifted_image = register_images(reference_img, target_img)

    print("Translation (x, y):", translation)

    #TEMP for visual confirmation
    output_dir = "E:\\Marcus\\gabby test data\\test"
    cv2.imwrite(os.path.join(output_dir, "target_img.tif"), target_img)
    cv2.imwrite(os.path.join(output_dir, "reference_img.tif"), reference_img)
    cv2.imwrite(os.path.join(output_dir, "translated.tif"), shifted_image)



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