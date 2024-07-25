"""
Filename: frog_register.py
------------------------------------------------------
This file names the segements in frog capillaries.

By: Juliette Levy
"""
import cv2
import os
import numpy as np

def main():
    # Load images
    image1 = cv2.imread('E:\\frog\\24-2-14 WkSl\\Frog4\\Right\\segmented\\SD_24-2-14_WkSlAwakeFrog4Rankle1.png')
    image2 = cv2.imread('E:\\frog\\24-2-14 WkSl\\Frog4\\Right\\segmented\\SD_24-2-14_WkSlSleepFrog4RankleCl630.png')
    image3 = cv2.imread('E:\\frog\\24-2-14 WkSl\\Frog4\\Right\\segmented\\SD_24-2-14WkSlExaustedFrog4Rankle1.png')

    # Align images to the first image
    aligned_image = register_images(image1, image2) #using the first image as reference point to allign third image
    overlayed_image = overlay_images(image1, aligned_image, alpha = 0.5)

    # Save the result
    output_path = 'E:\\frog\\aligned_overlayed_image.png'
    # save_image(output_path, overlayed_image)
    cv2.imwrite(output_path, overlayed_image)
    overlayed_image = cv2.imread('E:\\frog\\aligned_overlayed_image.png')

    final_aligned_image = register_images2(overlayed_image, image3) #using the first image as reference point to allign third image
    final_overlayed_image = overlay_images2(overlayed_image, final_aligned_image, alpha = 0.5)

    final_output_path = 'E:\\frog\\final_aligned_overlayed_image.png'
    # save_image(output_path, overlayed_image)
    cv2.imwrite(final_output_path, final_overlayed_image)

def register_images(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors. (Oriented FAST and Rotated BRIEF)
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None) #parameters- 1st: image to be processed, 2nd: masked area for consideration
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Match descriptors using the BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    # matches2 = sorted(matches2, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
 
    # Find homography
    matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    # Use the homography matrix to warp the second image
    height, width, channels = image1.shape
    aligned_image = cv2.warpPerspective(image2, matrix, (width, height))

    return aligned_image

def register_images2(overlayed_imaged, image3):
    # Convert images to grayscale
    final_gray1 = cv2.cvtColor(overlayed_imaged, cv2.COLOR_BGR2GRAY)
    final_gray2 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors. (Oriented FAST and Rotated BRIEF)
    orb = cv2.ORB_create()
    final_keypoints1, final_descriptors1 = orb.detectAndCompute(final_gray1, None) #parameters- 1st: image to be processed, 2nd: masked area for consideration
    final_keypoints2, final_descriptors2 = orb.detectAndCompute(final_gray2, None)

    # Match descriptors using the BFMatcher
    final_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    final_matches = final_bf.match(final_descriptors1, final_descriptors2)

    # Sort matches by distance
    final_matches = sorted(final_matches, key=lambda x: x.distance)
    # matches2 = sorted(matches2, key=lambda x: x.distance)

    # Extract location of good matches
    final_points1 = np.float32([final_keypoints1[m.queryIdx].pt for m in final_matches]).reshape(-1, 1, 2)
    final_points2 = np.float32([final_keypoints2[m.trainIdx].pt for m in final_matches]).reshape(-1, 1, 2)
 
    # Find homography
    final_matrix, mask = cv2.findHomography(final_points2, final_points1, cv2.RANSAC, 5.0)

    # Use the homography matrix to warp the second image
    height, width, channels = overlayed_imaged.shape
    final_aligned_image = cv2.warpPerspective(image3, final_matrix, (width, height))

    return final_aligned_image

def overlay_images(image1, image2, alpha=0.5):
    # blended_image = cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)
    # blended_image = cv2.addWeighted(blended_image, 1 - alpha, image3, alpha, 0)

    #Convert the overlay image to grayscale
    gray_overlay = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Create a mask where the overlay image is not black
    mask = gray_overlay > 0
    
    # Create an output image
    output = image1.copy()
    
    # Blend images using the mask
    output[mask] = cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)[mask]

    return output

def overlay_images2(overlayed_image, image3, alpha=0.5):
    # blended_image = cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)
    # blended_image = cv2.addWeighted(blended_image, 1 - alpha, image3, alpha, 0)

    #Convert the overlay image to grayscale
    gray_overlay = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

    # Create a mask where the overlay image is not black
    mask = gray_overlay > 0
    
    # Create an output image
    output = overlayed_image.copy()
    
    # Blend images using the mask
    output[mask] = cv2.addWeighted(overlayed_image, 1 - alpha, image3, alpha, 0)[mask]

    return output


if __name__ == "__main__":
    main()