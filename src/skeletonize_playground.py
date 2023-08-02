"""
Filename: skeletonize_playground.py
------------------------------------------------------ 
Skeletonizes an image and finds the centerline coordinates.

By: Marcus Forst
"""

import cv2
import numpy as np

def skeletonize(image):
    # Source: https://stackoverflow.com/questions/38025838/opencv-python-thinning-algorithm
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)
    
    ret, image = cv2.threshold(image, 128, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()
        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            done = True
    return skel

def find_centerline_coordinates(skeleton):
    coordinates = np.column_stack(np.where(skeleton > 0))
    centerline_coordinates = coordinates[np.argsort(coordinates[:, 1])]
    return centerline_coordinates

def main():
    # Replace 'your_image_path' with the path to your image file
    image_path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\segmented\\set_01_part15_230428_vid13_background_seg.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to convert it into a binary representation
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Skeletonize the binary image
    skeleton = skeletonize(binary_image)

    # Find the centerline coordinates from the skeletonized image
    centerline_coordinates = find_centerline_coordinates(skeleton)

    # Plot the skeletonized image
    cv2.imshow('skeleton', skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the centerline coordinates
    print(centerline_coordinates)

if __name__ == "__main__":
    main()