"""
Filename: register_images.py
By: Gabby Rincon & ChatGPT
"""

import os
import time
import numpy as np
import cv2
from moco_py import MotionCorrector

# Define a maximum shift threshold (adjust as necessary)
MAX_SHIFT = 50  # Maximum shift in pixels

def register_images_moco(reference_img, target_img, max_shift=MAX_SHIFT):
    """Register target image to reference image using moco-py.

    This function uses the MotionCorrector from moco-py to align the target
    image with the reference image.

    Args:
        reference_img (np.ndarray): The reference image to align to.
        target_img (np.ndarray): The target image to be aligned.
        max_shift (int, optional): Maximum allowed shift in pixels. Defaults to MAX_SHIFT.

    Returns:
        tuple: A tuple containing:
            - shift (tuple): The (x, y) shift applied to align the target image.
            - contrast_corrected_image (np.ndarray): The aligned and contrast-enhanced target image.

    Raises:
        ValueError: If the input images are not 2D (grayscale) or 3D (color) arrays.
    """
    # Initialize MotionCorrector
    corrector = MotionCorrector(max_shift=max_shift, crop_edges=False)

    # Convert images to grayscale if they're not already
    if len(reference_img.shape) == 3:
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    if len(target_img.shape) == 3:
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # Create a stack with the target image
    image_stack = np.array([target_img])

    # Perform motion correction
    corrected_stack, shifts = corrector.correct_stack(image_stack, reference_img)

    # Get the shift and corrected image
    shift = shifts[0]  # There's only one image in the stack
    corrected_image = corrected_stack[0]

    # Apply histogram equalization for contrast enhancement
    contrast_corrected_image = cv2.equalizeHist(corrected_image)

    return shift, contrast_corrected_image

"""
This function takes 2 images as np arrays (reference and target) and returns the x and y translation values and the target translated to the reference.
"""
def register_images(reference_img, target_img, prevdx=0, prevdy=0, max_shift=MAX_SHIFT, pin = False):
    # Grayscale
    equalized_reference_img = cv2.equalizeHist(cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY))
    equalized_target_img = cv2.equalizeHist(cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY))

    # Use SIFT to extract keypoints & descriptors
    detector = cv2.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(equalized_reference_img, None)
    keypoints2, descriptors2 = detector.detectAndCompute(equalized_target_img, None)

    # Match keypoints & descriptors between images
    matcher = cv2.BFMatcher()
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Filter for good matches
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    # Remove outliers by their slopes
    slopes = []
    for match in good_matches:
        m = match
        slope = (keypoints2[m.trainIdx].pt[1] - keypoints1[m.queryIdx].pt[1]) / (keypoints2[m.trainIdx].pt[0] - keypoints1[m.queryIdx].pt[0])
        slopes.append(slope)
    median_slope = np.median(slopes)
    mad = np.median(np.abs(slopes - median_slope))
    filtered_matches = []
    outlier_threshold = 1.2 * mad  # Free parameter
    for match, slope in zip(good_matches, slopes):
        if np.abs(slope - median_slope) < outlier_threshold:
            filtered_matches.append(match)

    # If no matches found, return (0,0) & untranslated image
    if len(good_matches) <= 1:
        transformation_matrix = np.array([[1, 0, -(prevdx)], [0, 1, -(prevdy)]], dtype=np.float32)
        target_img = np.pad(target_img, ((250, 250), (250, 250), (0, 0)))
        shifted_image = cv2.warpAffine(target_img, transformation_matrix, (reference_img.shape[1] + 500, reference_img.shape[0] + 500))
        contrast_shifted_image = cv2.equalizeHist(cv2.cvtColor(shifted_image, cv2.COLOR_BGR2GRAY))
        return (0, 0), contrast_shifted_image
    if pin:
        transformation_matrix = np.array([[1, 0, -(prevdx)], [0, 1, -(prevdy)]], dtype=np.float32)
        target_img = np.pad(target_img, ((250, 250), (250, 250), (0, 0)))
        shifted_image = cv2.warpAffine(target_img, transformation_matrix, (reference_img.shape[1] + 500, reference_img.shape[0] + 500))
        contrast_shifted_image = cv2.equalizeHist(cv2.cvtColor(shifted_image, cv2.COLOR_BGR2GRAY))
        return (0, 0), contrast_shifted_image



    # Average translations across all matches
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

    # # Apply maximum shift limit
    # dx_avg = np.clip(dx_avg, -max_shift, max_shift)
    # dy_avg = np.clip(dy_avg, -max_shift, max_shift)

    # Transform
    transformation_matrix = np.array([[1, 0, -(dx_avg + prevdx)], [0, 1, -(dy_avg + prevdy)]], dtype=np.float32)
    target_img = np.pad(target_img, ((250, 250), (250, 250), (0, 0)))
    shifted_image = cv2.warpAffine(target_img, transformation_matrix, (reference_img.shape[1] + 500, reference_img.shape[0] + 500))
    contrast_shifted_image = cv2.equalizeHist(cv2.cvtColor(shifted_image, cv2.COLOR_BGR2GRAY))

    return (dx_avg, dy_avg), contrast_shifted_image

def main():
    reference_img = cv2.imread("E:\\Marcus\\gabby test data\\part14\\230428\\vid06\\moco\\vid06_moco_0000.tif")
    target_img = cv2.imread("E:\\Marcus\\gabby test data\\part14\\230428\\vid07\\moco\\vid07_moco_0000.tif")

    translation, shifted_image = register_images_moco(reference_img, target_img)

    print("Translation (x, y):", translation)

    # TEMP for visual confirmation
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
