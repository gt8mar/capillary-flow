import cv2
import numpy as np

MAX_SHIFT = 50  # Maximum shift in pixels

def register_images(reference_img, target_img, max_shift=MAX_SHIFT):
    # Convert images to grayscale
    gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized_ref = cv2.equalizeHist(gray_ref)
    equalized_target = cv2.equalizeHist(gray_target)

    # Use SIFT to detect keypoints and compute descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(equalized_ref, None)
    kp2, des2 = sift.detectAndCompute(equalized_target, None)

    # Use BFMatcher with default parameters
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Check if we have enough good matches
    if len(good_matches) >= 4:
        # Extract locations of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Compute translation offsets
        dxs = dst_pts[:, 0] - src_pts[:, 0]
        dys = dst_pts[:, 1] - src_pts[:, 1]

        # Use the median translation to minimize the effect of outliers
        dx = np.median(dxs)
        dy = np.median(dys)

        # Limit the shifts to prevent large translations
        dx = np.clip(dx, -max_shift, max_shift)
        dy = np.clip(dy, -max_shift, max_shift)

        # Create the translation matrix
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

        # Warp the target image using the translation matrix
        shifted_image = cv2.warpAffine(target_img, M, (reference_img.shape[1], reference_img.shape[0]))

        # Apply histogram equalization to the aligned image
        shifted_image_eq = cv2.equalizeHist(cv2.cvtColor(shifted_image, cv2.COLOR_BGR2GRAY))

        return (dx, dy), shifted_image_eq
    else:
        print("Warning: Not enough good matches for reliable registration.")
        # Return the target image converted to grayscale and equalized
        equalized_target_gray = cv2.equalizeHist(gray_target)
        return (0, 0), equalized_target_gray
