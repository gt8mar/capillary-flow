"""
Filename: align_segmented.py
-------------------------------------------------------------
This file aligns segmented images based on translations between moco images.
By: Gabby Rincon and Marcus Forst
"""

import os
import time
import platform
import numpy as np
import pandas as pd
import cv2
import csv
from skimage.color import rgb2gray
from skimage import io
from src.tools.parse_filename import parse_filename

PAD_VALUE = 250
MAX_TRANSLATION = 50  # Adjusted maximum translation to prevent excessive shifts

def register_images(reference_img, target_img, max_shift=MAX_TRANSLATION):
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

        # Limit the shifts to prevent excessive translations
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

def uncrop_segmented(path, input_seg_img):
    """
    Uncrops a segmented image based on shifts from a CSV file.

    Args:
        path (str): Path to the "video" folder containing the 'Results.csv' file with shift data.
        input_seg_img (numpy.ndarray): Input segmented image.

    Returns:
        numpy.ndarray: Uncropped segmented image.
    """
    shifts = pd.read_csv(os.path.join(path, 'metadata', 'Results.csv'))
    gap_left = shifts['x'].max()
    gap_right = shifts['x'].min()
    gap_bottom = shifts['y'].min()
    gap_top = shifts['y'].max()

    # Ensure that the gaps are non-negative
    if gap_left < 0:
        gap_left = 0
    if gap_top < 0:
        gap_top = 0
    if gap_right > 0:
        gap_right = 0
    if gap_bottom > 0:
        gap_bottom = 0

    # Convert the segmented image to grayscale
    input_seg_img = rgb2gray(input_seg_img)

    # Pad the image based on the calculated gaps
    uncropped_input_seg_img = np.pad(
        input_seg_img,
        ((abs(gap_top), abs(gap_bottom)), (abs(gap_left), abs(gap_right))),
        mode='constant',
        constant_values=0
    )
    return uncropped_input_seg_img

def align_segmented(path="f:\\Marcus\\data\\part30\\231130\\loc02"):
    """
    Aligns segmented images based on translations between moco images.

    Args:
        path (str): The path to the location directory for a given participant.

    Creates:
        Directories for registered moco images and registered segmented images.
        CSV files with translations, resize values, and crop values.
    """
    vid_folder_fp = os.path.join(path, "vids")
    segmented_folder_fp = os.path.join(path, "segmented", "hasty")

    # Create folder to save registered moco images
    reg_moco_folder = os.path.join(segmented_folder_fp, "moco_registered")
    os.makedirs(reg_moco_folder, exist_ok=True)

    # Make list of filepaths of vid 0 in moco folders of all vids
    moco_vids_fp = []
    sorted_vids_listdir = sorted(
        filter(
            lambda x: os.path.exists(os.path.join(vid_folder_fp, x)) and "vid" in x,
            os.listdir(vid_folder_fp)
        )
    )  # Sort numerically
    for vid in sorted_vids_listdir:
        vid_path = os.path.join(vid_folder_fp, vid)
        if os.path.exists(os.path.join(vid_folder_fp, vid, "mocoslice")):
            moco_folder_fp = os.path.join(vid_folder_fp, vid, "mocoslice")
        elif os.path.exists(os.path.join(vid_folder_fp, vid, "mocosplit")):
            moco_folder_fp = os.path.join(vid_folder_fp, vid, "mocosplit")
        else:
            moco_folder_fp = os.path.join(vid_folder_fp, vid, "moco")
        sorted_moco_ld = sorted(
            filter(
                lambda x: os.path.exists(os.path.join(moco_folder_fp, x)) and x.endswith(".tif"),
                os.listdir(moco_folder_fp)
            )
        )
        if sorted_moco_ld:
            # Append the video name and the first moco image path
            moco_vids_fp.append((vid, os.path.join(moco_folder_fp, sorted_moco_ld[0])))
        else:
            print(f"Warning: No moco images found in {moco_folder_fp}")

    if not moco_vids_fp:
        print("No moco images found. Exiting.")
        return

    # Set reference image
    reference_moco_tuple = moco_vids_fp[0]
    first_video = reference_moco_tuple[0]
    first_video_path = os.path.join(vid_folder_fp, first_video)
    reference_moco_fp = reference_moco_tuple[1]
    reference_moco_img = cv2.imread(reference_moco_fp)
    reference_moco_filename = f'{first_video}_moco_0000.tif'

    # Save reference moco image with contrast adjustment
    contrast_reference_moco_img = cv2.equalizeHist(cv2.cvtColor(reference_moco_img, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(
        os.path.join(reg_moco_folder, reference_moco_filename),
        np.pad(contrast_reference_moco_img, ((PAD_VALUE, PAD_VALUE), (PAD_VALUE, PAD_VALUE)))
    )

    # Create folder to save registered segmented images
    reg_folder_path = os.path.join(segmented_folder_fp, "registered")
    os.makedirs(reg_folder_path, exist_ok=True)

    translations = []
    crops = []

    # Process the first segmented frame
    sorted_seg_listdir = sorted(
        filter(
            lambda x: os.path.isfile(os.path.join(segmented_folder_fp, x)) and x.endswith(".png"),
            os.listdir(segmented_folder_fp)
        )
    )  # Sort numerically
    if not sorted_seg_listdir:
        print("No segmented images found. Exiting.")
        return

    first_seg_fp = os.path.join(segmented_folder_fp, sorted_seg_listdir[0])
    first_seg_img = cv2.imread(first_seg_fp)

    # Use the shifts from the Results.csv file to uncrop the first segmented image we will register to
    first_seg_img = uncrop_segmented(first_video_path, first_seg_img)

    # Initialize translations list with zero shift for the reference image
    translations.append([0, 0])

    # Process remaining frames
    for i in range(len(sorted_seg_listdir)):
        if i == 0:
            # Reference image already processed
            continue

        # Register moco images
        input_moco_tuple = moco_vids_fp[i]
        input_moco_fp = input_moco_tuple[1]
        video = input_moco_tuple[0]
        input_moco_img = cv2.imread(input_moco_fp)
        input_moco_filename = f'{video}_moco_0000.tif'

        # Register directly to the initial reference image
        (dx, dy), registered_image = register_images(
            reference_moco_img,
            input_moco_img,
            max_shift=MAX_TRANSLATION
        )

        translations.append([dx, dy])

        # Save registered moco frame
        cv2.imwrite(os.path.join(reg_moco_folder, input_moco_filename), registered_image)

    # Calculate the overall maximum dimensions after padding
    dxs, dys = zip(*translations)
    minx, maxx = min(dxs), max(dxs)
    miny, maxy = min(dys), max(dys)

    # Determine the target output size for all images
    # Assuming all input_seg_img have the same dimensions
    input_seg_fp = os.path.join(segmented_folder_fp, sorted_seg_listdir[0])
    input_seg_img = cv2.imread(input_seg_fp, cv2.IMREAD_GRAYSCALE)
    base_height, base_width = input_seg_img.shape

    # Calculate the maximum required padding
    total_pad_top = int(np.ceil(maxy - miny))
    total_pad_bottom = total_pad_top
    total_pad_left = int(np.ceil(maxx - minx))
    total_pad_right = total_pad_left

    target_height = base_height + total_pad_top + total_pad_bottom
    target_width = base_width + total_pad_left + total_pad_right

    resize_vals = []

    for x in range(len(sorted_seg_listdir)):
        participant, date, location, seg_video, __ = parse_filename(sorted_seg_listdir[x])
        seg_video_filepath = os.path.join(vid_folder_fp, seg_video)

        # Get image to segment
        input_seg_fp = os.path.join(segmented_folder_fp, sorted_seg_listdir[x])
        input_seg_img = cv2.imread(input_seg_fp)

        # Make segmented image same size using Results.csv file from video folder
        input_seg_img = uncrop_segmented(seg_video_filepath, input_seg_img)
        # Note: The crops list is not used in this version, but you can store crop values if needed
        crops.append((0, 0, 0, 0))  # Placeholder for compatibility

        dx, dy = translations[x]
        pad_top = int(np.ceil(maxy - dy))
        pad_bottom = int(np.ceil(dy - miny))
        pad_left = int(np.ceil(maxx - dx))
        pad_right = int(np.ceil(dx - minx))

        # Ensure padding amounts are non-negative
        pad_top = max(pad_top, 0)
        pad_bottom = max(pad_bottom, 0)
        pad_left = max(pad_left, 0)
        pad_right = max(pad_right, 0)

        # Pad the image
        registered_seg_img = np.pad(
            input_seg_img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )

        # After padding, crop or pad to the target size to ensure consistency
        current_height, current_width = registered_seg_img.shape
        if current_height != target_height or current_width != target_width:
            # Adjust the image size
            registered_seg_img = cv2.resize(
                registered_seg_img,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST
            )

        # Save the registered segmented image
        registered_seg_img = (registered_seg_img * 255).astype(np.uint8)
        io.imsave(os.path.join(reg_folder_path, os.path.basename(input_seg_fp)), registered_seg_img)
        
    # Save translations to CSV file
    translations_csv_fp = os.path.join(segmented_folder_fp, "translations.csv")
    with open(translations_csv_fp, 'w', newline='') as translations_csv_file:
        writer = csv.writer(translations_csv_file)
        writer.writerows(translations)

    # Save resize values to CSV file
    resize_csv_fp = os.path.join(segmented_folder_fp, "resize_vals.csv")
    with open(resize_csv_fp, 'w', newline='') as resize_csv_file:
        writer = csv.writer(resize_csv_file)
        writer.writerows(resize_vals)

    # Save crop values to CSV file
    crops_csv_fp = os.path.join(segmented_folder_fp, "crop_values.csv")
    with open(crops_csv_fp, 'w', newline='') as crops_csv_file:
        writer = csv.writer(crops_csv_file)
        writer.writerows(crops)

    # # Optional: Visualize translations
    # import matplotlib.pyplot as plt

    # translations_array = np.array(translations)
    # plt.figure(figsize=(10, 5))
    # plt.plot(translations_array[:, 0], label='dx')
    # plt.plot(translations_array[:, 1], label='dy')
    # plt.title('Translations Over Videos')
    # plt.xlabel('Video Index')
    # plt.ylabel('Translation (pixels)')
    # plt.legend()
    # plt.show()

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    align_segmented()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
