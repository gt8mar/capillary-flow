"""
Filename: name_capillaries.py
--------------------------------------------------------------
This file names capillaries based on the order they were segmented. 
It sets up a later process to rename the capillaries with their 
actual names.

By: Marcus Forst
"""

import os, time, platform
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from src.tools.parse_filename import parse_filename
from skimage.color import rgb2gray
from scipy.ndimage import label



def uncrop_segmented(video_path, input_seg_img):
    """
    Uncrops a segmented image based on shifts from a CSV file.

    Args:
        path (str): Path to the "video" folder containing the 'Results.csv' file with shift data.
        input_seg_img (numpy.ndarray): Input segmented image.

    Returns:
        tuple: Uncropped image and the left, right, bottom, and top gaps.
    """
    shifts = pd.read_csv(os.path.join(video_path, 'metadata', 'Results.csv'))
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

    # check if the segmented image is grayscale
    if len(input_seg_img.shape) == 3:    
        # Convert the segmented image to grayscale
        input_seg_img = rgb2gray(input_seg_img)

    # Pad the image based on the calculated gaps
    uncropped_input_seg_img = np.pad(input_seg_img, ((abs(gap_top), abs(gap_bottom)), (abs(gap_left), abs(gap_right))), mode='constant', constant_values=0)
    return uncropped_input_seg_img, gap_left, gap_right, gap_bottom, gap_top

def create_capillary_masks(binary_mask):
    """
    Creates separate binary masks for each capillary in the original binary mask.

    Parameters:
    binary_mask (numpy array): Original binary mask with multiple capillaries indicated by pixel values > 0.

    Returns:
    list of numpy arrays: A list where each element is a binary mask showing only one capillary.
    """
    # Label each connected component in the binary mask
    labeled_mask, num_features = label(binary_mask > 0)
    
    # List to store the individual capillary masks
    capillary_masks = []

    # Iterate over each unique capillary label
    for i in range(1, num_features + 1):
        # Create a new binary mask for the current capillary
        capillary_mask = (labeled_mask == i).astype(np.uint8)
        
        # Append to list of capillary masks
        capillary_masks.append(capillary_mask)
    
    return capillary_masks

def create_overlay_with_label(frame_img, cap_mask, color, label):
    """
    Create an overlay of a capillary mask on a frame image with a label.

    Args:
        frame_img (numpy.ndarray): The original frame image in BGR format.
        cap_mask (numpy.ndarray): The capillary mask (2D binary mask).
        color (tuple): The color to use for the capillary overlay (BGR).
        label (str): The label to add to the overlay.
    """
    # Get dimensions of the mask
    height, width = cap_mask.shape
    
    # Create colored mask
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    colored_mask[cap_mask > 0] = color
    
    # Create alpha channel
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[cap_mask > 0] = 128  # 50% transparency
    
    # Convert to BGRA
    frame_bgra = cv2.cvtColor(frame_img, cv2.COLOR_BGR2BGRA)
    overlay_bgra = np.dstack((colored_mask, alpha))
    
    # Blend images
    result = frame_bgra.copy()
    mask_region = (overlay_bgra[:, :, 3] > 0)
    result[mask_region] = cv2.addWeighted(
        frame_bgra[mask_region],
        0.5,
        overlay_bgra[mask_region],
        0.5,
        0
    )
    
    # Find centroid of the mask for label placement
    moments = cv2.moments(cap_mask)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = width // 2, height // 2
    
    # Add label
    cv2.putText(result, label, (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(result, label, (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
    
    return cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)

def main(location_path):
    """
    have: segmented images (cropped), videos (uncropped)

    pre
    1. Get the segmented images
    2. Get the backgrounds
    3. Get the video metadata
    4. Uncrop the segmented images

    main
    1. use contours to find the capillaries in the segmented images
    2. name the capillaries based on the order contours found them
    3. save the names in a csv file
    4. save each individual capillary in a folder
    5. save an overlay of the capillary segmentations on the video stack average with their names (colored)
    
    """
    # Get the segmented images
    segmented_folder = os.path.join(location_path, 'segmented', 'hasty')
    background_folder = os.path.join(location_path, 'backgrounds')
    # segmented images end in .png
    segmented_images_list = [f for f in os.listdir(segmented_folder) if f.endswith('_seg.png')]
    # make list where the key is the video number and the value is the image name:
    cap_names = pd.DataFrame(columns=['File Name', 'Capillary Name'])
    for image_name in segmented_images_list:
        participant, date, location, video, file_prefix = parse_filename(image_name)
        background_name = image_name.replace('_seg.png', '_background.tiff')
        video_number = video[3:]
        segmented_image = cv2.imread(os.path.join(segmented_folder, image_name), cv2.IMREAD_GRAYSCALE)
        background_image = cv2.imread(os.path.join(background_folder, background_name), cv2.IMREAD_GRAYSCALE)
        background_image_color = background_image

        # set values in segmented image to 0 or 255
        segmented_image[segmented_image > 0 ] = 255
        segmented_image_uncropped, gap_left, gap_right, gap_bottom, gap_top = uncrop_segmented(os.path.join(location_path, 'vids', f'vid{video_number}'), segmented_image)
        background_image_uncropped, gap_left, gap_right, gap_bottom, gap_top = uncrop_segmented(os.path.join(location_path, 'vids', f'vid{video_number}'), background_image)

        # Initialize the background image once
        background_image_bgr = cv2.cvtColor(background_image_uncropped, cv2.COLOR_GRAY2BGR)
        # Convert to BGRA for alpha blending
        result = cv2.cvtColor(background_image_bgr, cv2.COLOR_BGR2BGRA)

        # find capillaries using 
        capillaries = create_capillary_masks(segmented_image_uncropped)
        colors = plt.cm.get_cmap('tab20', len(capillaries))

        for i in range(len(capillaries)):
            capillary_mask = (capillaries[i] * 255).astype(np.uint8)  # Scale to 0-255
            capillary_name = str(i).zfill(2)
            capillary_filename = image_name.replace('_seg.png', f'_seg_cap_{capillary_name}.png')
            cap_names = cap_names.append({'File Name': capillary_filename, 'Capillary Name': ''}, ignore_index=True)
            # make folders if they don't exist
            os.makedirs(os.path.join(location_path, 'segmented','hasty','individual_caps_original'), exist_ok=True)
            # save the capillary image
            cv2.imwrite(os.path.join(location_path, 'segmented','hasty', 'individual_caps_original', capillary_filename), capillary_mask)
                
            # Convert the color to BGR format and scale to 0-255
            color = (np.array(colors(i)[:3]) * 255).astype(np.uint8)
            # Create overlay for this capillary
            result = create_overlay_with_label(cv2.cvtColor(result, cv2.COLOR_BGRA2BGR), capillary_mask, color, capillary_name)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)  # Convert back to BGRA for next iteration

        # Final conversion to BGR for saving
        final_result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)

        # Make the folders if they don't exist
        os.makedirs(os.path.join(location_path, 'segmented','hasty','overlays'), exist_ok=True)
        os.makedirs(os.path.join('/hpc/projects/capillary-flow/results/size/overlays'), exist_ok=True)
        cv2.imwrite(os.path.join(location_path, 'segmented', 'hasty', 'overlays', image_name.replace('_seg.png', '_overlay.png')), final_result)
        cv2.imwrite(os.path.join(f'/hpc/projects/capillary-flow/results/size/overlays', image_name.replace('_seg.png', '_overlay.png')), final_result)

            
    

    
        
    # save the capillary names
    cap_names_filename = f'{file_prefix}_cap_names.csv'
    cap_names_filename = cap_names_filename.replace('set01_', '').replace('set_01', '')
    # make folders if they don't exist
    os.makedirs(os.path.join(location_path, 'segmented','hasty','individual_caps_original'), exist_ok=True)
    os.makedirs(os.path.join('/hpc/projects/capillary-flow/results/size/name_csvs'), exist_ok=True)
    cap_names.to_csv(os.path.join(location_path, 'segmented', 'hasty','individual_caps_original', cap_names_filename), index=False)
    cap_names.to_csv(os.path.join('/hpc/projects/capillary-flow/results/size/name_csvs', cap_names_filename), index=False)   

    return 0
