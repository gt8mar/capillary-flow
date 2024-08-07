"""
Filename: naming_overlay.py
----------------------------
This module contains a function that creates overlay images by combining frames with segmented caps.

By: Gabriela Rincon
Updated by: Marcus Forst
"""

import time
import os
import re
import csv
import cv2
import numpy as np
from skimage.color import rgb2gray
import platform
import pandas as pd

PAD_VALUE = 250  # Padding value to ensure images are large enough

def get_label_position(input_array):
    """
    Calculate the position for a label based on non-zero elements in the array.

    Args:
        input_array (np.ndarray): The input array to process.

    Returns:
        tuple: Coordinates (x, y) for the label position.
    """
    # Find the indices of non-zero elements in the array
    non_zero_indices = np.argwhere(input_array != 0)
    
    # Calculate the minimum and maximum x and y values of non-zero elements
    min_x = np.min(non_zero_indices[:, 1])
    max_x = np.max(non_zero_indices[:, 1])
    min_y = np.min(non_zero_indices[:, 0])
    max_y = np.max(non_zero_indices[:, 0])
    
    # Define the edge margin
    edge_margin = 30
    
    # Check if minimum and maximum coordinates are within bounds
    if min_x > edge_margin:
        x_coord = min_x
    else:
        x_coord = max_x
    
    if min_y > edge_margin:
        y_coord = min_y
    else:
        y_coord = max_y
    
    return x_coord, y_coord

def rename_files(directory_path):
    """
    Rename files in a directory to ensure they have two-digit video numbers.

    Args:
        directory_path (str): The path to the directory containing the files.
    """
    # Get a list of files in the directory
    file_list = os.listdir(directory_path)

    # Iterate through each file in the directory
    for filename in file_list:
        # Search for a number in the filename
        match = re.search(r'vid(\d+)', filename)
        
        # Check if a match was found and the number is 1 or 2 digits
        if match and len(match.group(1)) <= 2:
            num = int(match.group(1))
            new_filename = filename.replace(match.group(0), f'vid{num:02}')

            # Construct the full paths for the old and new filenames
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)

            # Rename the file
            os.rename(old_path, new_path)

def extract_file_info(filename):
    """
    Extract date, location, and video information from a filename.

    Args:
        filename (str): The filename to extract information from.

    Returns:
        tuple: Extracted date, location, and video information.
    """
    set_part_date = filename[:20]  # with trailing underscore
    lmatch = re.search(r'loc(\d{2})', filename)
    location = "" if lmatch is None else "loc" + lmatch.group(1) + "_"
    vmatch = re.search(r'vid(\d{2})', filename)
    vid = "" if vmatch is None else "vid" + vmatch.group(1) + "_"
    return set_part_date, location, vid

def make_overlays(path, rename = False):
    """
    Create overlay images by combining frames with segmented caps.

    Args:
        path (str): The path to the directory containing the data.
    """
    # Set file paths
    reg_moco_fp = os.path.join(path, "segmented", "hasty", "moco_registered")

    # Read resize values from CSV file
    resize_csv = os.path.join(path, "segmented", "hasty", "resize_vals.csv")
    resize_df = pd.read_csv(resize_csv, header=None)
    minx = int(resize_df.iloc[0, 0])
    maxx = int(resize_df.iloc[0, 1])
    miny = int(resize_df.iloc[0, 2])
    maxy = int(resize_df.iloc[0, 3])

    # Define predefined colors for elements
    predefined_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Lime
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (255, 140, 0),  # Dark Orange
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Light Teal
        (139, 69, 19),  # Saddle Brown
        (255, 69, 0),   # Red-Orange
        (128, 0, 0),    # Maroon
    ]
    element_colors = {}
    colored_elements = []

    """# Rename files in the directory
    if rename:
        rename_files(reg_moco_fp)"""

    # Process each frame in the registered motion correction folder
    for frame in os.listdir(reg_moco_fp):
        vmatch = re.search(r'vid(\d{2})', frame)
        vidnum = vmatch.group(1)
        frame_img = cv2.imread(os.path.join(reg_moco_fp, frame))
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2BGRA)
        frame_img[:, :, 3] = 255
        
        # Get all caps in vid
        cap_dir = os.listdir(os.path.join(path, "segmented", "hasty", "individual_caps_translated"))
        if rename:
            cap_dir = os.listdir(os.path.join(path, "segmented", "hasty", "renamed_individual_caps_translated"))
        for cap in cap_dir:
            if "vid" + vidnum in cap: 
                cmatch = re.search(r'cap_(\d{2})', cap)
                capnum = "cap_" + cmatch.group(1) 

                # Match to previous color if used, else pop from predefined colors
                if capnum in element_colors:
                    color = element_colors[capnum]
                else:
                    if len(predefined_colors) > 0:
                        color = predefined_colors.pop(0)
                    else:
                        color = (255, 255, 255) 
                    element_colors[capnum] = color
                colored_elements.append((capnum, color))

                cap_img = cv2.imread(os.path.join(path, "segmented", "hasty", "individual_caps_translated", cap))
                if rename:
                    cap_img = cv2.imread(os.path.join(path, "segmented", "hasty", "renamed_individual_caps_translated", cap))
                cap_img = rgb2gray(cap_img)

                # Pad cap to match frame
                cap_img = np.pad(cap_img, ((PAD_VALUE, PAD_VALUE), (PAD_VALUE, PAD_VALUE)))

                # Translate cap
                if miny == 0 and minx == 0:
                    resized_cap = cap_img[maxy:, maxx:]
                elif miny == 0:
                    resized_cap = cap_img[maxy:, maxx:minx]
                elif minx == 0:
                    resized_cap = cap_img[maxy:miny, maxx:]
                else:
                    resized_cap = cap_img[maxy:miny, maxx:minx]

                if np.argwhere(resized_cap != 0).size == 0:
                    continue

                # Get label coordinates
                xcoord, ycoord = get_label_position(resized_cap)

                height, width = len(resized_cap), len(resized_cap[0])
                
                # Make overlay
                overlay = np.zeros_like(frame_img)
                for y in range(height):
                    for x in range(width):
                        if resized_cap[y][x] != 0:
                            alpha = int(0.5 * resized_cap[y][x])
                            overlay[y, x] = [color[0], color[1], color[2], alpha]
                overlay = overlay.astype(np.uint8)
                overlayed = cv2.addWeighted(frame_img, 1, overlay, 1, 0)

                # Add label
                c2match = re.search(r'cap_(.{3})', cap)
                capnuma = "cap_" + c2match.group(1)
                cv2.putText(overlayed, capnuma, (xcoord, ycoord), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3, cv2.LINE_AA)  # Black outline
                cv2.putText(overlayed, capnuma, (xcoord, ycoord), cv2.FONT_HERSHEY_PLAIN, 2, (225, 225, 225), 2, cv2.LINE_AA)  # White text

                # Save to location folder
                set_part_date, location, vid = extract_file_info(cap)
                filename = set_part_date + location + vid + "overlay.png"
                frame_img = overlayed
                overlay_folder = os.path.join(path, "segmented", "hasty", "overlays")
                os.makedirs(overlay_folder, exist_ok=True)
                cv2.imwrite(os.path.join(overlay_folder, filename), overlayed)

                # Save to results
                if platform.system() != 'Windows':
                    if rename:
                        overlays_fp = '/hpc/projects/capillary-flow/results/size/renamed_overlays'
                        os.makedirs(overlays_fp, exist_ok=True)
                        cv2.imwrite(os.path.join(overlays_fp, filename), overlayed)
                    else:
                        overlays_fp = '/hpc/projects/capillary-flow/results/size/overlays'
                        os.makedirs(overlays_fp, exist_ok=True)
                        cv2.imwrite(os.path.join(overlays_fp, filename), overlayed)


if __name__ == "__main__":
    # Measure the runtime of the script
    ticks = time.time()
    make_overlays()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
