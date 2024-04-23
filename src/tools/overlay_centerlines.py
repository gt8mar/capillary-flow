"""
Filename: overlay_centerlines.py
---------------------------------

This script provides functions to overlay a centerline on an image using a mask and centerline data.

Functions:
- load_image_and_mask(image_path, mask_path): Loads an image and a mask from the given file paths.
- apply_mask(image, mask, color=(0, 255, 0), alpha=0.5): Creates an overlay of the mask on the image.
- plot_centerline(overlay, csv_path): Plots the centerline on the overlay using centerline data from a CSV file.
- save_image(overlay, output_path): Saves the final image with the overlay applied.
- main(image_path, mask_path, csv_path, output_path): The main function that orchestrates the overlay process.

Example usage:
main('path_to_image.jpg', 'path_to_mask.png', 'path_to_centerline.csv', 'output_image.jpg')
"""

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the image and the mask
def load_image_and_mask(image_path, mask_path):
    """
    Loads an image and a mask from the given file paths.

    Args:
        image_path (str): The file path of the image.
        mask_path (str): The file path of the mask.

    Returns:
        tuple: A tuple containing the loaded image and mask.
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return image, mask

# Create an overlay of the mask on the image
def apply_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Creates an overlay of the mask on the image.

    Args:
        image (numpy.ndarray): The image to overlay the mask on.
        mask (numpy.ndarray): The mask to be applied.
        color (tuple, optional): The color of the overlay. Defaults to (0, 255, 0) (green).
        alpha (float, optional): The transparency of the overlay. Defaults to 0.5.

    Returns:
        numpy.ndarray: The image with the mask overlay applied.
    """
    print(mask)
    # Create an overlay of the mask on the image
    # Ensure mask is binary
    mask = cv2.threshold(mask, 125, 255, cv2.THRESH_BINARY)[1]

    # Create an RGB version of the binary mask
    colored_mask = np.zeros_like(image)
    # Check if mask is not empty and the size matches the image
    if mask.any() and mask.shape == image.shape[:2]:
        colored_mask[mask == 255] = color
        # Blend the colored mask with the image
        overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
        return overlay
    else:
        raise ValueError("Mask is empty or its dimensions do not match the image dimensions")


# Plot the centerline
def plot_centerline(overlay, csv_path, color = (255, 0, 0)):
    """
    Plots the centerline on the overlay using centerline data from a CSV file.

    Args:
        overlay (numpy.ndarray): The image with the mask overlay applied.
        csv_path (str): The file path of the centerline data in CSV format.

    Returns:
        numpy.ndarray: The image with the centerline plotted.
    """
    # Load centerline data without headers, only the first two columns
    centerline = pd.read_csv(csv_path, header=None, usecols=[0, 1], names=['y', 'x'])

    # Plotting the centerline
    for i in range(len(centerline) - 5):
        pt1 = (int(centerline.iloc[i]['x']), int(centerline.iloc[i]['y']))
        pt2 = (int(centerline.iloc[i + 1]['x']), int(centerline.iloc[i + 1]['y']))
        cv2.line(overlay, pt1, pt2, color, 2)  # Blue color line

    return overlay

# Save the final image
def save_image(overlay, output_path):
    """
    Saves the final image with the overlay applied.

    Args:
        overlay (numpy.ndarray): The image with the overlay applied.
        output_path (str): The file path to save the final image.

    Returns:
        None
    """
    cv2.imwrite(output_path, overlay)

def main(image_path, mask_path, csv_path, output_path):
    """
    The main function that orchestrates the overlay process.

    Args:
        image_path (str): The file path of the image.
        mask_path (str): The file path of the mask.
        csv_path (str): The file path of the centerline data in CSV format.
        output_path (str): The file path to save the final image.

    Returns:
        None
    """
    color2 = (214, 39, 40)
    color = (255, 127, 14)
    image, mask = load_image_and_mask(image_path, mask_path)
    overlay = apply_mask(image, mask, color = color) #31, 119, 180
    overlay = plot_centerline(overlay, csv_path, color = color2) #255, 127, 14
    save_image(overlay, output_path)

if __name__ == '__main__':
    vid = 'vid21'
    image_path = "C:\\Users\\gt8mar\\capillary-flow\\results\\backgrounds\\set01_part29_231130_loc02_vid21_background.tiff"
    mask_path = "C:\\Users\\gt8mar\\capillary-flow\\results\\segmented\\individual_caps_original\\set01_part29_231130_loc02_vid21_seg_cap_03a.png"
    csv_path = "C:\\Users\\gt8mar\\capillary-flow\\results\\centerlines\\set01_part29_231130_loc02_vid21_centerline_03a.csv"
    output_path = "C:\\Users\\gt8mar\\capillary-flow\\results\\set01_part29_231130_loc02_vid21_centerline_03a_overlay.png"
    main(image_path, mask_path, csv_path, output_path)