"""
Filename: capillary_contrast.py
-----------------------------------------
This file automatically contrasts videos of capillaries, without having to manually check. 
By: Juliette Levy
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from src.tools.get_images import get_images
from src.tools.load_image_array import load_image_array

def calculate_histogram_cutoffs(histogram, total_pixels, saturated_percentage):
    """
    Calculate the cutoffs for histogram stretching based on saturation percentage.

    Args:
        histogram (numpy.ndarray): The image histogram.
        total_pixels (int): Total number of pixels in the image.
        saturated_percentage (float): Percentage of pixels to saturate.

    Returns:
        tuple: A tuple containing the lower and upper cutoff values.
    """
    saturated_pixels = total_pixels * (saturated_percentage / 100.0) / 2
    lower_sum, upper_sum = 0, 0
    lower_cutoff, upper_cutoff = 0, len(histogram) - 1

    for i in range(len(histogram)):
        lower_sum += histogram[i]
        if lower_sum > saturated_pixels:
            lower_cutoff = i
            break

    for i in reversed(range(len(histogram))):
        upper_sum += histogram[i]
        if upper_sum > saturated_pixels:
            upper_cutoff = i
            break

    return lower_cutoff, upper_cutoff

def apply_contrast(image, lower_cutoff, upper_cutoff, hist_size=256):
    """
    Apply contrast adjustment based on the computed histogram cutoffs.

    Args:
        image (numpy.ndarray): Input image.
        lower_cutoff (int): Lower cutoff value for contrast adjustment.
        upper_cutoff (int): Upper cutoff value for contrast adjustment.
        hist_size (int, optional): Size of the histogram. Defaults to 256.

    Returns:
        numpy.ndarray: Contrast-adjusted image.
    """
    lut = np.arange(hist_size, dtype=np.uint8)
    lut[:lower_cutoff] = 0
    lut[upper_cutoff:] = 255
    if upper_cutoff > lower_cutoff:
        scale = 255 / (upper_cutoff - lower_cutoff)
        lut[lower_cutoff:upper_cutoff] = np.clip(scale * (np.arange(lower_cutoff, upper_cutoff) - lower_cutoff), 0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)

def capillary_contrast(input_folder, output_folder, saturated_percentage=0.85, plot=False):
    """
    Process capillary images by applying contrast enhancement.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
        saturated_percentage (float, optional): Percentage of pixels to saturate. Defaults to 0.85.
        plot (bool, optional): Whether to plot the original and processed first image. Defaults to False.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    filenames = get_images(input_folder)
    loaded_images = load_image_array(filenames, input_folder)
    first_image = loaded_images[0].astype(np.uint8)

    histogram = cv2.calcHist([first_image], [0], None, [256], [0, 256]).flatten()
    total_pixels = first_image.size
    lower_cutoff, upper_cutoff = calculate_histogram_cutoffs(histogram, total_pixels, saturated_percentage)
    first_frame_contrast = apply_contrast(first_image, lower_cutoff, upper_cutoff)

    if plot:
        plot_comparison(first_image, first_frame_contrast)

    for i, image in enumerate(loaded_images):
        processed_image = apply_contrast(image.astype(np.uint8), lower_cutoff, upper_cutoff)
        output_path = os.path.join(output_folder, f'processed_{filenames[i]}')
        cv2.imwrite(output_path, processed_image)

    print(f"Processed {len(loaded_images)} images using cutoffs: lower={lower_cutoff}, upper={upper_cutoff}")

def plot_comparison(original_image, contrasted_image):
    """
    Plot the original and contrasted images side by side.

    Args:
        original_image (numpy.ndarray): The original image.
        contrasted_image (numpy.ndarray): The contrast-enhanced image.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_image, cmap='viridis')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(contrasted_image, cmap='viridis')
    ax[1].set_title('Contrasted Image')
    ax[1].axis('off')

    plt.show()

if __name__ == "__main__":
    input_folder = "D:\\Marcus\\data\\part35\\240517\\loc01\\vids\\vid03\\moco"
    output_folder = "D:\\Marcus\\data\\part35\\240517\\loc01\\vids\\vid03\\moco-contrasted"
    start_time = time.time()
    capillary_contrast(input_folder, output_folder)
    print(f"Runtime: {time.time() - start_time:.2f} seconds")