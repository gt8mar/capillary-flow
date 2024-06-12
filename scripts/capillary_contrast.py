"""
Filename: capillary_contrast.py
-----------------------------------------
This file automatically contrasts videos of capillaries, without having to manually check. 
By: Juliette Levy
"""

import os
import cv2
import numpy as np
import time
from src.tools.get_images import get_images
from src.tools.load_image_array import load_image_array


# Input_path = "C:\\Users\\gt8mar\\capillary-flow\\data\\part*\\" #edit to bring patht to moco folder
# Output_path = "C:\\Users\\ejerison\\capillary-flow\\frog\\results\\stdevs-contrasted"
# os.makedirs(OUTPUT_PATH, exist_ok= True)

# def main(method = "hist"):

def main(input_folder, output_folder):
    #making input and output folders
    input_folder = "C:\\Users\\gt8mar\\capillary-flow\\data\\part35\\240517\\loc01\\vids\\vid01\\moco" 
    output_folder = "C:\\Users\\gt8mar\\capillary-flow\\data\\part35\\240517\\loc01\\vids\\vid01\\moco-contrasted"
    os.makedirs(output_folder, exist_ok= True) #dont crash if the folder is already there!!

    #grabbing files
    filenames = get_images(filenames, input_folder) #puts each file from input folder into a numerical list
    first_filename = filenames[0] #getting first image
    loaded_images = load_image_array(filenames, input_folder) #
    first_image = loaded_images[0]
    first_fame_contrast = cv2.equalizeHist(first_image)
    min_, max_ = calculate_contrast_limits(first_image, first_image.dtype)
    processed_image = apply_contrast(image, min_, max_)

    # Concatenate images horizontally
    concatenated_image = np.hstack((first_fame_contrast, processed_image))

    # Display the concatenated image
    cv2.imshow('Side by Side Images', concatenated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for i in range(len(loaded_images)):
        filename = filenames[i] #the [i] tracks which iteration of the filename you are on - remembers which filename you are on
        image = loaded_images[i] #does the same for the image

        # #makes sure the file is a video
        # if not filename.lower().endswith(('.tif')):
        #     continue

        # #make path for each image file to the input/output folder
        # input_path = os.path.join(input_folder, filename)
        # output_path = os.path.join(output_folder, 'processed_' + filename)
        # first_fame_contrast = cv2.equalizeHist(image)
        






        # #opens video file
        # cap = cv2.VideoCapture(Input_path)

        # #read the first frame, and print if error
        # if not cap.isOpened():
        #     print("Error: Could not open video file.")
        # else:
        # # Read the first frame
        #     ret, first_frame = cap.read()
        #     first_fame_contrast = cv2.equalizeHist(filename)
        #     if True:

                    

        # cv2.imwrite(OUTPUT_PATH, file_image)
        # file_image = cv2.imread(os.path.join(FOLDER, filename), cv2.IMREAD_GRAYSCALE)


    # def main()):
        # filenames = os.listdir(FOLDER)
        # print(filenames)
        # for filename in filenames:
        # file_image = cv2.imread(os.path.join(FOLDER, filename), cv2.IMREAD_GRAYSCALE)
        # if method == "hist":
        #     file_image = cv2.equalizeHist(file_image)
        # else:
        #     clahe = cv2.create_CLAHE(cliplimit = 2.0, tileGRIDSIZE = (8,8))
        #     file_image = clahe.apply(file_image)
        # cv2.imwrite(OUTPUT_PATH, file_image)



"""
Helpful functions
-----------------
"""
def calculate_contrast_limits(image, image_type):
    """Calculate the histogram-based contrast limits of an image.

    Args:
        image (ndarray): The image array in grayscale.
        image_type (dtype): The data type of the image (e.g., np.uint8).

    Returns:
        tuple: A tuple containing the minimum and maximum values for contrast adjustment.

    Raises:
        NotImplementedError: If the image type is not supported.
    """
    # Set the histogram parameters based on the image data type
    if image_type == np.uint8:
        hist_min = 0
        hist_max = 256
    elif image_type in (np.uint16, np.int32):
        hist_min = np.min(image)
        hist_max = np.max(image)
    else:
        raise NotImplementedError(f"Not implemented for dtype {image_type}")

    # Compute histogram
    histogram, bin_edges = np.histogram(image, bins=256, range=(hist_min, hist_max))
    bin_size = (hist_max - hist_min) / 256

    # Calculate the thresholds
    pixel_count = image.size
    limit = pixel_count / 10
    auto_threshold = max(5000, pixel_count / 1000)
    threshold = int(pixel_count / auto_threshold)

    # Find the minimum bin with count above the threshold
    hmin = next((i for i, count in enumerate(histogram) if count > threshold and count > limit), 0)

    # Find the maximum bin with count above the threshold
    hmax = next((i for i in reversed(range(256)) if histogram[i] > threshold and histogram[i] > limit), 255)

    # Convert histogram bins to actual pixel values
    min_ = hist_min + hmin * bin_size
    max_ = hist_min + hmax * bin_size

    # Handle edge cases where no valid range is found
    if hmax <= hmin:
        min_ = hist_min
        max_ = hist_max

    return min_, max_

def apply_contrast(image, min_, max_):
    """Apply contrast stretching to an image using specified min and max values.

    Args:
        image (ndarray): The original image array.
        min_ (float): The minimum pixel value for contrast stretching.
        max_ (float): The maximum pixel value for contrast stretching.

    Returns:
        ndarray: The contrast-stretched image.
    """
    return np.clip((image - min_) / (max_ - min_) * 255, 0, 255).astype(np.uint8)


# """
# -----------------------------------------------------------------------------
# """
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    input_folder = "C:\\Users\\gt8mar\\capillary-flow\\data\\part35\\240517\\loc01\\vids\\vid01\\mocso" 
    output_folder = "C:\\Users\\gt8mar\\capillary-flow\\data\\part35\\240517\\loc01\\vids\\vid01\\moco-contrasted"
    ticks = time.time()
    main(input_folder, output_folder)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))