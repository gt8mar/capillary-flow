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


# Input_path = "C:\\Users\\gt8mar\\capillary-flow\\data\\part*\\" #edit to bring patht to moco folder
# Output_path = "C:\\Users\\ejerison\\capillary-flow\\frog\\results\\stdevs-contrasted"
# os.makedirs(OUTPUT_PATH, exist_ok= True)

# def main(method = "hist"):

def main(input_folder, output_folder, saturated_percentage=.85):  # the default is 0.35 or 0.35% saturation
    #making input and output folders
    # input_folder = "C:\\Users\\gt8mar\\capillary-flow\\data\\part35\\240517\\loc01\\vids\\vid01\\moco" 
    # output_folder = "C:\\Users\\gt8mar\\capillary-flow\\data\\part35\\240517\\loc01\\vids\\vid01\\moco-contrasted"
    os.makedirs(output_folder, exist_ok= True) #dont crash if the folder is already there!!

    #grabbing files
    filenames = get_images(input_folder) #puts each file from input folder into a numerical list
    first_filename = filenames[0] #getting first image
    loaded_images = load_image_array(filenames, input_folder) #
    first_image = loaded_images[0].astype(np.uint8) #converts the first image to 8 bit unsigned integer
    histogram = cv2.calcHist([first_image], [0], None, [256], [0, 256]).flatten()
    total_pixels = first_image.size

    first_fame_contrast = cv2.equalizeHist(first_image)
    lower_cutoff, upper_cutoff = calculate_histogram_cutoffs(histogram, total_pixels, saturated_percentage)
    processed_image = apply_contrast(first_image, lower_cutoff, upper_cutoff)

    # # plot processed image
    # plt.imshow(processed_image, cmap='viridis')
    # plt.show()

     # Plotting the images side by side using matplotlib
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Adjusted to have three subplots
    ax[0].imshow(first_image, cmap='viridis')
    ax[0].title.set_text('Original Image')
    ax[0].axis('off')  # Turn off axis

    ax[1].imshow(first_fame_contrast, cmap='viridis')
    ax[1].title.set_text('Histogram Equalized')
    ax[1].axis('off')  # Turn off axis

    ax[2].imshow(processed_image, cmap='viridis')
    ax[2].title.set_text('Contrast Stretched')
    ax[2].axis('off')  # Turn off axis

    plt.show()

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
def calculate_histogram_cutoffs(histogram, total_pixels, saturated_percentage):
    """ Calculate the cutoffs for histogram stretching based on saturation percentage. """
    saturated_pixels = total_pixels * (saturated_percentage / 100.0) / 2
    lower_sum = 0
    upper_sum = 0
    lower_cutoff = 0
    upper_cutoff = len(histogram) - 1

    # Calculate lower cutoff
    for i in range(len(histogram)):
        lower_sum += histogram[i]
        if lower_sum > saturated_pixels:
            lower_cutoff = i
            break

    # Calculate upper cutoff
    for i in reversed(range(len(histogram))):
        upper_sum += histogram[i]
        if upper_sum > saturated_pixels:
            upper_cutoff = i
            break

    return lower_cutoff, upper_cutoff

def apply_contrast(image, lower_cutoff, upper_cutoff, hist_size=256):
    """Apply contrast adjustment based on the computed histogram cutoffs."""
    # Create the lookup table
    lut = np.linspace(0, hist_size - 1, hist_size, dtype=np.uint8)  # linear LUT for all values

    # Modify LUT to stretch the histogram between lower_cutoff and upper_cutoff
    lut[:lower_cutoff] = 0  # Set all values below lower_cutoff to black
    lut[upper_cutoff:] = 255  # Set all values above upper_cutoff to white
    if upper_cutoff > lower_cutoff:
        scale = 255 / (upper_cutoff - lower_cutoff)
        lut[lower_cutoff:upper_cutoff] = scale * (np.arange(lower_cutoff, upper_cutoff) - lower_cutoff)

    # Apply the LUT
    return cv2.LUT(image, lut)

# """
# -----------------------------------------------------------------------------
# """
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    # input_folder = "C:\\Users\\gt8mar\\capillary-flow\\data\\part35\\240517\\loc01\\vids\\vid01\\moco" 
    # output_folder = "C:\\Users\\gt8mar\\capillary-flow\\data\\part35\\240517\\loc01\\vids\\vid01\\moco-contrasted"
    input_folder = "E:\\Marcus\\data\\part35\\240517\\loc01\\vids\\vid01\\moco" 
    output_folder = "E:\\Marcus\\data\part35\\240517\\loc01\\vids\\vid01\\moco-contrasted"
    ticks = time.time()
    main(input_folder, output_folder)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))