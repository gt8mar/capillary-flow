"""
Filename: capillary_contrast.py
-----------------------------------------
This file automatically contrasts videos of capillaries, without having to manually check. 
By: Juliette Levy
"""

import os
import cv2
import numpy
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


    for i in range(len(loaded_images)):
        filename = filenames[i] #the [i] tracks which iteration of the filename you are on - remembers which filename you are on
        image = loaded_images[i] #does the same for the image

        # #makes sure the file is a video
        # if not filename.lower().endswith(('.tif')):
        #     continue

        #make path for each image file to the input/output folder
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, 'processed_' + filename)
        first_fame_contrast = cv2.equalizeHist(image)






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






# """
# -----------------------------------------------------------------------------
# """
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))