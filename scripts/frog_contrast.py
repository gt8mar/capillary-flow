"""
Filename: frog_contrast.py
-----------------------------------------
This file automatically contrasts videos of frog capillaries, without having to manually check. 
By: Juliette Levy
"""

import os
import cv2
import numpy
import time

FOLDER = "C:\\Users\\ejerison\\capillary-flow\\frog\\results\\stdevs"
OUTPUT_PATH = "C:\\Users\\ejerison\\capillary-flow\\frog\\results\\stdevs-contrasted"
os.makedirs(OUTPUT_PATH, exist_ok= True)

def main(method = "hist"):
    filenames = os.listdir(FOLDER)
    print(filenames)
    for filename in filenames:
        file_image = cv2.imread(os.path.join(FOLDER, filename), cv2.IMREAD_GRAYSCALE)
        if method == "hist":
            file_image = cv2.equalizeHist(file_image)
        else:
            clahe = cv2.create_CLAHE(cliplimit = 2.0, tileGRIDSIZE = (8,8))
            file_image = clahe.apply(file_image)
        cv2.imwrite(OUTPUT_PATH, file_image)






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