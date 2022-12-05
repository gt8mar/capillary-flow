import os
import numpy as np
import cv2

def load_image_array(image_list, input_folder):
    """
    This function loads images into a numpy array. It trims them based on 
    the shifts file.  
    :param image_list: List of image names
    :return: image_array: 3D numpy array
    """
    # Initialize array for images
    z_time = len(image_list)
    image_example = cv2.imread(os.path.join(input_folder, 'vid', image_list[0]), cv2.IMREAD_GRAYSCALE)
    rows, cols= image_example.shape
    image_array = np.zeros((z_time, rows, cols), dtype=int)
    # loop to populate array
    for i in range(z_time):
        image = cv2.imread(os.path.join(input_folder,'vid', image_list[i]), cv2.IMREAD_GRAYSCALE)
        image_array[i] = image
    return image_array