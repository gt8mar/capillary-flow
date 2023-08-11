"""
Filename: normalize_rows.py
---------------------------
This function normalizes rows in an image using a z-score
"""
import time
import numpy as np

def normalize_image(image):
    image = image - np.min(image)
    image = image / np.max(image)
    image *= 255
    image = np.rint(image)
    return image.astype('uint8')

def normalize_rows(image):
    """ this function normalizes the rows of an image """
    # This normalizes using a z score
    average_col = np.mean(image, axis = 1) # averages along the rows to give one big column
    std_col = np.std(image, axis = 1)
    big_average = np.tile(average_col, (image.shape[1], 1)).transpose()
    big_std = np.tile(std_col, (image.shape[1], 1)).transpose()
    subtracted_image = (image - big_average)/big_std
    new_image = normalize_image(subtracted_image)               # this scales to 0-255
    return new_image

def normalize_rows_mean_division(image):
    """ this function normalizes the rows of an image """
    # This normalizes using a z score
    average_col = np.mean(image, axis = 1) # averages along the rows to give one big column
    std_col = np.std(image, axis = 1)
    big_average = np.tile(average_col, (image.shape[1], 1)).transpose()
    big_std = np.tile(std_col, (image.shape[1], 1)).transpose()
    divided_image = image/big_average
    new_image = normalize_image(divided_image)               # this scales to 0-255
    return new_image

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    import time
    import os
    import matplotlib.pyplot as plt
    from src.tools.load_csv_list import load_csv_list
    ticks = time.time()
    folder = 'C:\\Users\\ejerison\\capillary-flow\\data\\processed\\set_01\\sample_009\\F_blood_flow'
    images = load_csv_list(folder)
    for i in range(len(images)):
        norm_image_div = normalize_rows_mean_division(images[i])
        norm_image_z = normalize_rows(images[i])
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(norm_image_z)
        ax[0].set_title('Z-score')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Centerline (0.5um)')
        ax[1].imshow(norm_image_div)
        ax[1].set_title('division method')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Centerline (0.5um)')
        plt.title(f"Capillary {i}")
        plt.show()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))