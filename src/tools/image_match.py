import os
import cv2
import numpy as np

def list_images(directory):
    # List all .tif image files in the specified directory
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]

def read_images(image_files):
    # Load each image file into a NumPy array
    images = {}
    for file in image_files:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)  # Reading the image using OpenCV
        images[file] = img
    return images

def match_images(folder1, folder2):
    # Find matching images between two folders
    images1 = list_images(folder1)
    images2 = list_images(folder2)

    loaded_images1 = read_images(images1)
    loaded_images2 = read_images(images2)

    matched_images = []
    for file1, img1 in loaded_images1.items():
        for file2, img2 in loaded_images2.items():
            if img1.shape == img2.shape and np.array_equal(img1, img2):
                matched_images.append([os.path.basename(file1), os.path.basename(file2)])
    
    return matched_images

if __name__ == "__main__":
    folder1 = 'f:\\Marcus\\data\\part30\\231130\\loc02\\vids\\vid24\\moco'
    folder2 = 'f:\\Marcus\\data\\part30\\231130\\loc02\\vids\\vid24\\mocoslice'
    matched_images = match_images(folder1, folder2)
    for pair in matched_images:
        print(f'Matched images: {pair[0]} and {pair[1]}')