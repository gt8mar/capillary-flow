"""
Filename: frawg_match_pics.py
-----------------------------
This script contains functions to match images in a folder based on their similarity.
The similarity score is calculated based on the overlap between the images after
applying a homography transformation. The script uses ORB feature detection and
matching to find correspondences between images.

By: Marcus Forst
"""
import cv2
import numpy as np
import os
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

def load_image(image_path):
    """Loads an image in grayscale.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded grayscale image.
    """
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def extract_features(image):
    """Extracts keypoints and descriptors from an image using ORB.

    Args:
        image (numpy.ndarray): The grayscale image.

    Returns:
        tuple: Keypoints and descriptors.
    """
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    """Matches descriptors between two images using BFMatcher.

    Args:
        desc1 (numpy.ndarray): Descriptors of the first image.
        desc2 (numpy.ndarray): Descriptors of the second image.

    Returns:
        list: Sorted list of matches.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def find_homography(kp1, kp2, matches):
    """Finds the homography matrix to transform one image's keypoints to another's.

    Args:
        kp1 (list): Keypoints of the first image.
        kp2 (list): Keypoints of the second image.
        matches (list): List of matches between keypoints.

    Returns:
        tuple: Homography matrix and mask.
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask

def calculate_similarity_score(image1, image2, H):
    """Calculates a similarity score based on the overlap of two images.

    Args:
        image1 (numpy.ndarray): The first grayscale image.
        image2 (numpy.ndarray): The second grayscale image.
        H (numpy.ndarray): Homography matrix.

    Returns:
        float: The similarity score.
    """
    height, width = image2.shape
    warped_image1 = cv2.warpPerspective(image1, H, (width, height))
    overlap = cv2.bitwise_and(warped_image1, image2)
    score = np.sum(overlap) / np.sum(image2)  # Similarity score based on overlap
    return score

def process_images_in_folder(folder_path):
    """Processes all images in a folder, comparing each pair and calculating similarity scores.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        tuple: List of image file names and symmetric similarity score matrix.
    """
    images = []
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = load_image(image_path)
        images.append((image_file, image))
    
    n = len(images)
    similarity_matrix = np.zeros((n, n))
    
    for (i, (file1, img1)), (j, (file2, img2)) in combinations(enumerate(images), 2):
        kp1, desc1 = extract_features(img1)
        kp2, desc2 = extract_features(img2)
        matches = match_features(desc1, desc2)
        
        if len(matches) > 4:  # Need at least 4 matches to compute homography
            H, mask = find_homography(kp1, kp2, matches)
            score = calculate_similarity_score(img1, img2, H)
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score
    
    return image_files, similarity_matrix

def plot_similarity_matrix(image_files, similarity_matrix):
    """Plots a heatmap of similarity scores between image pairs.

    Args:
        image_files (list): List of image file names.
        similarity_matrix (numpy.ndarray): Symmetric matrix of similarity scores.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=image_files, yticklabels=image_files, annot=True, fmt=".2f", cmap='viridis')
    plt.title('Similarity Scores Between Image Pairs')
    plt.xlabel('Image Files')
    plt.ylabel('Image Files')
    plt.show()

# Example usage
folder_path = 'C:\\Users\\gt8ma\\capillary-flow\\tests\\stdevs_frogs'
image_files, similarity_matrix = process_images_in_folder(folder_path)

# Display the similarity scores
plot_similarity_matrix(image_files, similarity_matrix)

