import time
import os
import cv2
import numpy as np
import csv

def align_images(reference_image_path, target_image_path):
    # Load the images
    img1 = cv2.imread(reference_image_path, 0)  # referenceImage
    img2 = cv2.imread(target_image_path, 0)  # targetImage

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate the affine transformation matrix
    M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method='cv2.LMEDS')

    # Warp img2 to img1
    aligned_img = cv2.warpAffine(img2, M, (img1.shape[1], img1.shape[0]))

    return aligned_img, M

def main(path):
    segmented_folder = os.path.join(path, 'segmented')
    output_folder = os.path.join(path, 'segmented_aligned')
    os.makedirs(output_folder, exist_ok=True)
    
    segmented_folder_list = os.listdir(segmented_folder)
    reference_image_path = os.path.join(segmented_folder, segmented_folder_list[0])

    # Save the reference image to the output folder
    reference_image_output_path = os.path.join(output_folder, segmented_folder_list[0])
    reference_image = cv2.imread(reference_image_path, 0)
    cv2.imwrite(reference_image_output_path, reference_image)
    
    # Prepare CSV file for saving transformation matrices
    csv_file_path = os.path.join(output_folder, 'transformation_matrices.csv')
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Image', 'M[0,0]', 'M[0,1]', 'M[0,2]', 'M[1,0]', 'M[1,1]', 'M[1,2]'])  # Header

        for i in range(1, len(segmented_folder_list)):
            target_image_path = os.path.join(segmented_folder, segmented_folder_list[i])
            
            # Align the current image to the reference image
            aligned_image, M = align_images(reference_image_path, target_image_path)
            
            # Save the aligned image with the same name in the new folder
            aligned_image_path = os.path.join(output_folder, segmented_folder_list[i])
            cv2.imwrite(aligned_image_path, aligned_image)
            
            # Save the transformation matrix to CSV
            csv_writer.writerow([
                segmented_folder_list[i],
                M[0, 0], M[0, 1], M[0, 2],
                M[1, 0], M[1, 1], M[1, 2]
            ])

if __name__ == "__main__":
    ticks = time.time()
    main(path = 'E:\\frog\\24-2-14 WkSl\\Frog4\\Right')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))