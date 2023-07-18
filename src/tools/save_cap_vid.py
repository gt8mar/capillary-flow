"""
File: save_cap_vid.py
---------------------
This program saves the masked region of a series of TIFF images as a video.

By: Marcus Forst
"""

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os, time
from src.tools.parse_vid_path import parse_vid_path
from src.tools.get_images import get_images
from src.tools.load_image_array import load_image_array
from src.tools.get_shifts import get_shifts

def crop_frame_around_mask(image_array, mask, padding=20):
    # Find the bounding box coordinates of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cropped_frames = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the new cropping dimensions
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding

        # Perform the cropping
        cropped_frame = image_array[:, max(0, y):y + h, max(0, x):x + w]
        cropped_frames.append(cropped_frame)
    return cropped_frames

def save_video(cropped_masked_array, cap_name, path, fps=114, plot=False):
    """
    Saves the cropped frames as a video.

    Args:
        cropped_masked_array (np.ndarray): The array of cropped frames.
        path (str): The path to the video folder.
        fps (int, optional): The frames per second of the video. Defaults to 114.
    
    Returns:
        0 if successful.
    """
    # Get the base file name (without extension) to find the corresponding mask file
    participant, date, video, file_prefix = parse_vid_path(path)

    # make zeros in masked array equal to the average of masked array
    cropped_masked_array[cropped_masked_array == 0] = np.mean(cropped_masked_array[cropped_masked_array != 0])
    # normalize masked array to be between 0 and 255
    cropped_masked_array = (cropped_masked_array - np.min(cropped_masked_array)) / (np.max(cropped_masked_array) 
                                - np.min(cropped_masked_array)) * 255
    if plot:
        plt.imshow(cropped_masked_array[0])
        plt.show() 

    # ------------------------make video-------------------------------------------------------------
    # Create a VideoWriter object to save the video
    os.makedirs(os.path.join(path, 'G_masked_vid'), exist_ok=True)
    output_file = os.path.join(path,'G_masked_vid', f'{file_prefix}_capillary_{cap_name}.avi')
    fourcc = 'raw'  # Specify the codec (MP4V)
    frame_size = (cropped_masked_array.shape[2], cropped_masked_array.shape[1])  # (width, height)
    video_writer = imageio.get_writer(output_file, format='FFMPEG', mode='I')

    # Iterate over each frame and write it to the video
    for frame in cropped_masked_array:
        frame = frame.astype(np.uint8)  # Convert the frame to uint8 data type
        video_writer.append_data(frame)

    # Release the video writer
    video_writer.close()

def main(path = "F:\\Marcus\\data\\part11\\230427\\vid16", plot=False):
    """
    This function saves the masked region of a series of TIFF images as a video.

    Args:
        path (str): The path to the folder containing capillary flow data.
    
    Returns:
        0 (int): The function returns 0 if it runs successfully.
    """

    # input folders
    moco_folder = os.path.join(path, 'moco')
    mask_folder = os.path.join(path, "D_segmented")
    metadata_folder = os.path.join(path, 'metadata')
    
    # Get the base file name (without extension) to find the corresponding mask file
    participant, date, video, file_prefix = parse_vid_path(path)
    base_name = file_prefix + '_background_seg'
    mask_path = os.path.join(mask_folder, base_name + '.png')

    # Import images
    start = time.time()
    images = get_images(moco_folder)
    image_array = load_image_array(images, moco_folder)      # this has the shape (frames, row, col)
    gap_left, gap_right, gap_bottom, gap_top = get_shifts(metadata_folder)
    print(f"Gap left: {gap_left}, gap right: {gap_right}, gap bottom: {gap_bottom}, gap top: {gap_top}")
    image_array = image_array[:, gap_top:image_array.shape[1] + gap_bottom, gap_left:image_array.shape[2] + gap_right]
    example_image = image_array[0]
    print(f"Loading images for {file_prefix} took {time.time() - start} seconds")
    print("The size of the array is " + str(image_array.shape))

    # Read the mask file as a grayscale image
    mask = cv2.imread(mask_path, 0)
    binary_mask = cv2.threshold(mask, 0, 1,cv2.THRESH_BINARY)[1]
    if plot:
        plt.imshow(binary_mask)
        plt.show()

    # Apply the mask to the image array
    masked_array = (binary_mask * image_array) 
    cropped_masked_arrays = crop_frame_around_mask(masked_array, binary_mask)

    # TODO: add correct naming to save_video

    # Save the masked arrays as videos
    for i in range(len(cropped_masked_arrays)):
        cap_name = str(i).zfill(2)  
        save_video(cropped_masked_arrays[i], cap_name, path, plot=plot)
    return 0

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))