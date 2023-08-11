"""
File: optical_flow.py
---------------------
This program calculates the optical flow of the masked region in a series of TIFF images.

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

def crop_frame_around_mask(image_array, mask, padding=20):
    # Find the bounding box coordinates of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    # Calculate the new cropping dimensions
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    # Perform the cropping
    cropped_frame = image_array[:, max(0, y):y + h, max(0, x):x + w]

    return cropped_frame

def main(path = "F:\\Marcus\\data\\part11\\230427\\vid01"):
    # video_folder = "/hpc/projects/capillary-flow/data/part11/230427/vid01"
    # moco_folder = "/hpc/projects/capillary-flow/data/part11/230427/vid01/moco"
    # mask_folder = '/hpc/projects/capillary-flow/data/part11/230427/vid01/D_segmented'

    moco_folder = os.path.join(path, 'moco')
    mask_folder = os.path.join(path, "D_segmented")
    
    # Get the base file name (without extension) to find the corresponding mask file
    participant, date, video, file_prefix = parse_vid_path(path)
    base_name = file_prefix + '_background_seg'
    mask_path = os.path.join(mask_folder, base_name + '.png')

    # Import images
    start = time.time()
    images = get_images(moco_folder)
    image_array = load_image_array(images, moco_folder)      # this has the shape (frames, row, col)
    example_image = image_array[0]
    print(f"Loading images for {file_prefix} took {time.time() - start} seconds")
    print("The size of the array is " + str(image_array.shape))

    # Read the mask file as a grayscale image
    mask = cv2.imread(mask_path, 0)
    binary_mask = cv2.threshold(mask, 0, 1,cv2.THRESH_BINARY)[1]
    # plt.imshow(binary_mask)
    # plt.show()

    # Apply the mask to the image array
    masked_array = (binary_mask * image_array) 
    cropped_masked_array = crop_frame_around_mask(masked_array, binary_mask)   
    # make zeros in masked array equal to the average of masked array
    cropped_masked_array[cropped_masked_array == 0] = np.mean(cropped_masked_array[cropped_masked_array != 0])
    # normalize masked array to be between 0 and 255
    cropped_masked_array = (cropped_masked_array - np.min(cropped_masked_array)) / (np.max(cropped_masked_array) - np.min(cropped_masked_array)) * 255
    # plt.imshow(masked_array[0])
    # plt.show()

    

# ------------------------video-------------------------------------------------------------
    # Create a VideoWriter object to save the video
    output_file = path + '\\masked_vid\\'+'output_video.avi'
    fourcc = 'raw'  # Specify the codec (MP4V)
    fps = 30  # Specify the frames per second
    frame_size = (cropped_masked_array.shape[2], cropped_masked_array.shape[1])  # (width, height)
    video_writer = imageio.get_writer(output_file, format='FFMPEG', mode='I')

    # Iterate over each frame and write it to the video
    for frame in cropped_masked_array:
        frame = frame.astype(np.uint8)  # Convert the frame to uint8 data type
        video_writer.append_data(frame)

    # Release the video writer
    video_writer.close()

    # -----------------------------optical flow--------------------------------------------------
    # Create a list to store the optical flow points
    optical_flow_points = []

    # # Loop through the masked frames
    # for i in range(0, len(masked_array)-1):
    #     # Calculate optical flow
    #     flow = cv2.calcOpticalFlowFarneback(masked_array[i], masked_array[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)

    #     # Get the optical flow points within the masked region
    #     flow_masked = cv2.bitwise_and(flow, flow, mask=mask)
    #     flow_points = flow_masked[mask > 0]

    #     # Add the flow points to the list
    #     optical_flow_points.append(flow_points)

    #     # Convert the frame to an 8-bit unsigned integer format for display
    #     display_frame = cv2.convertScaleAbs(masked_array[i])

    #     # Display the frame with optical flow
    #     cv2.imshow('Optical Flow', display_frame)

    #     # Wait for the 'q' key to exit
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # Process the optical flow points as needed
    # # ...

    # # Close all windows
    # cv2.destroyAllWindows()

    return 0

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))