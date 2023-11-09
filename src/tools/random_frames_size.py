import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from enumerate_capillaries2 import find_connected_components
from group_caps import separate_caps
from PIL import Image

def group_caps(seg_path):
    rows, cols = cv2.imread(os.path.join(seg_path, os.listdir(seg_path)[0]), cv2.IMREAD_GRAYSCALE).shape
    max_project = np.zeros((rows, cols))
    for file in os.listdir(seg_path):
        max_project += cv2.imread(os.path.join(seg_path, file), cv2.IMREAD_GRAYSCALE)

    caps = find_connected_components(max_project)

    caps_fp = os.path.join(os.path.dirname(seg_path), "proj_caps")
    os.makedirs(caps_fp, exist_ok=True)
    for x in range(len(caps)):
        filename = "cap_" + str(x).zfill(2) + ".png"
        cap_fp = os.path.join(caps_fp, filename)
        cv2.imwrite(str(cap_fp), caps[x])

    separate_caps(seg_path)


def plot_size(caps_path):
    image_dir = caps_path
    image_files = os.listdir(image_dir)

    image_dict = {}
    all_video_names = set()  # Store all unique video names

    for image_file in image_files:
        video_name, cap_number = image_file.split('_cap_')
        cap_number = cap_number.split('.')[0]
        video_name = video_name.split('_')[2]
        if cap_number not in image_dict:
            image_dict[cap_number] = []
        image_dict[cap_number].append((video_name, image_file))
        all_video_names.add(video_name)

    for cap_number in image_dict:
        image_dict[cap_number].sort(key=lambda x: x[0])

    fig, ax = plt.subplots(len(image_dict), 1, figsize=(10, 14), sharex=True, sharey=True)
    plt.tight_layout(pad=3, w_pad=1, h_pad=5)

    for i, (cap_number, images) in enumerate(image_dict.items()):
        video_names, non_zero_counts = [], []

        for video_name, image_file in images:
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path)
            image_array = np.array(image)
            non_zero_pixels = np.count_nonzero(image_array)

            video_names.append(video_name)  # Keep video_name as a string
            non_zero_counts.append(non_zero_pixels)

        # Create a dictionary to map video_names to non_zero_counts
        data_dict = dict(zip(video_names, non_zero_counts))

        # Sort all video names to ensure a consistent x-axis
        all_video_names = sorted(all_video_names)

        # Create a list of y values with 0 for missing x values
        filled_non_zero_counts = [data_dict.get(video_name, 0) for video_name in all_video_names]

        ax[i].scatter(all_video_names, filled_non_zero_counts, marker='o', s=30)
        ax[i].set_title(f'Cap {cap_number}')
        ax[i].set_xlabel('Frame Number')
        ax[i].set_ylabel('Nonzero Pixels')

    plt.show()



if __name__ == "__main__":
    ticks = time.time()
    #group_caps(seg_path='C:\\Users\\Luke\Documents\\capillary-flow\\random_frames\\part16_230501_vid06\\segmented')
    plot_size(caps_path='C:\\Users\\Luke\Documents\\capillary-flow\\random_frames\\part16_230501_vid06\\individual_caps_translated')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))