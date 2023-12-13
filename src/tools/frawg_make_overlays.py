import time
import os
import numpy as np
import cv2
from naming_overlay import get_label_position
import csv

def rename_files(rename_map_fp, individual_caps_folder):
    with open(rename_map_fp, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            old_name = os.path.join(individual_caps_folder, row[0])
            new_name = os.path.join(individual_caps_folder, row[1])
            os.rename(old_name, new_name)

def main(path, rename = False):
    backgrounds_folder = os.path.join(path, 'backgrounds')
    individual_caps_folder = os.path.join(path, 'individual_caps')
    if rename:
        old_overlays_folder = os.path.join(path, 'overlays')
        overlays_folder = os.path.join(path, 'overlays_renamed')
        rename_files(os.path.join(old_overlays_folder, 'rename_map.csv'), individual_caps_folder)
    else:
        overlays_folder = os.path.join(path, 'overlays')
    os.makedirs(overlays_folder, exist_ok = True)

    predefined_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Lime
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (255, 140, 0),  # Dark Orange
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
    ]

    colors = predefined_colors.copy()
    for bg_file in os.listdir(backgrounds_folder):
        if bg_file.endswith('.png'):
            bg_image = cv2.imread(os.path.join(backgrounds_folder, bg_file))
            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2BGRA)
            for cap_file in os.listdir(individual_caps_folder):
                if cap_file.endswith('.png'):
                    cap_image = cv2.imread(os.path.join(individual_caps_folder, cap_file), cv2.IMREAD_GRAYSCALE)
                    cap_name = cap_file[:-7]
                    if cap_name in bg_file:
                        overlay= np.zeros_like(bg_image)
                        color = colors.pop(0)

                        #add cap overlay
                        for y, x in np.ndindex(bg_image.shape[:2]):
                            if cap_image[y][x] != 0:
                                alpha = int(0.5 * cap_image[y][x])
                                overlay[y, x] = color[0], color[1], color[2], alpha
                        overlay = overlay.astype(np.uint8)
                        overlayed = cv2.addWeighted(bg_image, 1, overlay, 1, 0)

                        #add label
                        xcoord, ycoord = get_label_position(cap_image)
                        capnum = cap_file.split('_')[1][:-4]
                        cv2.putText(overlayed, capnum, (xcoord, ycoord), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 6, cv2.LINE_AA)

                        bg_image = overlayed
            
            #save
            cv2.imwrite(os.path.join(overlays_folder, bg_file[:-4] + '_overlay.png'), overlayed)
            colors = predefined_colors.copy()


if __name__ == "__main__":
    ticks = time.time()
    main(path = 'E:\\frawg\\Wake Sleep Pairs\\gabby_analysis', rename = True)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))