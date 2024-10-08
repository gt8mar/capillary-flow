import time
import os
import numpy as np
import cv2
from src.tools.naming_overlay_old import get_label_position
import csv

def rename_files(rename_map_fp, individual_caps_folder):
    with open(rename_map_fp, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            old_name = os.path.join(individual_caps_folder, row[0])
            new_name = os.path.join(individual_caps_folder, row[1])
            os.rename(old_name, new_name)

def main(path, rename = False):
    backgrounds_folder = os.path.join(path, 'stdevs')
    individual_caps_folder = os.path.join(path, 'individual_caps')
    segmented_folder = os.path.join(path, 'segmented')
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
        (0, 128, 128),  # Light Teal
        (139, 69, 19),  # Saddle Brown
        (255, 69, 0),   # Red-Orange
        (128, 0, 0),    # Maroon
    ]
    i = 0
    colors = predefined_colors.copy()
    for bg_file in os.listdir(backgrounds_folder):
        for sg_file in os.listdir(segmented_folder):
            if bg_file.endswith('.tiff') and sg_file.endswith('.png'):
                if bg_file.strip('.tiff') == sg_file.strip('.png'):
                    bg_image = cv2.imread(os.path.join(backgrounds_folder, bg_file))
                    bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2BGRA)
                    for cap_file in os.listdir(individual_caps_folder):
                        if cap_file.endswith('.png'):
                            cap_name = cap_file.strip('.png')
                            # cap_name = cap_file[:-7]
                            cap_name = cap_name[:-3] # this takes off the capillary number label. Ex: 24-2-14_WkSlAwakeFrog4Rankle1_00 would lose the _00
                            # put SD_ at the front of the capillary name
                            cap_name = 'SD_' + cap_name
                            if cap_name == bg_file.strip('.tiff'):
                                cap_image = cv2.imread(os.path.join(individual_caps_folder, cap_file), cv2.IMREAD_GRAYSCALE)

                                overlay= np.zeros_like(bg_image)
                                color = colors[i]
                                if i < len(predefined_colors) - 1:
                                    i += 1
                                else:
                                    i = 0
                                #add cap overlay
                                overlay = np.zeros((*bg_image.shape[:2], 4), dtype=np.uint8)
                                for y, x in np.ndindex(bg_image.shape[:2]):
                                    if cap_image[y][x] != 0:
                                        alpha = int(0.5 * cap_image[y][x])
                                        overlay[y, x] = color[0], color[1], color[2], alpha
                                overlay_rgb = overlay[..., :3]
                                overlay_alpha = overlay[..., 3] / 255.0
                                if bg_image.ndim == 2:
                                    bg_image = cv2.cvtColor(bg_image, cv2.COLOR_GRAY2BGR)   
                                # Apply the alpha blending
                                for c in range(3):  # Loop over the color channels
                                    bg_image[..., c] = (1.0 - overlay_alpha) * bg_image[..., c] + overlay_alpha * overlay_rgb[..., c]
                                bg_image = bg_image.astype(np.uint8)
                            
                    # bg_image = cv2.imread('path_to_background_image.png')
                    # cap_image = cv2.imread('path_to_capillary_image.png', cv2.IMREAD_GRAYSCALE)
                    # color = [255, 0, 0]  # Example color (red)
                            
                                overlay = overlay.astype(np.uint8)
                                overlayed = cv2.addWeighted(bg_image, 1, overlay, 1, 0)

                                #add label
                                xcoord, ycoord = get_label_position(cap_image)
                                capnum = cap_file.split('_')[-1]
                                cv2.putText(overlayed, capnum, (xcoord, ycoord), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 6, cv2.LINE_AA)

                                bg_image = overlayed
                            else:
                                continue
                    
                    #save
                    cv2.imwrite(os.path.join(overlays_folder, bg_file[:-5] + '_overlay.png'), bg_image)
                    colors = predefined_colors.copy()
                else:
                    continue
            else:
                continue


if __name__ == "__main__":
    ticks = time.time()
    umbrella_folder = '/hpc/projects/capillary-flow/frog/'
    for date in os.listdir(umbrella_folder):
        if not date.startswith('240729'):
            continue
        if date == 'archive':
            continue
        if date.endswith('alb'):
            continue
        for frog in os.listdir(os.path.join(umbrella_folder, date)):
            if frog.startswith('STD'):
                continue
            if not frog.startswith('Frog4'): # only process Frog4 for now
                continue
            for side in os.listdir(os.path.join(umbrella_folder, date, frog)):
                if side.startswith('STD'):
                    continue
                if not side.startswith('Left'): # only process the left side for now
                    continue
                if side == 'archive':
                    continue
                print('Processing: ' + date + ' ' + frog + ' ' + side)
                path = os.path.join(umbrella_folder, date, frog, side)
                main(path)
    #main(path = 'D:\\frog\\data\\240530\\Frog5\\Right')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))