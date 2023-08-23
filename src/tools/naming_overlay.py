import time
import os
import re
import csv
import cv2
import numpy as np
from skimage.color import rgb2gray
import platform

def get_label_position(input_array):
    # Find the indices of non-zero elements in the array
    non_zero_indices = np.argwhere(input_array != 0)
    
    # Calculate the minimum and maximum x and y values of non-zero elements
    min_x = np.min(non_zero_indices[:, 1])
    max_x = np.max(non_zero_indices[:, 1])
    min_y = np.min(non_zero_indices[:, 0])
    max_y = np.max(non_zero_indices[:, 0])
    
    # Define the edge margin
    edge_margin = 30
    
    # Check if minimum and maximum coordinates are within bounds
    if min_x > edge_margin:
        x_coord = min_x
    else:
        x_coord = max_x
    
    if min_y > edge_margin:
        y_coord = min_y
    else:
        y_coord = max_y
    
    return x_coord, y_coord

def rename_files(directory_path):
    # Get a list of files in the directory
    file_list = os.listdir(directory_path)

    # Iterate through each file in the directory
    for filename in file_list:
        # Search for a number in the filename
        match = re.search(r'vid(\d+)', filename)
        
        # Check if a match was found and the number is 1 or 2 digits
        if match and len(match.group(1)) <= 2:
            num = int(match.group(1))
            new_filename = filename.replace(match.group(0), f'vid{num:02}')

            # Construct the full paths for the old and new filenames
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)

            # Rename the file
            os.rename(old_path, new_path)

def extract_file_info(filename):
    set_part_date = filename[:20] #with trailing underscore
    lmatch = re.search(r'loc(\d{2})', filename)
    location = "" if lmatch == None else "loc" + lmatch.group(1) + "_"
    vmatch = re.search(r'vid(\d{2})', filename)
    vid = "" if vmatch == None else "vid" + vmatch.group(1) + "_"
    return set_part_date, location, vid

def make_overlays(path="E:\\Marcus\\gabby_test_data\\part09\\230414\\loc01"):
    reg_moco_fp = os.path.join(path, "segmented", "moco_registered")

    resize_csv = os.path.join(path, "segmented", "resize_vals.csv")
    with open(resize_csv, 'r') as resize_values:
        reader = csv.reader(resize_values)
        rows = list(reader)
        minx = abs(int(rows[0][0]))
        maxx = abs(int(rows[0][1]))
        miny = abs(int(rows[0][2]))
        maxy = abs(int(rows[0][3]))
        #minx = None if minx == 0 else -minx
        #maxx = None if maxx == 0 else maxx
        #miny = None if miny == 0 else -miny
        #maxy = None if maxy == 0 else maxy

        predefined_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Maroon
            (0, 128, 0),    # Green (different shade)
            (0, 0, 128),    # Navy
            (128, 128, 0),  # Olive
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (255, 165, 0),  # Orange
            (139, 69, 19),  # Saddle Brown
            (0, 128, 128),  # Teal (different shade)
        ]
    element_colors = {}
    colored_elements = []
    rename_files(reg_moco_fp)
    for frame in os.listdir(reg_moco_fp):
        vmatch = re.search(r'vid(\d{2})', frame)
        vidnum = vmatch.group(1)
        frame_img = cv2.imread(os.path.join(reg_moco_fp, frame))
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2BGRA)
        frame_img[:, :, 3] = 255
        for cap in os.listdir(os.path.join(path, "segmented", "individual_caps_translated")):
            if "vid" + vidnum in cap:
                cmatch = re.search(r'cap_(\d{2})', cap)
                capnum = "cap_" + cmatch.group(1)
                if capnum in element_colors:
                    color = element_colors[capnum]
                else:
                    if len(predefined_colors) > 0:
                        color = predefined_colors.pop(0)
                    else:
                        color = (255, 255, 255) 
                    element_colors[capnum] = color
                colored_elements.append((capnum, color))

                cap_img = cv2.imread(os.path.join(path, "segmented", "individual_caps_translated", cap))
                cap_img = rgb2gray(cap_img)
                print("cap_img size: " + str(cap_img.shape))
                if miny==0 and minx==0:
                    resized_cap = cap_img[maxy:, maxx:]
                elif miny==0:
                    resized_cap = cap_img[maxy:, maxx:minx]
                elif minx==0:
                    resized_cap = cap_img[maxy:miny, maxx:]
                else:
                    resized_cap = cap_img[maxy:miny, maxx:minx]
                print("resized_cap_size: " + str(resized_cap.shape))
                #get label coordinates
                xcoord, ycoord = get_label_position(resized_cap)

                height, width = len(resized_cap), len(resized_cap[0])
                
                overlay = np.zeros_like(frame_img)
                for y in range(height):
                    for x in range(width):
                        if resized_cap[y][x] != 0:
                            alpha = int(0.5 * resized_cap[y][x])
                            overlay[y, x] = [color[0], color[1], color[2], alpha]
                                
                overlay = overlay.astype(np.uint8)
                overlayed = cv2.addWeighted(frame_img, 1, overlay, 1, 0)
                cv2.putText(overlayed, capnum, (xcoord, ycoord), cv2.FONT_HERSHEY_PLAIN, 2, color, 2, cv2.FILLED)

                set_part_date, location, vid = extract_file_info(cap)
                filename = set_part_date + location + vid + "overlay.png"
                frame_img = overlayed
                overlay_folder = os.path.join(path, "segmented", "overlays")
                os.makedirs(overlay_folder, exist_ok=True)
                cv2.imwrite(os.path.join(overlay_folder, filename), overlayed)
                if platform.system() != 'Windows':
                    overlays_fp = '/hpc/projects/capillary-flow/results/size/overlays'
                    os.makedirs(overlays_fp, exist_ok=True)
                    cv2.imwrite(os.path.join(overlays_fp, filename), overlayed)


if __name__ == "__main__":
    ticks = time.time()
    make_overlays()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))