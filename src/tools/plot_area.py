"""
Filename: plot_radii.py
-------------------------------------------------------------
This file 
by: Gabby Rincon 
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import re


def exclude_fragmented(caps_fp):
    caps_listdir = []
    for x in range(len(os.listdir(caps_fp))):
        cap = os.listdir(caps_fp)[x]
        if cap.endswith("a.png"):
            caps_listdir.append(cap)
        elif cap.endswith("b.png"):
            caps_listdir = caps_listdir[:-1]
    print(caps_listdir)
    return caps_listdir

def plot_area(caps_fp):
    caps_listdir = exclude_fragmented(caps_fp)
    
    #group coords files by cap
    pattern = r"cap(\d{2})"
    groups = {}
    for string in caps_listdir:
        match = re.search(pattern, string)
        if match:
            number = match.group(1)
            groups.setdefault(number, []).append(string)
    grouped_caps_listdir = list(groups.values())

    for cap in grouped_caps_listdir:
        shortest_length = np.inf
        cap_index = 0
        counter = 0
        for vid in cap:
            with open(os.path.join(caps_fp, vid), 'r') as caps:
                reader = csv.reader(caps)
                rows = list(reader)
                top = float(rows[-1][0])
                bottom = float(max(rows, key=lambda row: float(row[0]))[0])
                length = int(top - bottom)
                if length < shortest_length:
                    shortest_length = length
                    cap_index = counter
                counter += 1
        for vid in cap:
            with open(os.path.join(coords_fp, vid), 'r') as coords:
                reader = csv.reader(coords)
                rows = list(reader)
                bottom_index = int(max(enumerate(rows), key=lambda row: float(row[1][0]))[0])
                top_index = int(bottom - shortest_length)
            image = cv2.imread(os.path.join(coords_fp, vid))
            for row in range(top_index, bottom_index):
                for col in range(image.shape[1]):
                    image[row][col] = (255, 0, 0)
            cv2.imshow(vid, image)
            cv2.waitKey(0)


def plot_area_by_length():
    return 0
    
def main():
    """coords_fp = "E:\\Marcus\\gabby test data\\part11_centerlines_test\\coords"
    registered_folder = "E:\\Marcus\\gabby test data\\part11_segmented\\registered"

    sorted_coords_listdir = sorted_seg_listdir = sorted(filter(lambda x: os.path.isfile(os.path.join(coords_fp, x)), os.listdir(coords_fp))) #sort numerically
    translations_csv = os.path.join(registered_folder, "translations.csv")
    projected_caps_fp = os.path.join(registered_folder, "proj_caps")
    crops_csv = os.path.join(registered_folder, "crop_values.csv")
    
    translated_coords_fp = translate_coords(coords_fp, sorted_coords_listdir, translations_csv, crops_csv)
    renamed_coords_fp = rename_caps(translated_coords_fp, projected_caps_fp)
    """
    caps_fp = "E:\\Marcus\\gabby test data\\part11_segmented\\registered\\individual_caps"

    plot_area(caps_fp)


"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))