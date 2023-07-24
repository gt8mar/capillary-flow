"""
Filename: translate_centerlines.py
-------------------------------------------------------------
This file translates centerlines based on the translations found in align_segmented and the crop values.
It also renames centerline files to the correct capillary number.
by: Gabby Rincon & ChatGPT
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label
from skimage.segmentation import find_boundaries
import csv
import re
import shutil
import random

#translates the y and x coordinates of centerlines by the translation values in ~/translations.csv and ~/crop_values.csv
def translate_coords(coords_fp, sorted_coords_listdir, translations_csv, crops_csv):
    #group coords files by video
    pattern = r"vid(\d{2})"
    groups = {}
    for string in sorted_coords_listdir:
        match = re.search(pattern, string)
        if match:
            number = match.group(1)
            groups.setdefault(number, []).append(string)
    grouped_coords_listdir = list(groups.values())

    #read translation csv
    with open(translations_csv, 'r') as translations_file:
        reader = csv.reader(translations_file)
        translations = []
        for row in reader:
            translations.append(row)

    #read crop csv
    with open(crops_csv, 'r') as crops_file:
        reader = csv.reader(crops_file)
        crops = []
        for row in reader:
            crops.append(row)

    translated_coords_fp = os.path.join(coords_fp, "translated")
    os.makedirs(translated_coords_fp, exist_ok=True)
    #apply translation
    for x in range(len(grouped_coords_listdir)):
        dy = int(float(translations[x][0])) - int(float(crops[x][0]))
        dx = int(float(translations[x][1])) - int(float(crops[x][3]))
        for file in grouped_coords_listdir[x]:
            with open(os.path.join(coords_fp, file), 'r') as orig_coords:
                reader = csv.reader(orig_coords)
                with open(os.path.join(translated_coords_fp, "translated_" + file), 'w', newline='') as translated_coords:
                    writer = csv.writer(translated_coords)
                    for row in reader:
                        xcol = float(row[0])
                        ycol = float(row[1])
                        translated_row = [xcol - dx, ycol - dy, *row[2:]]
                        writer.writerow(translated_row)

    return translated_coords_fp
"""
#this function renames the filenames of the centerline coordinates to the correct capillary name
def rename_caps(translated_coords_fp, projected_caps_fp):
    renamed_folder_fp = os.path.join(os.path.split(translated_coords_fp)[0], "renamed")
    os.makedirs(renamed_folder_fp, exist_ok=True)
    for file in os.listdir(translated_coords_fp):
        #get centerline midpoint x and y values
        with open(os.path.join(translated_coords_fp, file), 'r') as translated_coords:
            reader = csv.reader(translated_coords)
            rows = list(reader)
            midpoint_row = rows[len(rows) // 2]
            midpoint_x = midpoint_row[0]
            midpoint_y = midpoint_row[1]
        #check which capillary this file corresponds to
        for x in range(len(os.listdir(projected_caps_fp))):
            #if cap contains midpoint x, y
            image_array = cv2.imread(os.path.join(projected_caps_fp, os.listdir(projected_caps_fp)[x]))
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            midpoint_x_int = int(float(midpoint_x))
            midpoint_y_int = int(float(midpoint_y))
            if gray_image[midpoint_x_int][midpoint_y_int] > 0:
                new_csv_filename = file[:-6] + "cap_" + ("%02d" % x) + ".csv"
                shutil.copy(os.path.join(translated_coords_fp, file), os.path.join(renamed_folder_fp, new_csv_filename))
                break
    return renamed_folder_fp
"""

def rename_caps(coords_fp, individual_caps_fp):
    renamed_folder_fp = os.path.join(os.path.split(coords_fp)[0], "renamed")
    os.makedirs(renamed_folder_fp, exist_ok=True)
    for file in os.listdir(coords_fp):
        
        match = re.search(r'vid(\d{2})', file)
        vidnum = match.group(1)
        vids = [string for string in os.listdir(individual_caps_fp) if f"vid{vidnum}" in string]

        #get centerline midpoint x and y values
        with open(os.path.join(coords_fp, file), 'r') as coords:
            reader = csv.reader(coords)
            rows = list(reader)
            midpoint_row = rows[len(rows) // 2]
            midpoint_x = midpoint_row[0]
            midpoint_y = midpoint_row[1]
            
            for vid in vids:
                image_array = cv2.imread(os.path.join(individual_caps_fp, vid))
                gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                midpoint_x_int = int(float(midpoint_x))
                midpoint_y_int = int(float(midpoint_y))
                num_matches = 0
                if gray_image[midpoint_x_int][midpoint_y_int] > 0:
                    for row in rows:
                        if gray_image[int(float(row[0]))][int(float(row[1]))] > 0:
                            num_matches += 1
                    if num_matches > 0.8*len(rows):
                        new_csv_filename = file[:-6] + vid[-11:-4] + ".csv"
                        shutil.copy(os.path.join(coords_fp, file), os.path.join(renamed_folder_fp, new_csv_filename))
                        break

    return renamed_folder_fp

#TEMP
def show_centerlines(projected_caps_fp, coords_fp, individual_caps_fp):
    maxproj = np.zeros([1080,1440,3])
    for cap in os.listdir(projected_caps_fp):
        maxproj += cv2.imread(os.path.join(projected_caps_fp, cap))

    for file in os.listdir(coords_fp):
        with open(os.path.join(coords_fp, file), 'r') as coords:
            reader = csv.reader(coords)
            rows = list(reader)
            for row in rows:
                maxproj[int(float(row[0]))][int(float(row[1]))] = [255, 0, 0]
    cv2.imshow(str(file), maxproj)
    cv2.waitKey(0)

    for file in os.listdir(coords_fp):

        match1 = re.search(r'vid(\d{2})', file)
        vidnum = match1.group(1)

        match2 = re.search(r'cap_(.{3})', file)
        capnum = match2.group(1)

        cap_img = None
        for cap in os.listdir(individual_caps_fp):
            if cap.__contains__("vid" + vidnum) and cap.__contains__("cap_" + capnum):
                cap_img = cv2.imread(os.path.join(individual_caps_fp, cap))
                break

        with open(os.path.join(coords_fp, file), 'r') as coords:
            reader = csv.reader(coords)
            rows = list(reader)
            for row in rows:
                cap_img[int(float(row[0]))][int(float(row[1]))] = [255, 0, 0]
        cv2.imshow(str(file), cap_img)
        cv2.waitKey(0)  

"""def plot_radii(renamed_coords_fp):
    #group coords files by cap
    pattern = r"cap(\d{2})"
    groups = {}
    for string in os.listdir(renamed_coords_fp):
        match = re.search(pattern, string)
        if match:
            number = match.group(1)
            groups.setdefault(number, []).append(string)
    grouped_coords_listdir = list(groups.values())

    for cap in grouped_coords_listdir:
        fig, ax = plt.subplots()
        max_num_rows = 0 
        longest_cap_index = 0
        counter = 0
        #find longest cap
        for vid in cap:
            with open(os.path.join(renamed_coords_fp, vid), 'r') as coords:
                reader = csv.reader(coords)
                rows = list(reader)
                if len(rows) > max_num_rows:
                    max_num_rows = len(rows)
                    longest_cap_index = counter
                    max_yval_rownum = max(enumerate(rows), key=lambda row: float(row[1][0]))[0]
            counter += 1
        reference_xval = max_yval_rownum

        #plot longest cap
        with open(os.path.join(renamed_coords_fp, cap[longest_cap_index]), 'r') as coords:
                reader = csv.reader(coords)
                rows = list(reader)
                if rows[0][0] > rows[-1][0]:
                    rows.reverse()
                data = list(zip(*rows))
                radii = [float(y) for y in data[2]]
                x_values = np.arange(len(radii))
                ax.plot(x_values, radii)

        #plot all other caps
        for index, vid in enumerate(cap):
            if index != longest_cap_index:
                with open(os.path.join(renamed_coords_fp, vid), 'r') as coords:
                    reader = csv.reader(coords)
                    rows = list(reader)
                    if rows[0][0] > rows[-1][0]:
                        rows.reverse()
                    data = list(zip(*rows))
                    radii = [float(y) for y in data[2]]
                    #find max y value
                    max_yval_rownum = max(enumerate(rows), key=lambda row: float(row[1][0]))[0]
                    difference = reference_xval - max_yval_rownum
                    x_values = np.arange(len(radii)) + difference
                    ax.plot(x_values, radii)
        ax.set_title("cap " + str(cap[0])[-6:-4])
    plt.show()
"""


def main():
    coords_fp = "E:\\Marcus\\gabby test data\\part11_centerlines_test\\coords2"
    registered_folder = "E:\\Marcus\\gabby test data\\part11_segmented\\registered"

    sorted_coords_listdir = sorted_seg_listdir = sorted(filter(lambda x: os.path.isfile(os.path.join(coords_fp, x)), os.listdir(coords_fp))) #sort numerically
    translations_csv = os.path.join(registered_folder, "translations.csv")
    projected_caps_fp = os.path.join(registered_folder, "proj_caps")
    crops_csv = os.path.join(registered_folder, "crop_values.csv")
    individual_caps_fp = os.path.join(registered_folder, "individual_caps")

    translated_coords_fp = translate_coords(coords_fp, sorted_coords_listdir, translations_csv, crops_csv)
    
    #renamed_coords_fp = rename_caps(translated_coords_fp, projected_caps_fp)
    renamed_coords_fp = rename_caps(translated_coords_fp, individual_caps_fp)
    show_centerlines(projected_caps_fp, renamed_coords_fp, individual_caps_fp)
    #plot_radii(renamed_coords_fp)

    



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