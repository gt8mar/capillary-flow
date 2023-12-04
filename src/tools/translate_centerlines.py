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
import platform
import pandas as pd
if platform.system() != 'Windows':
    from src.tools.get_directory import get_directory_at_level
else:
    from get_directory import get_directory_at_level

#translates the y and x coordinates of centerlines by the translation values in ~/translations.csv and ~/crop_values.csv
def translate_coords(coords_fp, sorted_coords_listdir, translations_csv, crops_csv, resize_csv):
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
    with open(translations_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        data = [row for row in csvreader]
        translations = np.array(data).astype(float)

    #read crop csv
    with open(crops_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        crops = list(csvreader)

    #read resize csv
    with open(resize_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        data = [row for row in csvreader]
        minx = float(data[0][0])
        maxx = float(data[0][1])
        miny = float(data[0][2])
        maxy = float(data[0][3])

    translated_coords_fp = os.path.join(os.path.dirname(coords_fp), "translated")
    os.makedirs(translated_coords_fp, exist_ok=True)
    #apply translation
    for x in range(len(grouped_coords_listdir)):
        dy = int(float(translations[x][0])) - int(float(crops[x][0]) + int(maxx))
        dx = int(float(translations[x][1])) - int(float(crops[x][3]) + int(maxy))
        for file in grouped_coords_listdir[x]:
            orig_coords_path = os.path.join(coords_fp, file)
            translated_coords_path = os.path.join(translated_coords_fp, "translated_" + file)

            orig_df = pd.read_csv(orig_coords_path, header=None)

            #translate
            orig_df.iloc[:, 0] = orig_df.iloc[:, 0] - dx
            orig_df.iloc[:, 1] = orig_df.iloc[:, 1] - dy

            #save
            orig_df.to_csv(translated_coords_path, index=False, header=False, float_format='%.6f')

    return translated_coords_fp

def rename_caps(coords_fp, individual_caps_fp, participant, date, location):
    names = []
    renamed_folder_fp = os.path.join(os.path.split(coords_fp)[0], "renamed")
    os.makedirs(renamed_folder_fp, exist_ok=True)
    for file in os.listdir(coords_fp):        
        match = re.search(r'vid(\d{2})', file)
        vidnum = match.group(1)
        vids = [string for string in os.listdir(individual_caps_fp) if f"vid{vidnum}" in string]

        #get centerline midpoint x and y values
        coords_df = pd.read_csv(os.path.join(coords_fp, file), header=None)
        rows = coords_df.to_numpy()
        midpoint_index = len(rows) // 2
        midpoint_x, midpoint_y = rows[midpoint_index][:2]

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
                #if 80% of the centerline matches the capillary, rename the centerline file
                if num_matches > 0.8*len(rows): 
                    new_csv_filename = file[:-6] + vid[-11:-4] + ".csv"
                    shutil.copy(os.path.join(coords_fp, file), os.path.join(renamed_folder_fp, new_csv_filename))
                    names.append([file, new_csv_filename])
                    break

    #save old to new name map
    map_fp = os.path.join(os.path.dirname(coords_fp), "name_map.csv")
    with open(map_fp, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(names)

    if platform.system() != 'Windows':
        os.makedirs('/hpc/projects/capillary-flow/results/size/name_maps', exist_ok=True)
        map_fp = os.path.join('/hpc/projects/capillary-flow/results/size/name_maps', participant + "_" + date + "_" + location + "_name_map.csv")
        with open(map_fp, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(names)
    
    return renamed_folder_fp

def show_centerlines(projected_caps_fp, coords_fp, individual_caps_fp, registered_caps_fp):
    y, x, _ = cv2.imread(os.path.join(projected_caps_fp, os.listdir(projected_caps_fp)[0])).shape
    maxproj = np.zeros([y,x,3])
    for cap in os.listdir(projected_caps_fp):
        maxproj += cv2.imread(os.path.join(projected_caps_fp, cap))

    for file in os.listdir(coords_fp):
        with open(os.path.join(coords_fp, file), 'r') as coords:
            reader = csv.reader(coords)
            rows = list(reader)
            for row in rows:
                maxproj[int(float(row[0]))][int(float(row[1]))] = [255, 0, 0]
    #cv2.imshow(str(file), maxproj)
    #cv2.waitKey(0)

    for vid in os.listdir(registered_caps_fp):
        vid_img = cv2.imread(os.path.join(registered_caps_fp, vid))
        vidnum = re.search(r'vid(\d{2})', vid).group(1)
        for file in os.listdir(coords_fp):
            ctrl_vidnum = re.search(r'vid(\d{2})', file).group(1)
            if vidnum == ctrl_vidnum:
                with open(os.path.join(coords_fp, file), 'r') as coords:
                    reader = csv.reader(coords)
                    rows = list(reader)
                    for row in rows:
                        vid_img[int(float(row[0]))][int(float(row[1]))] = [255, 0, 0]
            else:
                continue
        #cv2.imshow(str(vid), vid_img)
        #cv2.waitKey(0)
        os.makedirs(os.path.join(os.path.dirname(coords_fp), 'centerline_images'), exist_ok=True)
        cv2.imwrite(os.path.join(os.path.dirname(coords_fp), 'centerline_images', vid), vid_img)

    """for file in os.listdir(coords_fp):
        match1 = re.search(r'vid(\d{2})', file)
        vidnum = match1.group(1)

        match2 = re.search(r'cap_(.{3})', file)
        capnum = match2.group(1)

        cap_img = None
        for cap in os.listdir(individual_caps_fp):
            if cap.__contains__("vid" + vidnum) and cap.__contains__("cap_" + capnum):
                cap_img = cv2.imread(os.path.join(individual_caps_fp, cap))
                break
        if cap_img is None: 
            continue
        with open(os.path.join(coords_fp, file), 'r') as coords:
            reader = csv.reader(coords)
            rows = list(reader)
            for row in rows:
                cap_img[int(float(row[0]))][int(float(row[1]))] = [255, 0, 0]

        cv2.imshow(str(file), cap_img)
        cv2.waitKey(0)  """

def main(path="C:\\Users\\Luke\\Documents\\capillary-flow\\temp\\part19\\230503\\loc01"):
    coords_fp = os.path.join(path, "centerlines", "coords")
    segmented_folder = os.path.join(path, "segmented", "hasty")

    sorted_coords_listdir = sorted(filter(lambda x: os.path.isfile(os.path.join(coords_fp, x)), os.listdir(coords_fp))) #sort numerically
    
    translations_csv = os.path.join(segmented_folder, "translations.csv")
    crops_csv = os.path.join(segmented_folder, "crop_values.csv")
    resize_csv = os.path.join(segmented_folder, "resize_vals.csv")
    individual_caps_fp = os.path.join(segmented_folder, "individual_caps_translated")

    participant = get_directory_at_level(path, 2)
    date = get_directory_at_level(path, 1)
    location = get_directory_at_level(path, 0)

    translated_coords_fp = translate_coords(coords_fp, sorted_coords_listdir, translations_csv, crops_csv, resize_csv)
    renamed_coords_fp = rename_caps(translated_coords_fp, individual_caps_fp, participant, date, location)
    show_centerlines(os.path.join(segmented_folder, "proj_caps"), translated_coords_fp, individual_caps_fp, os.path.join(segmented_folder, "registered"))
    
    
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