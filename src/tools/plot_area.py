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
    return caps_listdir

def plot_area(caps_fp, centerlines_fp, metadata_fp):
    caps_listdir = exclude_fragmented(caps_fp)
    plotinfo = []
    
    #group coords files by cap
    pattern = r"cap_(\d{2})"
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

        #find shortest length
        for vid in cap: 
            #find vid & cap num
            vmatch = re.search(r'vid(\d{2})', vid)
            vidnum = vmatch.group(1)
            cmatch = re.search(r'cap_(.{3})', vid)
            capnum = cmatch.group(1)

            #find centerline file that corresponds to vidnum
            centerline_file = ""
            for ctrline in os.listdir(centerlines_fp):
                if vidnum in ctrline and capnum in ctrline:
                    centerline_file = os.path.join(centerlines_fp, ctrline)
                    break
            if centerline_file == "": continue #no centerline file for this vid

            #find shortest capillary length
            with open(centerline_file, 'r') as caps:
                reader = csv.reader(caps)
                rows = list(reader)
                top = float(rows[-1][0])
                bottom = float(max(rows, key=lambda row: float(row[0]))[0])
                if top == bottom:
                    top = float(rows[0][0])
                length = int(bottom - top)
                if length < shortest_length:
                    shortest_length = length
                    cap_index = counter
                counter += 1

        #get areas & pressure
        for vid in cap: 
            #find area
            vid_img = cv2.imread(os.path.join(caps_fp, vid), cv2.IMREAD_GRAYSCALE)
            area = 0
            found_bottom = False
            for row in range(vid_img.shape[0] - 1, 0, -1):
                for col in range(vid_img.shape[1]):
                    if vid_img[row][col] > 1:
                        found_bottom = True
                        for r in range(row, row - shortest_length, -1):
                            for c in range(vid_img.shape[1]):
                                if vid_img[r][c] > 1:
                                    area += 1
                    if found_bottom: break
                if found_bottom: break

            #find pressure
            vmatch = re.search(r'vid(\d{2})', vid)
            vidnum = vmatch.group(1)
            with open(metadata_fp, 'r') as metadata:
                reader = csv.reader(metadata)
                rows = list(reader)
                for row in rows:
                    if vidnum in str(row[3]):
                        pressure = str(row[5])
                        break

            cmatch = re.search(r'cap_(.{3})', vid)
            capnum = cmatch.group(1)
            plotinfo.append([area, pressure, "cap" + capnum[:-1]])

    #plot areas


            


def plot_area_by_length(caps_fp, centerlines_fp, metadata_fp):
    caps_listdir = exclude_fragmented(caps_fp)
    plotinfo = []
    for cap in os.listdir(caps_fp):
        cap_img = cv2.imread(os.path.join(caps_fp, cap), cv2.IMREAD_GRAYSCALE)
        vmatch = re.search(r'vid(\d{2})', cap)
        vidnum = vmatch.group(1)
        cmatch = re.search(r'cap_(.{3})', cap)
        capnum = cmatch.group(1)[:-1]

        area = 0
        for row in range(cap_img.shape[0]):
            for col in range(cap_img.shape[1]):
                if cap_img[row][col] > 1:
                    area += 1
        
        centerline_file = ""
        print(vidnum)
        print(capnum)
        for centerline in os.listdir(centerlines_fp):
            if "vid" + vidnum in centerline and "cap_" + capnum in centerline:
                centerline_file = centerline
                break
        if centerline_file == "": continue
        with open(os.path.join(centerlines_fp, centerline_file), 'r') as centerlines:
            reader = csv.reader(centerlines)
            rows = list(reader)
            length = len(rows)
        
        with open(metadata_fp, 'r') as metadata:
            reader = csv.reader(metadata)
            rows = list(reader)
            pressure = 0
            for row in rows:
                    if vidnum in str(row[3]):
                        pressure = str(row[5])
                        break

        plotinfo.append([area/length, pressure, "cap_" + capnum])
    #plot
    y_values = [point[0] for point in plotinfo]
    x_values = [float(point[1]) for point in plotinfo] 
    legend_labels = [point[2] for point in plotinfo]
    unique_legend_labels = list(set(legend_labels))
    unique_legend_labels.sort() 
    for label in unique_legend_labels:
        mask = [l == label for l in legend_labels]
        plt.scatter(np.array(x_values)[mask], np.array(y_values)[mask], label=label)
    plt.xlabel('Pressure (psi)')
    plt.ylabel('Area/Length')
    plt.legend(title='Legend') 
    plt.savefig(os.path.join(caps_fp, "plot.png"))
    plt.show()


        
    
    
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
    centerlines_fp = "E:\\Marcus\\gabby test data\\part11_centerlines_test\\coords2\\renamed"
    metadata_fp = "E:\\Marcus\\gabby test data\\part11\\part11_230427.xlsx - Sheet1.csv"

    #plot_area(caps_fp, centerlines_fp, metadata_fp)
    plot_area_by_length(caps_fp, centerlines_fp, metadata_fp)

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