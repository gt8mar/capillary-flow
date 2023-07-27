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
import matplotlib.colors as mcolors

def exclude_fragmented(caps_fp):
    caps_listdir = []
    for x in range(len(os.listdir(caps_fp))):
        cap = os.listdir(caps_fp)[x]
        if cap.endswith("a.png"):
            caps_listdir.append(cap)
        elif cap.endswith("b.png"):
            caps_listdir = caps_listdir[:-1]
    return caps_listdir

def exclude_bp(caps_listdir, metadata_fp):
    new_caps_listdir = []
    for cap in caps_listdir:
        vmatch = re.search(r'vid(\d{2})', cap)
        vidnum = vmatch.group(1)
        with open(metadata_fp, 'r') as metadata:
            reader = csv.reader(metadata)
            rows = list(reader)
            for row in rows:
                if vidnum in str(row[3]):
                    if "bp" in str(row[3]):
                        break
                    else:
                        new_caps_listdir.append(cap)
                        break
    return new_caps_listdir

def exclude_twisty():
    #TOWRITE
    return 0

def group_by_cap(plotinfo):
    grouped_caps = {}
    
    for entry in plotinfo:
        capnum = entry[2]
        if capnum in grouped_caps:
            grouped_caps[capnum].append(entry)
        else:
            grouped_caps[capnum] = [entry]
    
    result_list = list(grouped_caps.values())
    return result_list   

"""def plot(plotinfo):
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
    plt.show()"""

"""def subplot(plotinfo, title):
    for point in plotinfo:
        point.append("Blue")
    for x in range(len(plotinfo)-1):
        if float(plotinfo[x][1]) > float(plotinfo[x+1][1]):
            plotinfo[x+1][4] = "Red"
    sorted_plotinfo = sorted(plotinfo, key=lambda x: x[1])
    x_line = [entry[1] for entry in plotinfo]
    y_line = [entry[0] for entry in plotinfo]

    # Create a figure and an array of subplots
    fig, ax = plt.subplots()

    if "Red" in [entry[4] for entry in sorted_plotinfo]:
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'red'], N=len(x_line) - 1)
        for i in range(len(x_line) - 1):
            ax.plot([x_line[i], x_line[i + 1]], [y_line[i], y_line[i + 1]], c=cmap(i))
    else:
        ax.plot(x_line, y_line)

    ax.scatter([entry[1] for entry in sorted_plotinfo], [entry[0] for entry in sorted_plotinfo], c=[entry[4] for entry in sorted_plotinfo])
    ax.set_xlabel('Pressure (psi)')
    ax.set_ylabel('Area/Length')
    ax.set_title(title)

    plt.show()"""

def plot_subplots(plotinfo):
    grouped_plotinfo = group_by_cap(plotinfo)
    num_plots = len(grouped_plotinfo)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharey=True)
    fig.suptitle("Capillaries idk")

    # Find overall min and max x-values for all subplots
    overall_min_x = float('inf')
    overall_max_x = float('-inf')

    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            cap = grouped_plotinfo[i]

            for point in cap:
                point.append("Blue")
            for x in range(len(cap) - 1):
                if float(cap[x][1]) > float(cap[x + 1][1]):
                    cap[x + 1][4] = "Red"

            sorted_cap = sorted(cap, key=lambda x: x[1])

            x_scatter = [float(entry[1]) for entry in sorted_cap]  # Convert to floats
            y_scatter = [entry[0] for entry in sorted_cap]
            colors = [entry[4] for entry in sorted_cap]

            ax.scatter(x_scatter, y_scatter, c=colors)

            x_line = [float(entry[1]) for entry in cap]  # Convert to floats
            y_line = [entry[0] for entry in cap]
            if "Red" in [entry[4] for entry in sorted_cap]:
                cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'red'], N=len(x_line) - 1)
                for i in range(len(x_line) - 1):
                    ax.plot([x_line[i], x_line[i + 1]], [y_line[i], y_line[i + 1]], c=cmap(i))
            else:
                ax.plot(x_line, y_line)

            ax.set_xlabel('Pressure (psi)')
            ax.set_ylabel('Area/Length')
            ax.set_title(cap[0][2])

            # Update overall min and max x-values
            overall_min_x = min(overall_min_x, min(x_scatter))
            overall_max_x = max(overall_max_x, max(x_scatter))

        else:
            ax.axis('off')

    # Set the same x-axis limits for all subplots
    for ax in axes.flat[:num_plots]:
        ax.set_xlim(overall_min_x, overall_max_x)

    plt.tight_layout()
    plt.show()

#not using this anymore
#only can compare areas between one capillary
def plot_area(caps_fp, centerlines_fp, metadata_fp):
    caps_listdir_nofrag = exclude_fragmented(caps_fp)
    
    plotinfo = []
    
    #group coords files by cap
    pattern = r"cap_(\d{2})"
    groups = {}
    for string in caps_listdir_nofrag:
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
    caps_listdir_nofrag = exclude_fragmented(caps_fp)
    caps_listdir_nobp = exclude_bp(caps_listdir_nofrag, metadata_fp)

    plotinfo = []

    for cap in caps_listdir_nobp:
        cap_img = cv2.imread(os.path.join(caps_fp, cap), cv2.IMREAD_GRAYSCALE)
        vmatch = re.search(r'vid(\d{2})', cap)
        vidnum = vmatch.group(1)
        cmatch = re.search(r'cap_(.{3})', cap)
        capnum = cmatch.group(1)[:-1]

        #get area
        area = 0
        for row in range(cap_img.shape[0]):
            for col in range(cap_img.shape[1]):
                if cap_img[row][col] > 1:
                    area += 1
        
        #get length
        centerline_file = ""
        for centerline in os.listdir(centerlines_fp):
            if "vid" + vidnum in centerline and "cap_" + capnum in centerline:
                centerline_file = centerline
                break
        if centerline_file == "": continue
        with open(os.path.join(centerlines_fp, centerline_file), 'r') as centerlines:
            reader = csv.reader(centerlines)
            rows = list(reader)
            length = len(rows)
        
        #get pressure
        with open(metadata_fp, 'r') as metadata:
            reader = csv.reader(metadata)
            rows = list(reader)
            pressure = 0
            for row in rows:
                    if vidnum in str(row[3]):
                        pressure = str(row[5])
                        break

        plotinfo.append([area/length, pressure, "cap_" + capnum, vidnum])

    #split plotinfo
    """grouped_plotinfo = group_by_cap(plotinfo)
    for cap in grouped_plotinfo:
        plot(cap, cap[0][2])
    plot(plotinfo)"""
    plot_subplots(plotinfo)

    
def main():
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