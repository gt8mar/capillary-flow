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
import pandas as pd
import platform

def exclude_fragmented(caps_fp):
    caps_listdir = []
    for x in range(len(os.listdir(caps_fp))):
        cap = os.listdir(caps_fp)[x]
        if cap.endswith("a.png"):
            caps_listdir.append(cap)
        elif cap.endswith("b.png"):
            caps_listdir = caps_listdir[:-1]
    return caps_listdir

def exclude_bp_scan(caps_listdir, metadata_fp):
    new_caps_listdir = []
    for cap in caps_listdir:
        vmatch = re.search(r'vid(\d{2})', cap)
        vidnum = vmatch.group(1)
        metadata = pd.read_excel(metadata_fp)
        for index, row in metadata.iterrows():
            if vidnum in str(row[3]):
                if "bp" in str(row[3]) or "scan" in str(row[3]):
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
        capnum = int(entry[2])
        if capnum in grouped_caps:
            grouped_caps[capnum].append(entry)
        else:
            grouped_caps[capnum] = [entry]
    
    result_list = list(grouped_caps.values())
    return result_list   

def group_by_vidnum(plotinfo):
    grouped_caps = {}
    
    for entry in plotinfo:
        vidnum = int(entry[3])
        if vidnum in grouped_caps:
            grouped_caps[vidnum].append(entry)
        else:
            grouped_caps[vidnum] = [entry]
    
    result_list = list(grouped_caps.values())
    return result_list   

def subplots(plotinfo, partnum, location):
    grouped_plotinfo = group_by_cap(plotinfo)
    num_plots = len(grouped_plotinfo)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharey=True)
    fig.suptitle(partnum + " " + location + " capillary size vs. pressure")

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

            x_scatter = [float(entry[1]) for entry in sorted_cap]  
            y_scatter = [entry[0] for entry in sorted_cap]
            colors = [entry[4] for entry in sorted_cap]

            ax.scatter(x_scatter, y_scatter, c=colors)

            x_line = [float(entry[1]) for entry in cap]
            y_line = [entry[0] for entry in cap]

            colors = [entry[4] for entry in cap[1:]] + [cap[-1][4]]

            cmap = mcolors.ListedColormap(colors)

            for i in range(len(x_line) - 1):
                color = cmap(i)
                ax.plot([x_line[i], x_line[i + 1]], [y_line[i], y_line[i + 1]], c=color)


            ax.set_xlabel('Pressure (psi)')
            ax.set_ylabel('Area/Length')
            ax.set_title("cap" + cap[0][2])

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

    return plt


def boxplot(plotinfo):
    grouped_by_vidnum = group_by_vidnum(plotinfo)
    xvals = []
    yvals = []
    labels = []

    for vidnum in grouped_by_vidnum:
        labels.append(vidnum[0][1])
        xvals.append(float(vidnum[0][1]))
        yval = []
        for point in vidnum:
            yval.append(point[0])
        yvals.append(yval)

    xvals_up = xvals
    xvals_down = []
    yvals_up = yvals
    yvals_down = []
    for i in range(1,len(grouped_by_vidnum)):
        if grouped_by_vidnum[i][0][1] < grouped_by_vidnum[i-1][0][1]:
            xvals_up = xvals[:i]
            xvals_down = xvals[i:]
            yvals_up = yvals[:i]
            yvals_down = yvals[i:]
            break

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey = True)
    
    for i in range(len(yvals_up)):
        ax[0].scatter([xvals_up[i] for _ in range(len(yvals_up[i]))], yvals_up[i], c="Black")
    for i in range(len(yvals_down)):
        ax[1].scatter([xvals_down[i] for _ in range(len(yvals_down[i]))], yvals_down[i], c="Black")
    
    ax[0].boxplot(yvals_up, positions=xvals_up)
    ax[1].boxplot(yvals_down, positions=xvals_down)

    #ax.legend()
    ax[0].set_xlabel('Pressure (psi)')
    ax[0].set_ylabel('Area/Length')
    ax[0].set_title('Up')

    ax[1].set_xlabel('Pressure (psi)')
    ax[1].set_ylabel('Area/Length')
    ax[1].set_title('Down')
    plt.show()
    

def plot_area_by_length(caps_fp, centerlines_fp, metadata_fp):
    caps_listdir_nofrag = exclude_fragmented(caps_fp)
    caps_listdir_nobp = exclude_bp_scan(caps_listdir_nofrag, metadata_fp)

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
        pressure = 0
        metadata = pd.read_excel(metadata_fp)
        for index, row in metadata.iterrows():
            if vidnum in str(row[3]):
                pressure = str(row[5])
                break

        plotinfo.append([area/length, pressure, capnum, vidnum])

    partnum = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(caps_fp)))))
    location = os.path.basename(os.path.dirname(os.path.dirname(caps_fp)))
    subplots(plotinfo, partnum, location)
    #boxplot(plotinfo)
    
def main(path="E:\\Marcus\\gabby_test_data\\part09\\230414\\loc01"):
    participant = os.path.basename(os.path.dirname(os.path.dirname(path)))
    date = os.path.basename(os.path.dirname(path))
    location = os.path.basename(path)
    
    caps_fp = os.path.join(path, "segmented", "individual_caps_translated")
    centerlines_fp = os.path.join(path, "centerlines", "renamed")
    metadata_fp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path)))), "metadata", participant + "_" + date + ".xlsx")

    size_plot = plot_area_by_length(caps_fp, centerlines_fp, metadata_fp)

    #save to folder in data/part/date/loc/segmented/plots
    filename = "set_01_" + participant + "_" + date + "_" + location + "_size_v_pressure.png"
    plot_fp = os.path.join(path, "size", "plots")
    os.makedirs(plot_fp, exist_ok=True)
    size_plot.savefig(os.path.join(plot_fp, filename))
    #save to folder in results
    if platform.system() != 'Windows':
        size_results_fp = "/hpc/projects/capillary-flow/results/size"
        os.makedirs(size_results_fp, exist_ok=True)
        size_plot.savefig(os.path.join(size_results_fp, filename))


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