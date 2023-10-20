"""
Filename: plot_area.py
-------------------------------------------------------------
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
from itertools import chain
if platform.system() != 'Windows':
    from src.tools.get_directory import get_directory_at_level
else:
    from get_directory import get_directory_at_level

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

        vidrow = None
        for index, vid_entry in enumerate(metadata['Video']):
            if vidnum in vid_entry:
                vidrow = index
                break
        if 'bp' in str(vidrow) or 'scan' in str(vidrow):
            pass
        else:
            new_caps_listdir.append(cap)

    return new_caps_listdir

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

def save_plotinfo(path, plotinfo, participant, date, location):
    # complete_plotinfo (grouped by capnum and sorted by vidnum) is in the format:
    # [[participant, date, location, area, pressure, capnum, vidnum], ...]
    grouped_plotinfo = group_by_cap(plotinfo)
    complete_plotinfo = []
    complete_plotinfo.append(["participant", "date", "location", "area", "pressure", "capnum", "vidnum"])
    for cap in grouped_plotinfo:
        sorted_cap = sorted(cap, key=lambda x: x[3]) #sort by vidnum
        for entry in sorted_cap:
            complete_plotinfo.append([participant, date, location] + entry)

    df = pd.DataFrame(complete_plotinfo)
    #save to results folder
    filename = participant + '_' + date + '_' + location + '_size_data.csv'
    if platform.system() != 'Windows':
        results_fp = '/hpc/projects/capillary-flow/results/size/size_data'
        os.makedirs(results_fp, exist_ok=True)
        df.to_csv(os.path.join(results_fp, filename), header=True, index=False)
    #save to location folder
    plotinfo_fp = os.path.join(path, "size", "size_data")
    os.makedirs(plotinfo_fp, exist_ok=True)
    df.to_csv(os.path.join(plotinfo_fp, filename), header=False, index=False)

def make_subplots(plotinfo, participant, date, location):
    grouped_plotinfo = group_by_cap(plotinfo)
    num_plots = len(grouped_plotinfo)
    num_cols = min(num_plots, 3) # Max 3 columns
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.suptitle(participant + " " + location + " capillary size vs. pressure")

    slope_data = []

    # Find overall min and max x-values for all subplots
    overall_min_x = float('inf')
    overall_max_x = float('-inf')
    overall_min_y = float('inf')
    overall_max_y = float('-inf')

    if num_plots == 1:
        ax = axes  # If only one cap, use a single axis
    else:
        ax = axes.flat # Otherwise, use a flat array of axes

    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            cap = grouped_plotinfo[i]
            cap = sorted(cap, key=lambda x: x[3]) #sort by vidnum

            # find index where pressure starts decreasing
            max_index = len(cap)
            for i in range(1, len(cap)):
                if cap[i][1] < cap[i - 1][1]:
                    max_index = i
                    break

            increasing_cap = cap[:max_index]
            decreasing_cap = cap[max_index - 1:]

            sorted_inc_cap = sorted(increasing_cap, key=lambda x: x[1])
            sorted_dec_cap = sorted(decreasing_cap, key=lambda x: x[1])

            x_scatter_inc = [float(entry[1]) for entry in sorted_inc_cap]  
            y_scatter_inc = [float(entry[0]) for entry in sorted_inc_cap]

            x_scatter_dec = [float(entry[1]) for entry in sorted_dec_cap]  
            y_scatter_dec = [float(entry[0]) for entry in sorted_dec_cap]

            # Plot scatter points
            ax.scatter(x_scatter_inc, y_scatter_inc, c="Black")
            ax.scatter(x_scatter_dec, y_scatter_dec, c="Black")

            # Plot lines: blue for increasing pressure, red for decreasing
            x_line_inc = [float(entry[1]) for entry in increasing_cap]
            y_line_inc = [entry[0] for entry in increasing_cap]
            for k in range(len(x_line_inc) - 1):
                ax.plot([x_line_inc[k], x_line_inc[k + 1]], [y_line_inc[k], y_line_inc[k + 1]], c="Blue")

            x_line_dec = [float(entry[1]) for entry in decreasing_cap]
            y_line_dec = [entry[0] for entry in decreasing_cap]
            for k in range(len(x_line_dec) - 1):
                ax.plot([x_line_dec[k], x_line_dec[k + 1]], [y_line_dec[k], y_line_dec[k + 1]], c="Red")

            # Calculate slopes
            if len(x_line_inc) > 1:
                inc_slope, _ = np.polyfit(x_line_inc, y_line_inc, 1)
            else:
                inc_slope = ""
            line_name_inc = "inc_" + participant + "_" + date + "_" + location + "_cap" + cap[0][2] 
            if len(x_line_dec) > 1:
                dec_slope, _ = np.polyfit(x_line_dec, y_line_dec, 1)
            else:
                dec_slope = ""
            line_name_dec = "dec_" + participant + "_" + date + "_" + location + "_cap" + cap[0][2] 
            slope_data.append([line_name_inc, inc_slope])
            slope_data.append([line_name_dec, dec_slope])

            ax.set_xlabel('Pressure (psi)')
            #ax.set_ylabel('Area/Length')
            ax.set_ylabel('Area')
            ax.set_title(participant + " cap" + cap[0][2])

            # Update overall min and max x-values
            overall_min_x = min(chain([overall_min_x], x_scatter_inc, x_scatter_dec))
            overall_max_x = max(chain([overall_max_x], x_scatter_inc, x_scatter_dec))
            overall_min_y = min(chain([overall_min_y], y_scatter_inc, y_scatter_dec))
            overall_max_y = max(chain([overall_max_y], y_scatter_inc, y_scatter_dec))
        else:
            ax.axis('off')

        # Set the same x-axis limits for all subplots
        for ax in axes.flat[:num_plots]:
            ax.set_xlim(overall_min_x, overall_max_x)
            ax.set_ylim(overall_min_y, overall_max_y)

    plt.subplots_adjust(hspace=0.5)
    #plt.show()
    return fig, slope_data 

def get_plotinfo(caps_fp, centerlines_fp, metadata_fp):
    caps_listdir_nofrag = exclude_fragmented(caps_fp)
    caps_listdir_nobp = exclude_bp_scan(caps_listdir_nofrag, metadata_fp)

    plotinfo = []
    for cap in caps_listdir_nobp:
        cap_img = cv2.imread(os.path.join(caps_fp, cap), cv2.IMREAD_GRAYSCALE)

        #get vidnum and capnum
        vmatch = re.search(r'vid(\d{2})', cap)
        vidnum = vmatch.group(1)
        cmatch = re.search(r'cap_(.{3})', cap)
        capnum = cmatch.group(1)[:-1]

        #get area
        area = np.sum(cap_img > 1)
        
        #get length
        centerline_file = ""
        for centerline in os.listdir(centerlines_fp): #find centerline file for this cap
            if "vid" + vidnum in centerline and "cap_" + capnum in centerline:
                centerline_file = centerline
                break
        if centerline_file == "": continue #skip cap if no centerline file found
        df = pd.read_csv(os.path.join(centerlines_fp, centerline_file))
        length = len(df)
        if length < 100: #skip cap if centerline is too short
            continue
        
        #get pressure
        metadata = pd.read_excel(metadata_fp)
        vidrow = None
        for index, vid_entry in enumerate(metadata['Video']):
            if vidnum in str(vid_entry):
                vidrow = index
                break
        pressure = metadata.iloc[vidrow]['Pressure']

        #plotinfo.append([area/length, pressure, capnum, vidnum])
        plotinfo.append([area, pressure, capnum, vidnum])

    return plotinfo
    
def main(path="C:\\Users\\Luke\\Documents\\capillary-flow\\data\\part10\\230425\\loc01"):
    participant = get_directory_at_level(path, 2)
    date = get_directory_at_level(path, 1)
    location = get_directory_at_level(path, 0)
    
    caps_fp = os.path.join(path, "segmented", "hasty", "individual_caps_translated")
    centerlines_fp = os.path.join(path, "centerlines", "renamed")
    metadata_fp = os.path.join(get_directory_at_level(path, 4, only_dir=False), "metadata", participant + "_" + date + ".xlsx")

    plotinfo = get_plotinfo(caps_fp, centerlines_fp, metadata_fp)
    save_plotinfo(path, plotinfo, participant, date, location)
    size_plot, slopes = make_subplots(plotinfo, participant, date, location)

    #save to plot folder in data/part/date/loc/size/plots
    filename = "set_01_" + participant + "_" + date + "_" + location + "_size_v_pressure.png"
    plot_fp = os.path.join(path, "size", "plots")
    os.makedirs(plot_fp, exist_ok=True)
    size_plot.savefig(os.path.join(plot_fp, filename))

    #save plot to folder in results
    if platform.system() != 'Windows':
        size_results_fp = "/hpc/projects/capillary-flow/results/size/plots"
        os.makedirs(size_results_fp, exist_ok=True)
        size_plot.savefig(os.path.join(size_results_fp, filename))

    #save slopes to data/part/date/loc/size/slopes.csv
    slopes_fp = os.path.join(path, "size", "slopes.csv")
    df = pd.DataFrame(slopes)
    df.to_csv(slopes_fp, header=False, index=False)
        
    #save slopes to folder in results
    if platform.system() != 'Windows':
        slope_results_fp = "/hpc/projects/capillary-flow/results/size/slopes.csv"
        with open(slope_results_fp, 'a') as slopes_file: #append to previous slopes file
            writer = csv.writer(slopes_file)
            for row in slopes:
                writer.writerow(row)


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