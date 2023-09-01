"""
Filename: find_radii.py
-----------------------
# TODO fix this
This is broken. 

By: Marcus Forst
"""

import os, time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.tools.load_csv_list import load_csv_list
from src.tools.parse_path import parse_vid_path


PIX_UM = 1.74

def plot_box_swarm(data, x_labels, y_axis_label,  plot_title, figure_name, verbose = True, write = False, remove_outliers = False):
    """Plot box-plot and swarm plot for data list.
 
    Args:
        data (list of list): List of lists with data to be plotted.
        y_axis_label (str): Y- axis label.
        x_labels (list of str): List with labels of x-axis.
        plot_title (str): Plot title.
        figure_name (str): Path to output figure.
         
    """
    if remove_outliers:
        # # Remove outliers using the clip function
        # data_clipped, outliers = select_outliers(data)
        # # initialize plot information:
        # sns.set(color_codes=True)
        # plt.figure(1, figsize=(9, 6))
        # plt.title(plot_title)

        # # Create box and swarm plot with clipped data
        # ax = sns.boxplot(data=data_clipped)
        # sns.swarmplot(data=data_clipped, color=".25")
        # sns.swarmplot(data = outliers, color = "red")
        # if verbose: plt.show()
        pass
    else:
        # initialize plot information:
        sns.set(color_codes=True)
        plt.figure(1, figsize=(9, 6))
        plt.title(plot_title)
    
        # plot data on swarmplot and boxplot
        ax = sns.boxplot(data=data)
        sns.swarmplot(data=data, color=".25")
        
        # y-axis label
        ax.set(ylabel=y_axis_label)
    
        # write labels with number of elements
        ax.set_xticks(np.arange(len(data)), labels = x_labels)
        ax.legend()
        
        if write:
            # write figure file with quality 400 dpi
            plt.savefig(figure_name, bbox_inches='tight', dpi=400)
            if verbose: plt.show()
            else: plt.close()
        if verbose: plt.show()
    return 0

def main(path = 'C:\\Users\\gt8mar\\capillary-flow\\data\\part11\\230427\\vid01',
          write = True, verbose = False):
    input_folder = os.path.join(path, "E_centerline", "coords")
    output_folder = os.path.join(path, "E_centerline", "images")
    participant, date, video, file_prefix = parse_vid_path(path)
    skeleton_data = load_csv_list(input_folder, dtype=float)
    # load radii from skeleton data
    radii = [array[:, 2] for array in skeleton_data] # note that the radii will be row vectors
    medians = []
    means = []
    for capillary in radii:
        median = np.median(capillary) * 2 /PIX_UM
        mean = np.mean(capillary) *2/PIX_UM
        medians.append(median)
        means.append(mean)
    plot_box_swarm([medians, means], x_labels = ["medians", "means"], y_axis_label="diameter (um)", 
                    plot_title=f"{file_prefix} capillary diameters", figure_name="figure 1",
                    write=True)
    return np.mean(means)

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    means = []
    # for i in range(1,9):
    #     sample = 'sample_' + str(i).zfill(3)
    #     mean = main("set_01", sample, write = False, verbose=False)
    #     means.append(mean)
    # plt.plot(means)
    # plt.show()   
    main(path = '/hpc/projects/capillary-flow/data/part11/230427/vid01', write = True, verbose = False) 
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
