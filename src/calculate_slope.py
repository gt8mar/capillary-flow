"""
Filename: calculate_slope.py
-------------------------------
This file uses a sliding window method to calculate the velocity of blood through capillaries.
By: Marcus Forst
"""

import os
import time
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from src.tools.load_csv_list import load_csv_list
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression

WINDOW_SIZE = 25

def sliding_window(flows, fps = 169.3, u_per_pix = 100/174, verbose = True):
    """This function uses the sliding window method to calculate the velocities of
    the capillary.
    Input: flows: list of 2d numpy arrays
    """
    average_flow = []
    average_sliding = []
    for i in range(len(flows)):
        flow = flows[i]
        pressure = (i+1)*0.2
        average = (flow[-25][1]-flow[25][1])/(flow[-25][0]-flow[25][0])
        average *= fps
        average *= u_per_pix
        average_flow.append(average)
        window_1 = flow[0:-WINDOW_SIZE]
        window_2 = flow[WINDOW_SIZE:]
        difference = window_2-window_1
        velocity = difference[:,1]/difference[:,0]
        velocity *= fps
        velocity *= u_per_pix
        if verbose:
            plt.plot(velocity)
        average_sl = np.mean(velocity)
        average_sliding.append(average_sl)
    if verbose:
        plt.title(f"velocity vs capillary position ")  # {pressure}psi
        plt.xlabel("capillary position")
        plt.ylabel("velocity")
        plt.legend(["0.2psi", "0.4psi", "0.6psi", "0.8psi"])
        plt.show()
    return average_flow, average_sliding

def spline(flows, fps = 169.3, u_per_pix = 100/174, verbose = True):
    x_splines = []
    v_splines = []
    for flow in flows:
        cs = CubicSpline(flow[:,0], flow[:,1])
        xs = np.arange(-0.5, 80, 1)
        # plt.plot(xs, cs(xs))
        plt.plot(xs, cs(xs, 1))
        plt.show()
    return x_splines, v_splines
def plot_box_swarm(data, x_labels, y_axis_label,  plot_title, figure_name):
    """Plot box-plot and swarm plot for data list.
 
    Args:
        data (list of list): List of lists with data to be plotted.
        y_axis_label (str): Y- axis label.
        x_labels (list of str): List with labels of x-axis.
        plot_title (str): Plot title.
        figure_name (str): Path to output figure.
         
    """
    sb.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
 
    # add title to plot
    plt.title(plot_title)
 
    # plot data on swarmplot and boxplot
    sb.swarmplot(data=data, color=".25")
    ax = sb.boxplot(data=data)
 
    # y-axis label
    ax.set(ylabel=y_axis_label)
 
    # write labels with number of elements
    ax.set_xticks(np.arange(4), labels = x_labels)
    ax.legend()
    
    # ax.set_xticklabels(["{} (n={})".format(l, len(data[x])) for x, l in enumerate(x_labels)], rotation=10)
 
    # write figure file with quality 400 dpi
    # plt.savefig(figure_name, bbox_inches='tight', dpi=400)
    plt.show()
    # plt.close()



def imagej_slopes(verbose = True, write = True):
    input_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed\\set_01\\participant_04_cap_04')
    data = []
    for i in range (1,5):
        hard_code_list = []
        sample_data = np.loadtxt(os.path.join(input_folder, "flow_00"+str(i)+".txt"), dtype = float)
        if type(sample_data) == float or int:
            pass
            # hard_code_list.append(sample_data)
            # data.append(hard_code_list)
        # else:
        data.append(sample_data.tolist())
    plot_box_swarm(data, ["0.2 psi", "0.4 psi", "0.6 psi", "0.8 psi"], 
                   "flow (um^3/s)", "Flow vs pressure cap_4", "figure 1")
    return 0

def calculate_slope(SET, sample, cap, verbose = False, write = False):
    input_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET),
                                 'participant_04_cap_04', "blood_flow_segmentations","part_04_cap_04\\coords\\sample_001")
    flows = load_csv_list(input_folder) # This outputs a list of numpy arrays
    average_flow, average_sliding = sliding_window(flows, verbose = verbose)
    fps = 169.3
    print(np.ones(len(average_flow))/average_flow)
    print(np.ones(len(average_sliding))/average_sliding)
    # x_splines, v_splines = spline(flows)
    return 0





"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    for i in range(1,2):
        calculate_slope("set_01", "sample_00"+ str(i), cap = 4, write = False, verbose=True)
    imagej_slopes(write = False)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
