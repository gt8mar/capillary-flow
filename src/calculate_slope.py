"""
Filename: calculate_slope.py
-------------------------------
This file uses a sliding window method to calculate the velocity of blood through capillaries.
By: Marcus Forst
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from src.tools.load_csv_list import load_csv_list

WINDOW_SIZE = 25


def calculate_slope(SET, sample, cap, verbose = False, write = False):
    input_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET),
                                 'participant_04_cap_04', "blood_flow_segmentations","part_04_cap_04\\coords\\sample_001")
    flows = load_csv_list(input_folder) # This outputs a list of numpy arrays
    average_flow = []
    average_sliding = []
    for flow in flows:
        average = (flow[-25][0]-flow[25][0])/(flow[-25][1]-flow[25][1])
        average_flow.append(average)
        window_1 = flow[0:-WINDOW_SIZE]
        window_2 = flow[WINDOW_SIZE:]
        difference = window_2-window_1
        velocity = difference[:,0]/difference[:,1]
        if verbose:
            plt.plot(velocity)
            plt.show()
        average_sl = np.mean(velocity)
        average_sliding.append(average_sl)
    print(average_flow)
    print(average_sliding)
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
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
