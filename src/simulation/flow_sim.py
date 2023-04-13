"""
Filename: flow_sim.py
---------------------
This file simulates flow velocities through capillaries
using bernoulli's equations (consv of Energy)
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
from src.tools.load_csv_list import load_csv_list

def calculate_velocities(radii, energy):
    """ Calculates velocities given an energy (int) and a list of radii (list of floats)"""
    v = int(energy) / (radii**2)
    v[0] = v[2] # chop outlier edge states
    v[1] = v[2]
    return v
def calculate_dist(v_list, dt):
    dist_list = []
    dist = 0
    for i in range(len(v_list)):
        dx = v_list[i]*dt
        dist = dist + dx
        dist_list.append(dist)
    return dist_list




def main(SET, sample):
    path = str(os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'D_segmented'))
    centerline_distances = load_csv_list(path, float)
    centerline_distances = centerline_distances[0]
    plt.plot(centerline_distances)
    plt.show()

    v1 = calculate_velocities(centerline_distances, 1000)
    v2 = calculate_velocities(centerline_distances, 500)


    plt.plot(v1)
    plt.plot(v2)
    plt.title('velocities vs time')
    plt.show()

    dt = 0.1 #this should be the inverse of the frame rate
    t = []
    for i in range(len(v1)):
        t = t+[i*dt]

    plt.plot(t, v1)
    plt.plot(t, v2)
    dist_list_1 = calculate_dist(v1, dt)
    dist_list_2 = calculate_dist(v2, dt)

    plt.plot(t, dist_list_1)
    plt.plot(t, dist_list_2)
    plt.show()


# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    SET = 'set_01'
    sample = 'sample_009'
    main(SET, sample)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))