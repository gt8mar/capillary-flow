"""
Filename: flow_sim.py
---------------------
This file simulates flow velocities through capillaries
using bernoulli's equations (consv of Energy)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from src.tools.load_csv_list import load_csv_list

path = 'C:\\Users\\gt8mar\\capillary-flow\\data\\processed\\set_01\\sample_009\\D_segmented'
centerline_distances = load_csv_list(path)
centerline_distances = centerline_distances[0]
# plt.plot(centerline_distances)
# plt.show()

v = 1000 / (centerline_distances**2)
v[0] = v[2]
v[1] = v[2]

# plt.plot(v)
# plt.title('velocities vs time')
# plt.show()

dt = 0.1 #this should be the inverse of the frame rate
t = []
for i in range(len(v)):
    t = t+[i*dt]

plt.plot(t, v)
dist = 0
dist_list = []
for i in range(len(v)):
    dx = v[i]*dt
    dist = dist + dx
    dist_list.append(dist)
plt.plot(t, dist_list)
plt.show()