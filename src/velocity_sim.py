"""
Filename: velocity_sim.py
--------------------------
This file simulates fluid flow using Bernoulli's equations
"""

import time
import numpy as np

def main():
    x = range(0, 450, 25)
    print(x)
    print(len(x))
    y1 = [10, 10, 11, 12, 12, 13, 14, 14, 15]
    y2 = [25] * 2
    y3 = [12] * 7
    ymid = y1.extend(y2)
    ymid2 = y1.extend(y3)
    print(len(ymid2))
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