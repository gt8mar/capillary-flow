"""
Filename: reynolds_visc.py
-----------------------
This function calculates the reynolds number and viscocity given a capillary diameter
This is based off the following paper: 
Pries, A. R. et al. Resistance to blood flow in microvessels in vivo. Circulation Research 75, 904–915 (1994).

By: Marcus Forst
"""

import os, time, math
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# from src.tools.load_csv_list import load_csv_list

PIX_UM = 1.74

def n_star(diameter):
    # This function calculates the in-vitro viscocity for a given diameter and standard
    # hemocrit of 0.45. This is one of the inputs to find the in-vivo viscocity.
    visc_vitro = 6*math.exp(-0.085*diameter)+3.2-2.44*math.exp(-0.06*diameter**0.645)
    return visc_vitro
def shape(diameter):
    # This function calculates the shape of the viscocity curve based on the diameter
    C = (0.8 + math.exp(-0.075*diameter))*(-1 + 1/(1+10**(-11)*diameter**12))+(1/(1+10**(-11)*diameter**12))
    return C

def visc_vivo(diameter, hemocrit = 0.45):
    # This function calculates the in-vivo viscocity of blood in a capillary of 
    # a specified diameter (in um)
    visc = (1+(n_star(diameter)-1)*((1-hemocrit)**shape(diameter)-1)/((1-0.45)**shape(diameter)-1))
    return visc

def main():
    visc = visc_vivo(16)
    print(visc)
    return 0

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
