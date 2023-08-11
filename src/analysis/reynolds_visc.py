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
import matplotlib.cm as cm
import numpy as np
# import seaborn as sns
# from src.tools.load_csv_list import load_csv_list

PIX_UM = 1.74

def n_star(diameter):
    # This function calculates the in-vitro viscocity for a given diameter and standard
    # hemocrit of 0.45. This is one of the inputs to find the in-vivo viscocity.
    visc_vitro = 6*math.exp(-0.085*diameter)+3.2-2.44*math.exp(-0.06*(diameter**0.645))
    return visc_vitro
def shape(diameter):
    # This function calculates the shape of the viscocity curve based on the diameter
    C = (0.8 + math.exp(-0.075*diameter))*(-1 + 1/(1+10**(-11)*(diameter**12)))+(1/(1+(10**(-11))*(diameter**12)))
    return C

def find_visc_vivo(D, hemocrit = 0.45):
    # This function calculates the in-vivo viscocity of blood in a capillary of 
    # a specified diameter (in um)
    visc = (1+(n_star(D)-1)*((1-hemocrit)**shape(D)-1)/((1-0.45)**shape(D)-1)*(D/((D-1.1)**2)))*(D/((D-1.1)**2))
    return visc

def find_reynolds(visc, D):
    density = 1.06 
    velocity = 1.5/10
    D /= (1000*10)
    reyn = (density*D*velocity)/visc
    return reyn

def main():
    viscs = []
    reynolds = []
    diameters = range(8,30)
    for d in diameters:
        visc = find_visc_vivo(d)
        viscs.append(visc)
        reynold = find_reynolds(visc, d)
        reynolds.append(reynold)
    # Create figure and axis objects
    fig, ax1 = plt.subplots(figsize=(8,5))
    fig.subplots_adjust(right=0.8)
    colors = cm.get_cmap('Set1', 2)

    # Plot first function with left y axis
    ax1.plot(diameters, viscs, color = colors(0))
    ax1.set_xlabel('Diameter (um)')
    ax1.set_ylabel('Viscosity (p)', color=colors(0))
    ax1.tick_params('y', colors=colors(0))

    # Create a twin axis with different y axis
    ax2 = ax1.twinx()

    # Plot second function with right y axis
    ax2.plot(diameters, reynolds, color = colors(1))
    ax2.set_ylabel('Reynold\'s number', color=colors(1))
    ax2.tick_params('y', colors=colors(1))
    ax2.yaxis.set_label_coords(1.1, 0.5)
    plt.title("Capillary Viscosity")
    plt.show()
    # plt.plot(diameters, viscs)
    # plt.plot(diameters, reynolds)
    # plt.title("Capillary Viscosity")
    # plt.ylabel("Viscosity (cp)")
    # plt.xlabel("Diameter (um)")
    # plt.show()
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
