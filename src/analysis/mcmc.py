"""
Filename: mcmc.py
------------------------------

By: Marcus Forst
"""
import os, sys, platform
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit



# Get the hostname of the computer
hostname = platform.node()

# Dictionary mapping hostnames to folder paths
cap_flow_folder_paths = {
    "LAPTOP-I5KTBOR3": 'C:\\Users\\gt8ma\\capillary-flow',
    "Quake-Blood": "C:\\Users\\gt8mar\\capillary-flow",
    "ComputerName3": "C:\\Users\\ejerison\\capillary-flow",
    # Add more computers as needed
}
default_folder_path = "/hpc/projects/capillary-flow"


# Determine the folder path based on the hostname
cap_flow_path = cap_flow_folder_paths.get(hostname, default_folder_path)




def flow_decay(P, Q0, k):
    return Q0 * (1 - P / k)**4

def flow_threshold(P, Q0, k, P_th):
    return np.where(P <= P_th, Q0, Q0 * (1 - (P - P_th) / k)**4)

def flow_sigmoid(P, Q_max, alpha, P0):
    return Q_max / (1 + np.exp(-alpha * (P - P0)))

def flow_piecewise(P, Q0, beta, P_static, P_plateau):
    return np.piecewise(P, 
        [P <= P_static, (P > P_static) & (P < P_plateau), P >= P_plateau],
        [lambda P: Q0, 
         lambda P: Q0 * np.exp(-beta * (P - P_static)), 
         lambda P: Q0 * 0.1])  # Plateau flow (0.1 as example)

def main():
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_medians.csv')
    data = pd.read_csv(data_filepath)

    normal_group = data[data['SET'] == 'set01']
    # print min and max age
    print(f"min age: {normal_group['Age'].min()}")
    print(f"max age: {normal_group['Age'].max()}")
    normal_group_old = normal_group[normal_group['Age']>50]

    pressure_data = normal_group_old['Pressure']
    velocity_data = normal_group_old['Video Median Velocity']

    # Model List
    model_list = ['decay', 'threshold', 'sigmoid', 'piecewise']
    
    # Initial guesses for parameters
    Q0_guess_decay = 600
    k_guess_decay = 1.5
    k_guess_threshold = 1.0
    input_params_decay = [Q0_guess_decay, k_guess_decay]
    input_params_threshold = [Q0_guess_decay, k_guess_decay, 0.3]
    input_params_sigmoid = [Q0_guess_decay, 1.0, 0.5]
    input_params_piecewise = [Q0_guess_decay, -300, 0.4, 1.0]

    # Example: Fit fourth-power decay model
    for model in model_list:
        if model == 'decay':
            popt, pcov = curve_fit(flow_decay, pressure_data, velocity_data, p0=input_params_decay)
        elif model == 'threshold':
            popt, pcov = curve_fit(flow_threshold, pressure_data, velocity_data, p0=input_params_threshold)
        elif model == 'sigmoid':
            popt, pcov = curve_fit(flow_sigmoid, pressure_data, velocity_data, p0=input_params_sigmoid, maxfev=5000)
        elif model == 'piecewise':
            popt, pcov = curve_fit(flow_piecewise, pressure_data, velocity_data, p0=input_params_piecewise)
        else:
            print(f"Model {model} not recognized")
            continue

        # Extract parameters
        print(f"Model: {model}")
        print(f"Parameters: {popt}")
        print(f"the pcov is {pcov}")
    # popt, pcov = curve_fit(flow_decay, pressure_data, velocity_data, p0=[Q0_guess, k_guess])
    # print(popt) # Generate predictions
        P_vals = np.linspace(0, max(pressure_data), 100)

        # Extract parameters
        if model == 'decay':
            Q0, k = popt
            Q_vals = flow_decay(P_vals, Q0, k)
        elif model == 'threshold':
            Q0, k, P_th = popt
            Q_vals = flow_threshold(P_vals, Q0, k, P_th)
        elif model == 'sigmoid':
            Q0, alpha, P0 = popt
            Q_vals = flow_sigmoid(P_vals, Q0, alpha, P0)
        elif model == 'piecewise':
            Q0, beta, P_static, P_plateau = popt
            Q_vals = flow_piecewise(P_vals, Q0, beta, P_static, P_plateau)
        else:
            print(f"Model {model} not recognized")
            continue

       

        # Plot
        plt.figure(figsize=(10, 6))
        # plt.scatter(pressure_data, velocity_data, label='Observed')
        # sns.violinplot(x=pressure_data, y=velocity_data)
        sns.boxplot(x=pressure_data, y=velocity_data)
        plt.plot(P_vals, Q_vals, label=f'Model Fit {model}', color='red')
        plt.xlabel('Pressure (psi)')
        plt.ylabel('Velocity (um/s)')
        plt.legend()
        plt.show()
    return 0

if __name__ == '__main__':
    main()