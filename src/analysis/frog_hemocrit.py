"""
Filename: frog_hemocrit.py
-------------------------------------------------
This script loads hemocrit data from frog_hemocrit.csv and merges it with
capillary radius information from frog_hemocrit_radii.csv.

It calculates mean_diameter and bulk_diameter values for each capillary
and merges them into the counting dataframe, then saves the result with
a _radii suffix.

By: Marcus Forst
""" 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import PATHS


def main():
    hemocrit_file_path = os.path.join(PATHS['cap_flow'], '25-4-11CountedVelocities.csv')
    hemocrit_df = pd.read_csv(hemocrit_file_path)
    print(hemocrit_df.columns)
    
    # calculate counts per unit time
    hemocrit_df['Counts_per_unit_time'] = hemocrit_df['Measured_Counts'] / hemocrit_df['SampledFrames'] * hemocrit_df['HertzSampled']
    hemocrit_df['Hc_index'] = hemocrit_df['Counts_per_unit_time'] / hemocrit_df['VelocityuMperSec']

    # Unique conditions
    unique_conditions = hemocrit_df['Condition'].unique()
    condition_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_conditions)))
    
    # Create a mapping of conditions to colors
    color_map = {condition: condition_colors[i] for i, condition in enumerate(unique_conditions)}
    
# -----------------------------------------------------------------
# Velocity, diameter, counts per unit time
# -----------------------------------------------------------------

    # # plot velocties vs mean_diameter
    # plt.figure(figsize=(10, 6))
    # for condition in unique_conditions:
    #     subset = hemocrit_df[hemocrit_df['Condition'] == condition]
    #     plt.scatter(subset['Mean_Diameter'], subset['VelocityuMperSec'], 
    #                color=color_map[condition], label=condition)
    # plt.xlabel('Mean Diameter')
    # plt.ylabel('Velocity (μm/sec)')
    # plt.title('Velocity vs Mean Diameter')
    # plt.legend()
    # plt.show()
   
    # # make 3D plot of velocity vs mean_diameter vs 'Measured_Counts'
    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(hemocrit_df['Mean_Diameter'], 
    #            hemocrit_df['VelocityuMperSec'], 
    #            hemocrit_df['Counts_per_unit_time'], 
    #            c=hemocrit_df['Condition'].astype('category').cat.codes, 
    #            cmap='viridis')
    # ax.set_xlabel('Mean Diameter')
    # ax.set_ylabel('Velocity (μm/sec)')
    # ax.set_zlabel('Counts per unit time')
    
    # # Add a legend for the 3D plot
    # legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
    #                   markerfacecolor=color_map[condition], label=condition, markersize=8)
    #                   for condition in unique_conditions]
    # ax.legend(handles=legend_elements, title="Condition")
    # plt.show() 

    # # plot counts per unit time vs mean_diameter
    # plt.figure(figsize=(10, 6))
    # for condition in unique_conditions:
    #     subset = hemocrit_df[hemocrit_df['Condition'] == condition]
    #     plt.scatter(subset['Mean_Diameter'], subset['Counts_per_unit_time'], 
    #                color=color_map[condition], label=condition)
    # plt.xlabel('Mean Diameter')
    # plt.ylabel('Counts per unit time')
    # plt.legend()
    # plt.show()

    # # plot counts per unit time vs velocity
    # plt.figure(figsize=(10, 6))
    # for condition in unique_conditions:
    #     subset = hemocrit_df[hemocrit_df['Condition'] == condition]
    #     plt.scatter(subset['VelocityuMperSec'], subset['Counts_per_unit_time'], 
    #                color=color_map[condition], label=condition)
    # plt.xlabel('Velocity (μm/sec)')
    # plt.ylabel('Counts per unit time')
    # plt.legend()
    # plt.show()

# --------------------------------------------------------------------------
# Hematocrit Index
# --------------------------------------------------------------------------

    # plot Hc_index vs mean_diameter
    plt.figure(figsize=(10, 6))
    for condition in unique_conditions:
        subset = hemocrit_df[hemocrit_df['Condition'] == condition]
        plt.scatter(subset['Mean_Diameter'], subset['Hc_index'], 
                   color=color_map[condition], label=condition)
    plt.xlabel('Mean Diameter')
    plt.ylabel('Hematocrit Index')
    plt.title('Hematocrit Index vs Mean Diameter')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot cdf of Hc_index for each condition
    plt.figure(figsize=(10, 6))
    for condition in unique_conditions:
        subset = hemocrit_df[hemocrit_df['Condition'] == condition]
        plt.plot(np.sort(subset['Hc_index']), np.arange(len(subset)) / len(subset), 
                 color=color_map[condition], label=condition)
    plt.show()

    # # plot Hc_index vs velocity
    # plt.figure(figsize=(10, 6))
    # for condition in unique_conditions:
    #     subset = hemocrit_df[hemocrit_df['Condition'] == condition]
    #     plt.scatter(subset['VelocityuMperSec'], subset['Hc_index'], 
    #                color=color_map[condition], label=condition)
    # plt.xlabel('Velocity (μm/sec)')
    # plt.ylabel('Hematocrit Index')
    # plt.title('Hematocrit Index vs Velocity')
    # plt.grid(alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()  

    # # plot Hc_index vs counts per unit time
    # plt.figure(figsize=(10, 6))
    # for condition in unique_conditions:
    #     subset = hemocrit_df[hemocrit_df['Condition'] == condition]
    #     plt.scatter(subset['Counts_per_unit_time'], subset['Hc_index'], 
    #                color=color_map[condition], label=condition)
    # plt.xlabel('Counts per unit time')
    # plt.ylabel('Hematocrit Index')
    # plt.title('Hematocrit Index vs Counts per unit time')
    # plt.grid(alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

# --------------------------------------------------
# Histograms
# --------------------------------------------------
    

    # # plot histograms of counts per unit time for each condition
    # plt.figure(figsize=(12, 7))
    
    # # Calculate optimal number of bins based on all data
    # all_counts = hemocrit_df['Counts_per_unit_time']
    # bin_count = min(int(np.sqrt(len(all_counts))), 30)  # Limit to 30 bins max
    # bin_count = 10
    
    # # Find common bin edges for all histograms
    # min_val = all_counts.min()
    # max_val = all_counts.max()
    # bin_edges = np.linspace(min_val, max_val, bin_count + 1)
    
    # for condition in unique_conditions:
    #     subset = hemocrit_df[hemocrit_df['Condition'] == condition]
    #     plt.hist(subset['Counts_per_unit_time'], 
    #              bins=bin_edges, 
    #              alpha=0.5,  # Make translucent with 50% opacity
    #              color=color_map[condition], 
    #              edgecolor='black',  # Add edge color for better separation
    #              linewidth=0.8,
    #              density=True,  # Normalize the histogram so the area sums to 1
    #              label=f"{condition} (n={len(subset)})")  # Add sample size to legend
    
    # plt.xlabel('Counts per unit time')
    # plt.ylabel('Probability Density')  # Changed from Frequency to Probability Density
    # plt.title('Normalized Distribution of Counts per Unit Time by Condition')
    # plt.grid(alpha=0.3)  # Add light grid for better readability
    # plt.legend()
    # plt.tight_layout()  # Adjust layout to make room for all elements
    # plt.show()
    
    return 0


if __name__ == '__main__':
    main()










