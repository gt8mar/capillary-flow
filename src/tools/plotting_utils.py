"""
Filename: src/tools/plotting_utils.py

This file contains utility functions for plotting.

By: Marcus Forst
"""

import os
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from scipy import stats
from sklearn.utils import resample
from scipy.stats import ks_2samp
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from src.config import PATHS
from src.analysis.plot_big import cap_flow_path, create_monochromatic_palette, adjust_brightness_of_colors
from src.analysis.plot_big import calculate_stats, calculate_median_ci

# Get the hostname of the computer
hostname = platform.node()



def plot_CI(df, variable='Age', method='bootstrap', n_iterations=1000, 
            ci_percentile=99.5, write=True, dimensionless=False, video_median=False, log_scale=False, old = False, velocity_variable = 'Corrected Velocity'):
    """Plots the mean/median and CI for the variable of interest, with KS statistic."""
    # Set up style and font
    sns.set_style("whitegrid")
    source_sans = FontProperties(fname='C:\\Users\\gt8mar\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    control_df = df[df['SET']=='set01']
    hypertensive_df = df[df['SET']=='set02']
    diabetic_df = df[df['SET']=='set03']
    affected_df = df[df['Set_affected']=='set04']

    # Set color palette based on variable
    if variable == 'Age':
        base_color = '#1f77b4'
        conditions = [df[variable] <= 50, df[variable] > 50]
        choices = ['≤50', '>50']
    elif variable == 'SYS_BP':
        base_color = '2ca02c'
        conditions = [df[variable] < 120, df[variable] >= 120]
        choices = ['<120', '≥120']
    elif variable == 'Sex':
        base_color = '674F92'
        conditions = [df[variable] == 'M', df[variable] == 'F']
        choices = ['Male', 'Female']
    elif variable == 'Diabetes':
        base_color = 'ff7f0e'
        # conditions = [
        #     df[variable].isin([False, None, 'Control', 'FALSE', 'PRE']),
        #     df[variable].isin([True, 'TRUE','TYPE 1', 'TYPE 2', 'Diabetes'])
        # ]
        conditions = [df['SET'] == 'set01', df['SET'] == 'set03']
        choices = ['Control', 'Diabetic']
    elif variable == 'Hypertension':
        base_color = 'd62728'
        # conditions = [
        #     df[variable].isin([False, None, 'Control', 'FALSE']),
        #     df[variable].isin([True, 1.0, 'Hypertension', 'TRUE'])
        # ]
        conditions = [df['SET'] == 'set01', df['SET'] == 'set02']
        choices = ['Control', 'Hypertensive']
    elif variable == 'Set_affected':
        base_color = '#00CED1' # sky blue
        conditions = [df['SET'] == 'set01', df['Set_affected'] == 'set04']
        choices = ['Control', 'Affected']
    else:
        raise ValueError(f"Unsupported variable: {variable}")

    palette = create_monochromatic_palette(base_color)
    palette = adjust_brightness_of_colors(palette, brightness_scale=.2)
    sns.set_palette(palette)

    if log_scale:
        df[velocity_variable] = df[velocity_variable] + 10


    if video_median:
        df = df.groupby(['Participant', 'Video', 'Capillary']).first().reset_index()
        df.rename(columns={'Dimensionless Velocity': 'Dimensionless Velocity OG'}, inplace=True) 
        df.rename(columns={'Video Median Dimensionless Velocity': 'Dimensionless Velocity'}, inplace=True)      

    # Group data with more explicit handling
    group_col = f'{variable} Group'
    df[group_col] = np.select(conditions, choices, default='Unknown')
    
    # # Filter out 'Unknown' values
    # df = df[df[group_col] != 'Unknown']
    
    # Print unique values for debugging
    print(f"Unique values in {group_col}: {df[group_col].unique()}")

    # Calculate stats
    stats_func = calculate_median_ci if method == 'bootstrap' else calculate_stats
    stats_df = df.groupby([group_col, 'Pressure']).apply(stats_func, ci_percentile=ci_percentile, dimensionless=dimensionless).reset_index()

    # Calculate KS statistic
    grouped = df.groupby(group_col)
    ks_stats = []
    
    for pressure in df['Pressure'].unique():
        try:
            group_1 = grouped.get_group(choices[0])
            group_2 = grouped.get_group(choices[1])

            if log_scale:
                group_1['Log Corrected Velocity'] = np.log(group_1['Corrected Velocity'])
                group_2['Log Corrected Velocity'] = np.log(group_2['Corrected Velocity'])
                group_1_velocities = group_1[group_1['Pressure'] == pressure]['Log Corrected Velocity']
                group_2_velocities = group_2[group_2['Pressure'] == pressure]['Log Corrected Velocity']
            else:
                group_1_velocities = group_1[group_1['Pressure'] == pressure]['Corrected Velocity']
                group_2_velocities = group_2[group_2['Pressure'] == pressure]['Corrected Velocity']
            
            
            ks_stat, p_value = ks_2samp(group_1_velocities, group_2_velocities)
            if log_scale:
                group_1_median = np.log(group_1[group_1['Pressure'] == pressure]['Log Corrected Velocity'].median())
                group_2_median = np.log(group_2[group_2['Pressure'] == pressure]['Log Corrected Velocity'].median())
            else:
                group_1_median = group_1[group_1['Pressure'] == pressure]['Corrected Velocity'].median()
                group_2_median = group_2[group_2['Pressure'] == pressure]['Corrected Velocity'].median()
            ks_stats.append({'Pressure': pressure, 'KS Statistic': ks_stat, 'p-value': p_value, 'Group 1 Median': group_1_median, 'Group 2 Median': group_2_median})

        except KeyError as e:
            print(f"Warning: Could not find group for pressure {pressure}: {e}")
            continue

    if ks_stats:  # Only create DataFrame if we have statistics
        ks_df = pd.DataFrame(ks_stats)
        print(variable)
        print(ks_df)

    # Plot
    plt.close()
    fig, ax = plt.subplots(figsize=(2.4, 2.0))

    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

 # Ensure consistent coloring by using only two colors
    control_color = palette[0]
    condition_color = palette[3]
    
    if dimensionless:
        y_col = 'Median Dimensionless Velocity' if method == 'bootstrap' else 'Mean Dimensionless Velocity'
    else:
        y_col = 'Median Velocity' if method == 'bootstrap' else 'Mean Velocity'
    lower_col = 'CI Lower Bound' if method == 'bootstrap' else 'Lower Bound'
    upper_col = 'CI Upper Bound' if method == 'bootstrap' else 'Upper Bound'

    if velocity_variable == 'Shear_Rate':
        y_col = 'Median Shear Rate' if method == 'bootstrap' else 'Mean Shear Rate'
        lower_col = 'CI Lower Bound' if method == 'bootstrap' else 'Lower Bound'
        upper_col = 'CI Upper Bound' if method == 'bootstrap' else 'Upper Bound'

    # Plot control group
    control_data = stats_df[stats_df[group_col] == choices[0]]
    ax.errorbar(control_data['Pressure'], control_data[y_col],
                yerr=[control_data[y_col] - control_data[lower_col], 
                      control_data[upper_col] - control_data[y_col]],
                label=choices[0], fmt='-o', markersize=2, color=control_color)
    ax.fill_between(control_data['Pressure'], control_data[lower_col], 
                    control_data[upper_col], alpha=0.4, color=control_color)

    # Plot condition group
    condition_data = stats_df[stats_df[group_col] == choices[1]]
    ax.errorbar(condition_data['Pressure'], condition_data[y_col],
                yerr=[condition_data[y_col] - condition_data[lower_col], 
                      condition_data[upper_col] - condition_data[y_col]],
                label=choices[1], fmt='-o', markersize=2, color=condition_color)
    ax.fill_between(condition_data['Pressure'], condition_data[lower_col], 
                    condition_data[upper_col], alpha=0.4, color=condition_color)

    # Add log scale if requested
    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        
    # Create legend handles with consistent colors
    legend_handles = [mpatches.Patch(color=control_color, label=choices[0], alpha=0.6),
                     mpatches.Patch(color=condition_color, label=choices[1], alpha=0.6)]

    ax.set_xlabel('Pressure (psi)', fontproperties=source_sans)
    if dimensionless:
        ax.set_ylabel('Dimensionless Velocity', fontproperties=source_sans)
        ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Dimensionless Velocity vs. Pressure with {ci_percentile}% CI', 
                    fontproperties=source_sans, fontsize=8)
    else:
        if velocity_variable == 'Shear_Rate':
            ax.set_ylabel('Shear Rate (1/s)', fontproperties=source_sans)
        else:
            ax.set_ylabel('Velocity (um/s)', fontproperties=source_sans)
        if velocity_variable == 'Shear_Rate':
            ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Shear Rate vs. Pressure with {ci_percentile}% CI', 
                        fontproperties=source_sans, fontsize=8)
        else:
            ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Velocity vs. Pressure with {ci_percentile}% CI', 
                        fontproperties=source_sans, fontsize=8)
    
    ax.legend(handles=legend_handles, prop=source_sans)
    ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    if write:
        if video_median:
            if old:
                plt.savefig(os.path.join(cap_flow_path, 'results', f'{variable}_videomedians_CI_old.png'), dpi=600)
            else:
                if velocity_variable == 'Shear_Rate':
                    plt.savefig(os.path.join(cap_flow_path, 'results', f'{variable}_videomedians_CI_new_shear_rate.png'), dpi=600)
                else:
                    plt.savefig(os.path.join(cap_flow_path, 'results', f'{variable}_videomedians_CI_new.png'), dpi=600)
        else:
            if velocity_variable == 'Shear_Rate':
                plt.savefig(os.path.join(cap_flow_path, 'results', f'{variable}_CI_new_shear_rate.png'), dpi=600)
            else:
                plt.savefig(os.path.join(cap_flow_path, 'results', f'{variable}_CI_new.png'), dpi=600)
    else:
        plt.show()
    return 0

