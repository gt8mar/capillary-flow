"""
Filename: src/analysis/figs_ci.py
---------------------------------

Plots the CI bands for the figures in the paper.

By: Marcus Forst
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.plotting_utils import plot_CI, plot_CI_multiple_bands

from src.config import PATHS, load_source_sans

source_sans = load_source_sans()

def main():
    """
    Plots the CI bands for the figures in the paper.
    """
    # Load the data
    data_filepath = os.path.join(PATHS['cap_flow'], 'summary_df_nhp_video_stats2.csv')
    df = pd.read_csv(data_filepath)

    # ------------------------------------------------------------------------------------------------
    # Data cleaning
    # ------------------------------------------------------------------------------------------------
    
    df = df.dropna(subset=['Age'])
    df = df.dropna(subset=['Corrected Velocity'])

    # Drop duplicate velocities for the same video
    df = df.drop_duplicates(subset=['Participant', 'Video'])

    # Plot Control CI bands
    # ------------------------------------------------------------------------------------------------
    # Filter for controls
    controls_df = df[df['SET'] == 'set01']
    # Filter for affected
    df['Set_affected'] = np.where(df['SET'] == 'set01', 'set01', 'set04')
    
    # plot_CI_multiple_bands(controls_df, thresholds=[29, 49], variable='Age', method='bootstrap', 
    #                        n_iterations=1000, ci_percentile=95, write=True, dimensionless=False, 
    #                        video_median=False, log_scale=False, velocity_variable='Corrected Velocity')
    # Age

    # print any rows in controls_df that don't have a value for 'Age'

    plot_CI(controls_df, variable='Age', method='bootstrap', n_iterations=1000, 
            ci_percentile=95, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = False, velocity_variable = 'Corrected Velocity')
    # Sex
    plot_CI(controls_df, variable='Sex', method='bootstrap', n_iterations=1000, 
            ci_percentile=95, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = False, velocity_variable = 'Corrected Velocity')
    # Blood Pressure
    plot_CI(controls_df, variable='SYS_BP', method='bootstrap', n_iterations=1000, 
            ci_percentile=95, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = False, velocity_variable = 'Corrected Velocity')

    # ------------------------------------------------------------------------------------------------
    # Plot Affected CI bands
    # ------------------------------------------------------------------------------------------------
    plot_CI(df, variable='Set_affected', method='bootstrap', n_iterations=1000, 
            ci_percentile=95, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = False, velocity_variable = 'Corrected Velocity')
    plot_CI(df, variable='Set_affected', method='bootstrap', n_iterations=1000, 
            ci_percentile=95, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = True, velocity_variable = 'Corrected Velocity')


    # ------------------------------------------------------------------------------------------------
    # Plot Diabetes CI bands
    # ------------------------------------------------------------------------------------------------
    # Plot the CI bands for the diabetes data
    plot_CI(df, variable='Diabetes', method='bootstrap', n_iterations=1000, 
            ci_percentile=95, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = False, velocity_variable = 'Corrected Velocity')
    plot_CI(df, variable='Diabetes', method='bootstrap', n_iterations=1000, 
            ci_percentile=95, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = True, velocity_variable = 'Corrected Velocity')

    # ------------------------------------------------------------------------------------------------
    # Plot Hypertension CI bands
    # ------------------------------------------------------------------------------------------------
    plot_CI(df, variable='Hypertension', method='bootstrap', n_iterations=1000, 
            ci_percentile=95, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = False, velocity_variable = 'Corrected Velocity') 
    plot_CI(df, variable='Hypertension', method='bootstrap', n_iterations=1000, 
            ci_percentile=95, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = True, velocity_variable = 'Corrected Velocity')

    return 0

if __name__ == '__main__':
    main()








