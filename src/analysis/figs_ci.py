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
from src.tools.plotting_utils import (
    plot_CI, 
    plot_CI_multiple_bands, 
    plot_participant_velocity_profiles,
    plot_participant_velocity_profiles_by_group
)

from src.config import PATHS, load_source_sans

source_sans = load_source_sans()

def main(plot_participant_profiles=False, plot_group_profiles=False):
    """
    Plots the CI bands for the figures in the paper.
    
    Args:
        plot_participant_profiles (bool): Whether to plot participant velocity profiles.
                                         Default is True.
        plot_group_profiles (bool): Whether to plot group comparisons using participant profiles.
                                   Default is True.
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
    # Plot Diabetes CI bands with participant weighting
    # ------------------------------------------------------------------------------------------------
    # Plot the CI bands for the diabetes data
    plot_CI(df, variable='Diabetes', method='bootstrap', n_iterations=1000, 
            ci_percentile=95, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = False, velocity_variable = 'Corrected Velocity')
    plot_CI(df, variable='Diabetes', method='bootstrap', n_iterations=1000, 
            ci_percentile=95, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = True, velocity_variable = 'Corrected Velocity')
    
#     # Plot with even participant weighting (each participant contributes equally to the distribution)
#     plot_CI(df, variable='Diabetes', method='bootstrap', n_iterations=1000, 
#             ci_percentile=95, write=True, dimensionless=False, video_median=False, 
#             log_scale=False, old = False, velocity_variable = 'Corrected Velocity',
#             participant_weighting=True)

    # ------------------------------------------------------------------------------------------------
    # Plot Hypertension CI bands
    # ------------------------------------------------------------------------------------------------
    plot_CI(df, variable='Hypertension', method='bootstrap', n_iterations=1000, 
            ci_percentile=95, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = False, velocity_variable = 'Corrected Velocity') 
    plot_CI(df, variable='Hypertension', method='bootstrap', n_iterations=1000, 
            ci_percentile=95, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = True, velocity_variable = 'Corrected Velocity')
            
#     # Plot with even participant weighting
#     plot_CI(df, variable='Hypertension', method='bootstrap', n_iterations=1000, 
#             ci_percentile=95, write=True, dimensionless=False, video_median=False, 
#             log_scale=False, old = False, velocity_variable = 'Corrected Velocity',
#             participant_weighting=True)

    # ------------------------------------------------------------------------------------------------
    # Plot Participant Velocity Profiles (optional)
    # ------------------------------------------------------------------------------------------------
    if plot_participant_profiles:
        # Plot profiles for all participants
        participant_profiles = plot_participant_velocity_profiles(
            df, 
            method='bootstrap', 
            n_iterations=1000, 
            ci_percentile=95, 
            write=True, 
            dimensionless=False, 
            log_scale=False, 
            velocity_variable='Corrected Velocity',
            filename_prefix='all_participants'
        )
        
        # Plot profiles for control participants only
        control_profiles = plot_participant_velocity_profiles(
            controls_df, 
            method='bootstrap', 
            n_iterations=1000, 
            ci_percentile=95, 
            write=True, 
            dimensionless=False, 
            log_scale=False, 
            velocity_variable='Corrected Velocity',
            filename_prefix='control_participants'
        )
        
        print(f"Created velocity profiles for {len(participant_profiles['Participant'].unique())} participants")
    
    # ------------------------------------------------------------------------------------------------
    # Plot Participant Velocity Profiles by Group (optional)
    # ------------------------------------------------------------------------------------------------
    if plot_group_profiles:
        print("\nCreating participant velocity profile group comparisons...")
        
        # Age comparison
        age_profiles = plot_participant_velocity_profiles_by_group(
            controls_df,
            variable='Age',
            method='bootstrap',
            n_iterations=1000,
            ci_percentile=95,
            write=True,
            dimensionless=False,
            log_scale=False,
            velocity_variable='Corrected Velocity'
        )
        
        # Sex comparison
        sex_profiles = plot_participant_velocity_profiles_by_group(
            controls_df,
            variable='Sex',
            method='bootstrap',
            n_iterations=1000,
            ci_percentile=95,
            write=True,
            dimensionless=False,
            log_scale=False,
            velocity_variable='Corrected Velocity'
        )
        
        # Blood Pressure comparison
        bp_profiles = plot_participant_velocity_profiles_by_group(
            controls_df,
            variable='SYS_BP',
            method='bootstrap',
            n_iterations=1000,
            ci_percentile=95,
            write=True,
            dimensionless=False,
            log_scale=False,
            velocity_variable='Corrected Velocity'
        )
        
        # Affected comparison
        affected_profiles = plot_participant_velocity_profiles_by_group(
            df,
            variable='Set_affected',
            method='bootstrap',
            n_iterations=1000,
            ci_percentile=95,
            write=True,
            dimensionless=False,
            log_scale=False,
            velocity_variable='Corrected Velocity'
        )
        
        # Diabetes comparison
        diabetes_profiles = plot_participant_velocity_profiles_by_group(
            df,
            variable='Diabetes',
            method='bootstrap',
            n_iterations=1000,
            ci_percentile=95,
            write=True,
            dimensionless=False,
            log_scale=False,
            velocity_variable='Corrected Velocity'
        )
        
        # Hypertension comparison
        hypertension_profiles = plot_participant_velocity_profiles_by_group(
            df,
            variable='Hypertension',
            method='bootstrap',
            n_iterations=1000,
            ci_percentile=95,
            write=True,
            dimensionless=False,
            log_scale=False,
            velocity_variable='Corrected Velocity'
        )
        
        print("Completed participant velocity profile group comparisons")

    return 0

if __name__ == '__main__':
    main(plot_participant_profiles=False, plot_group_profiles=False)  # Set parameters to False to skip respective plots








