"""
Filename: bpvideo_swelling.py
-----------------------------
This script analyzes how capillaries swell during systemic occlusion
during a BP measurement.

By: Marcus Forst
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from scipy import stats

# Import paths from config
from src.config import PATHS, load_source_sans

diabetes_list = ['part78','part79', 'part76', 'part73', 'part74', 'part75', 'part70', 'part65']
ignore_list = ['part28']

def analyze_swelling_changes(mask_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze mask area changes across timepoints for each participant.
    
    Args:
        mask_df: DataFrame with columns ['participant_id', 'timepoint', 'mask_area']
    
    Returns:
        DataFrame with swelling analysis results
    """
    # Pivot to get timepoints as columns
    pivot_df = mask_df.pivot(index='participant_id', columns='timepoint', values='mask_area')
    
    # Calculate changes
    analysis_df = pd.DataFrame()
    analysis_df['participant_id'] = pivot_df.index
    analysis_df['start_area'] = pivot_df['start'].values
    analysis_df['swollen_area'] = pivot_df['swollen'].values
    analysis_df['end_area'] = pivot_df['end'].values
    
    # Calculate absolute and percentage changes
    analysis_df['start_to_swollen_abs'] = analysis_df['swollen_area'] - analysis_df['start_area']
    analysis_df['swollen_to_end_abs'] = analysis_df['end_area'] - analysis_df['swollen_area']
    analysis_df['start_to_swollen_pct'] = (analysis_df['start_to_swollen_abs'] / analysis_df['start_area']) * 100
    analysis_df['swollen_to_end_pct'] = (analysis_df['swollen_to_end_abs'] / analysis_df['swollen_area']) * 100
    analysis_df['total_change_pct'] = (analysis_df['end_area'] - analysis_df['start_area']) / analysis_df['start_area'] * 100
    
    # Add diabetes status
    analysis_df['is_diabetic'] = analysis_df['participant_id'].isin(diabetes_list)
    
    return analysis_df

def perform_statistical_tests(analysis_df: pd.DataFrame) -> dict:
    """Perform statistical tests comparing diabetic vs non-diabetic participants.
    
    Args:
        analysis_df: DataFrame with swelling analysis results
    
    Returns:
        Dictionary containing test results
    """
    diabetic = analysis_df[analysis_df['is_diabetic']]
    non_diabetic = analysis_df[~analysis_df['is_diabetic']]
    
    results = {}
    
    # Test for start to swollen change
    stat, p_val = stats.mannwhitneyu(
        diabetic['start_to_swollen_pct'], 
        non_diabetic['start_to_swollen_pct'],
        alternative='two-sided'
    )
    results['start_to_swollen'] = {'statistic': stat, 'p_value': p_val}
    
    # Test for swollen to end change
    stat, p_val = stats.mannwhitneyu(
        diabetic['swollen_to_end_pct'], 
        non_diabetic['swollen_to_end_pct'],
        alternative='two-sided'
    )
    results['swollen_to_end'] = {'statistic': stat, 'p_value': p_val}
    
    # Test for total change
    stat, p_val = stats.mannwhitneyu(
        diabetic['total_change_pct'], 
        non_diabetic['total_change_pct'],
        alternative='two-sided'
    )
    results['total_change'] = {'statistic': stat, 'p_value': p_val}
    
    return results

def create_individual_trajectory_plots(analysis_df: pd.DataFrame, bp_folder: str):
    """Create individual trajectory plots showing each participant's changes.
    
    Args:
        analysis_df: DataFrame with swelling analysis results
        bp_folder: Path to save plots
    """
    # Import color utility functions
    from src.tools.plotting_utils import create_monochromatic_palette, adjust_brightness_of_colors
    
    # Set up plotting style according to coding standards
    sns.set_style("whitegrid")
    source_sans = load_source_sans()
    
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 5,
        'lines.linewidth': 0.5
    })
    
    # Create figure with standard dimensions (matching swelling_plots size)
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 4.0))
    
    # Set up color scheme for diabetes (following codebase conventions)
    diabetes_base_color = '#ff7f0e'  # Orange for diabetes
    control_base_color = '#1f77b4'   # Blue for control
    
    # Create color palettes
    diabetes_palette = create_monochromatic_palette(diabetes_base_color)
    control_palette = create_monochromatic_palette(control_base_color)
    
    # Adjust brightness for better contrast
    diabetes_palette = adjust_brightness_of_colors(diabetes_palette, brightness_scale=0.1)
    control_palette = adjust_brightness_of_colors(control_palette, brightness_scale=0.1)
    
    # Use specific colors from the palettes
    diabetes_color = diabetes_palette[1]  # Darker orange
    control_color = control_palette[1]    # Darker blue
    diabetes_mean_color = diabetes_palette[4]  # Lighter orange for mean
    control_mean_color = control_palette[4]    # Lighter blue for mean
    
    timepoints = ['Start', 'Swollen', 'End']
    
    # Plot 1: Individual absolute area trajectories
    ax1 = axes[0, 0]
    diabetic_data = analysis_df[analysis_df['is_diabetic']]
    non_diabetic_data = analysis_df[~analysis_df['is_diabetic']]
    
    # Plot individual trajectories with proper alpha
    for _, row in analysis_df.iterrows():
        areas = [row['start_area'], row['swollen_area'], row['end_area']]
        color = diabetes_color if row['is_diabetic'] else control_color
        ax1.plot(timepoints, areas, marker='o', markersize=2, alpha=0.7, 
                color=color, linewidth=0.5)
    
    # Add group means with thicker lines
    diabetic_means = [diabetic_data['start_area'].mean(), 
                     diabetic_data['swollen_area'].mean(), 
                     diabetic_data['end_area'].mean()]
    non_diabetic_means = [non_diabetic_data['start_area'].mean(), 
                         non_diabetic_data['swollen_area'].mean(), 
                         non_diabetic_data['end_area'].mean()]
    
    ax1.plot(timepoints, diabetic_means, marker='s', markersize=4, 
            color=diabetes_mean_color, linewidth=2, label='Diabetic (mean)')
    ax1.plot(timepoints, non_diabetic_means, marker='s', markersize=4, 
            color=control_mean_color, linewidth=2, label='Control (mean)')
    
    ax1.set_xlabel('Timepoint', fontproperties=source_sans if source_sans else None)
    ax1.set_ylabel('Mask Area (pixels)', fontproperties=source_sans if source_sans else None)
    ax1.set_title('Individual Area Trajectories', fontproperties=source_sans if source_sans else None)
    ax1.legend(prop=source_sans if source_sans else None)
    
    # Plot 2: Individual percentage trajectories (normalized to start)
    ax2 = axes[0, 1]
    for _, row in analysis_df.iterrows():
        areas_pct = [100, 
                    (row['swollen_area'] / row['start_area']) * 100,
                    (row['end_area'] / row['start_area']) * 100]
        color = diabetes_color if row['is_diabetic'] else control_color
        ax2.plot(timepoints, areas_pct, marker='o', markersize=2, alpha=0.7, 
                color=color, linewidth=0.5)
    
    # Calculate proper percentage means for each participant
    diabetic_pct_means = [100,
                         (diabetic_data['swollen_area'] / diabetic_data['start_area']).mean() * 100,
                         (diabetic_data['end_area'] / diabetic_data['start_area']).mean() * 100]
    non_diabetic_pct_means = [100,
                             (non_diabetic_data['swollen_area'] / non_diabetic_data['start_area']).mean() * 100,
                             (non_diabetic_data['end_area'] / non_diabetic_data['start_area']).mean() * 100]
    
    ax2.plot(timepoints, diabetic_pct_means, marker='s', markersize=4, 
            color=diabetes_mean_color, linewidth=2, label='Diabetic (mean)')
    ax2.plot(timepoints, non_diabetic_pct_means, marker='s', markersize=4, 
            color=control_mean_color, linewidth=2, label='Control (mean)')
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    
    ax2.set_xlabel('Timepoint', fontproperties=source_sans if source_sans else None)
    ax2.set_ylabel('Area (% of Start)', fontproperties=source_sans if source_sans else None)
    ax2.set_title('Individual Percentage Trajectories', fontproperties=source_sans if source_sans else None)
    ax2.legend(prop=source_sans if source_sans else None)
    
    # Plot 3: Individual percentage changes (start to swollen)
    ax3 = axes[1, 0]
    participants = analysis_df['participant_id'].tolist()
    for _, row in analysis_df.iterrows():
        participant = row['participant_id']
        change = row['start_to_swollen_pct']
        color = diabetes_color if row['is_diabetic'] else control_color
        ax3.bar(participant, change, color=color, alpha=0.7, width=0.6)
    
    ax3.set_xlabel('Participant ID', fontproperties=source_sans if source_sans else None)
    ax3.set_ylabel('Start to Swollen Change (%)', fontproperties=source_sans if source_sans else None)
    ax3.set_title('Individual Start to Swollen Changes', fontproperties=source_sans if source_sans else None)
    ax3.tick_params(axis='x', rotation=45, labelsize=5)
    
    # Plot 4: Individual total percentage changes
    ax4 = axes[1, 1]
    for _, row in analysis_df.iterrows():
        participant = row['participant_id']
        change = row['total_change_pct']
        color = diabetes_color if row['is_diabetic'] else control_color
        ax4.bar(participant, change, color=color, alpha=0.7, width=0.6)
    
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax4.set_xlabel('Participant ID', fontproperties=source_sans if source_sans else None)
    ax4.set_ylabel('Total Change (%)', fontproperties=source_sans if source_sans else None)
    ax4.set_title('Individual Total Changes', fontproperties=source_sans if source_sans else None)
    ax4.tick_params(axis='x', rotation=45, labelsize=5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(bp_folder, 'individual_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(bp_folder, 'individual_trajectories.pdf'), bbox_inches='tight')
    plt.close()

def create_swelling_plots(analysis_df: pd.DataFrame, bp_folder: str):
    """Create visualization plots for swelling analysis.
    
    Args:
        analysis_df: DataFrame with swelling analysis results
        bp_folder: Path to save plots
    """
    # Import color utility functions
    from src.tools.plotting_utils import create_monochromatic_palette, adjust_brightness_of_colors
    
    # Set up plotting style
    sns.set_style("whitegrid")
    source_sans = load_source_sans()
    
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 5,
        'lines.linewidth': 0.5
    })
    
    # Set up color scheme for consistency with individual plots
    diabetes_base_color = '#ff7f0e'  # Orange for diabetes
    control_base_color = '#1f77b4'   # Blue for control
    
    # Create color palettes
    diabetes_palette = create_monochromatic_palette(diabetes_base_color)
    control_palette = create_monochromatic_palette(control_base_color)
    
    # Adjust brightness for better contrast
    diabetes_palette = adjust_brightness_of_colors(diabetes_palette, brightness_scale=0.1)
    control_palette = adjust_brightness_of_colors(control_palette, brightness_scale=0.1)
    
    # Use specific colors from the palettes
    diabetes_color = diabetes_palette[2]  # Medium orange
    control_color = control_palette[2]    # Medium blue
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 4.0))
    
    # Set up color palette for boxplots (Control=Blue, Diabetic=Orange)
    box_colors = [control_color, diabetes_color]
    
    # Plot 1: Start to swollen percentage change
    ax1 = axes[0, 0]
    sns.boxplot(data=analysis_df, x='is_diabetic', y='start_to_swollen_pct', ax=ax1, palette=box_colors)
    ax1.set_xlabel('Diabetic Status', fontproperties=source_sans if source_sans else None)
    ax1.set_ylabel('Start to Swollen Change (%)', fontproperties=source_sans if source_sans else None)
    ax1.set_xticklabels(['Control', 'Diabetic'])
    
    # Plot 2: Swollen to end percentage change
    ax2 = axes[0, 1]
    sns.boxplot(data=analysis_df, x='is_diabetic', y='swollen_to_end_pct', ax=ax2, palette=box_colors)
    ax2.set_xlabel('Diabetic Status', fontproperties=source_sans if source_sans else None)
    ax2.set_ylabel('Swollen to End Change (%)', fontproperties=source_sans if source_sans else None)
    ax2.set_xticklabels(['Control', 'Diabetic'])
    
    # Plot 3: Total change percentage
    ax3 = axes[1, 0]
    sns.boxplot(data=analysis_df, x='is_diabetic', y='total_change_pct', ax=ax3, palette=box_colors)
    ax3.set_xlabel('Diabetic Status', fontproperties=source_sans if source_sans else None)
    ax3.set_ylabel('Total Change (%)', fontproperties=source_sans if source_sans else None)
    ax3.set_xticklabels(['Control', 'Diabetic'])
    
    # Plot 4: Area trajectory for each group
    ax4 = axes[1, 1]
    
    # Prepare data for trajectory plot
    diabetic_data = analysis_df[analysis_df['is_diabetic']]
    non_diabetic_data = analysis_df[~analysis_df['is_diabetic']]
    
    timepoints = ['start', 'swollen', 'end']
    
    # Calculate means and standard errors for each group
    diabetic_means = [
        diabetic_data['start_area'].mean(),
        diabetic_data['swollen_area'].mean(),
        diabetic_data['end_area'].mean()
    ]
    diabetic_sems = [
        diabetic_data['start_area'].sem(),
        diabetic_data['swollen_area'].sem(),
        diabetic_data['end_area'].sem()
    ]
    
    non_diabetic_means = [
        non_diabetic_data['start_area'].mean(),
        non_diabetic_data['swollen_area'].mean(),
        non_diabetic_data['end_area'].mean()
    ]
    non_diabetic_sems = [
        non_diabetic_data['start_area'].sem(),
        non_diabetic_data['swollen_area'].sem(),
        non_diabetic_data['end_area'].sem()
    ]
    
    ax4.errorbar(timepoints, diabetic_means, yerr=diabetic_sems, 
                label='Diabetic', marker='o', capsize=3, color=diabetes_color, linewidth=2)
    ax4.errorbar(timepoints, non_diabetic_means, yerr=non_diabetic_sems, 
                label='Control', marker='s', capsize=3, color=control_color, linewidth=2)
    ax4.set_xlabel('Timepoint', fontproperties=source_sans if source_sans else None)
    ax4.set_ylabel('Mean Mask Area (pixels)', fontproperties=source_sans if source_sans else None)
    ax4.legend(prop=source_sans if source_sans else None)
    
    plt.tight_layout()
    plt.savefig(os.path.join(bp_folder, 'swelling_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(bp_folder, 'swelling_analysis.pdf'), bbox_inches='tight')
    plt.close()

def create_percentage_trajectory_plot(analysis_df: pd.DataFrame, bp_folder: str):
    """Create a standalone percentage trajectory plot showing group means and error bars.
    
    Args:
        analysis_df: DataFrame with swelling analysis results
        bp_folder: Path to save plots
    """
    # Import color utility functions
    from src.tools.plotting_utils import create_monochromatic_palette, adjust_brightness_of_colors
    
    # Set up plotting style according to coding standards
    sns.set_style("whitegrid")
    source_sans = load_source_sans()
    
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 5,
        'lines.linewidth': 0.5
    })
    
    # Set up color scheme consistent with other plots
    diabetes_base_color = '#ff7f0e'  # Orange for diabetes
    control_base_color = '#1f77b4'   # Blue for control
    
    # Create color palettes
    diabetes_palette = create_monochromatic_palette(diabetes_base_color)
    control_palette = create_monochromatic_palette(control_base_color)
    
    # Adjust brightness for better contrast
    diabetes_palette = adjust_brightness_of_colors(diabetes_palette, brightness_scale=0.1)
    control_palette = adjust_brightness_of_colors(control_palette, brightness_scale=0.1)
    
    # Use specific colors from the palettes
    diabetes_color = diabetes_palette[2]  # Medium orange
    control_color = control_palette[2]    # Medium blue
    
    # Create figure with standard dimensions
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    
    # Prepare data for trajectory plot
    diabetic_data = analysis_df[analysis_df['is_diabetic']]
    non_diabetic_data = analysis_df[~analysis_df['is_diabetic']]
    
    timepoints = ['Start', 'Swollen', 'End']
    
    # Calculate percentage means and standard errors for each group
    # For diabetic group
    diabetic_start_pct = 100  # Always 100% at start
    diabetic_swollen_pcts = (diabetic_data['swollen_area'] / diabetic_data['start_area']) * 100
    diabetic_end_pcts = (diabetic_data['end_area'] / diabetic_data['start_area']) * 100
    
    diabetic_pct_means = [
        diabetic_start_pct,
        diabetic_swollen_pcts.mean(),
        diabetic_end_pcts.mean()
    ]
    diabetic_pct_sems = [
        0,  # No error at start (all start at 100%)
        diabetic_swollen_pcts.sem(),
        diabetic_end_pcts.sem()
    ]
    
    # For non-diabetic group
    non_diabetic_start_pct = 100  # Always 100% at start
    non_diabetic_swollen_pcts = (non_diabetic_data['swollen_area'] / non_diabetic_data['start_area']) * 100
    non_diabetic_end_pcts = (non_diabetic_data['end_area'] / non_diabetic_data['start_area']) * 100
    
    non_diabetic_pct_means = [
        non_diabetic_start_pct,
        non_diabetic_swollen_pcts.mean(),
        non_diabetic_end_pcts.mean()
    ]
    non_diabetic_pct_sems = [
        0,  # No error at start (all start at 100%)
        non_diabetic_swollen_pcts.sem(),
        non_diabetic_end_pcts.sem()
    ]
    
    # Plot the percentage trajectories with error bars
    ax.errorbar(timepoints, diabetic_pct_means, yerr=diabetic_pct_sems, 
               label='Diabetic', marker='o', capsize=3, color=diabetes_color, linewidth=2)
    ax.errorbar(timepoints, non_diabetic_pct_means, yerr=non_diabetic_pct_sems, 
               label='Control', marker='s', capsize=3, color=control_color, linewidth=2)
    
    # Add a reference line at 100%
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    
    # Set labels and formatting
    ax.set_xlabel('Timepoint', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Area (% of Start)', fontproperties=source_sans if source_sans else None)
    ax.set_title('Mean Area Percentage Trajectories', fontproperties=source_sans if source_sans else None)
    ax.legend(prop=source_sans if source_sans else None)
    
    plt.tight_layout()
    plt.savefig(os.path.join(bp_folder, 'percentage_trajectory.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(bp_folder, 'percentage_trajectory.pdf'), bbox_inches='tight')
    plt.close()

def main():
    bp_folder = os.path.join(PATHS['cap_flow'], 'data', 'bp-analysis')
    bp_image_folder = os.path.join(bp_folder, 'images')
    bp_mask_folder = os.path.join(bp_folder, 'masks')

    # make a list of all the images in the folder
    bp_image_files = [f for f in os.listdir(bp_image_folder) if f.endswith('.tif')]

    # make a list of all the masks in the folder
    bp_mask_files = [f for f in os.listdir(bp_mask_folder) if (f.endswith('.TIF') or f.endswith('.tiff') or f.endswith('.tif'))]

    # make a list of matching image and mask files
    bp_image_files = [f for f in bp_image_files if f.replace('.tif', '_mask.TIF') in bp_mask_files]


    mask_df = pd.DataFrame(columns=['participant_id', 'timepoint', 'mask_area'])
    for bp_image_file in bp_image_files:
        # load the image and mask files
        bp_image = cv2.imread(os.path.join(bp_image_folder, bp_image_file), cv2.IMREAD_GRAYSCALE)
        bp_mask = cv2.imread(os.path.join(bp_mask_folder, bp_image_file.replace('.tif', '_mask.TIF')), cv2.IMREAD_GRAYSCALE)
        # the mask area is the number of non-zero pixels in the mask
        mask_area = np.sum(bp_mask > 12)

        # label the image and mask
        participant_id = bp_image_file.split('_')[0]
        timepoint = bp_image_file.split('_')[1].replace('.tif', '')
        mask_df.loc[len(mask_df)] = [participant_id, timepoint, mask_area]

    # Filter out participants in the ignore list
    print(f"Filtering out ignored participants: {ignore_list}")
    original_count = len(mask_df)
    mask_df = mask_df[~mask_df['participant_id'].isin(ignore_list)]
    filtered_count = len(mask_df)
    print(f"Removed {original_count - filtered_count} entries from {len(ignore_list)} ignored participants")

    # save the mask dictionary
    mask_df.to_csv(os.path.join(bp_folder, 'bp_mask_df.csv'), index=False)

    # Perform swelling analysis
    print("Performing swelling analysis...")
    analysis_df = analyze_swelling_changes(mask_df)
    
    # Perform statistical tests
    stat_results = perform_statistical_tests(analysis_df)
    
    # Add statistical significance to analysis dataframe
    analysis_df_with_stats = analysis_df.copy()
    
    # Create a summary statistics dataframe
    summary_stats = pd.DataFrame({
        'metric': ['start_to_swollen_pct', 'swollen_to_end_pct', 'total_change_pct'],
        'diabetic_mean': [
            analysis_df[analysis_df['is_diabetic']]['start_to_swollen_pct'].mean(),
            analysis_df[analysis_df['is_diabetic']]['swollen_to_end_pct'].mean(),
            analysis_df[analysis_df['is_diabetic']]['total_change_pct'].mean()
        ],
        'diabetic_std': [
            analysis_df[analysis_df['is_diabetic']]['start_to_swollen_pct'].std(),
            analysis_df[analysis_df['is_diabetic']]['swollen_to_end_pct'].std(),
            analysis_df[analysis_df['is_diabetic']]['total_change_pct'].std()
        ],
        'non_diabetic_mean': [
            analysis_df[~analysis_df['is_diabetic']]['start_to_swollen_pct'].mean(),
            analysis_df[~analysis_df['is_diabetic']]['swollen_to_end_pct'].mean(),
            analysis_df[~analysis_df['is_diabetic']]['total_change_pct'].mean()
        ],
        'non_diabetic_std': [
            analysis_df[~analysis_df['is_diabetic']]['start_to_swollen_pct'].std(),
            analysis_df[~analysis_df['is_diabetic']]['swollen_to_end_pct'].std(),
            analysis_df[~analysis_df['is_diabetic']]['total_change_pct'].std()
        ],
        'p_value': [
            stat_results['start_to_swollen']['p_value'],
            stat_results['swollen_to_end']['p_value'],
            stat_results['total_change']['p_value']
        ],
        'significance': [
            "***" if stat_results['start_to_swollen']['p_value'] < 0.001 else "**" if stat_results['start_to_swollen']['p_value'] < 0.01 else "*" if stat_results['start_to_swollen']['p_value'] < 0.05 else "ns",
            "***" if stat_results['swollen_to_end']['p_value'] < 0.001 else "**" if stat_results['swollen_to_end']['p_value'] < 0.01 else "*" if stat_results['swollen_to_end']['p_value'] < 0.05 else "ns",
            "***" if stat_results['total_change']['p_value'] < 0.001 else "**" if stat_results['total_change']['p_value'] < 0.01 else "*" if stat_results['total_change']['p_value'] < 0.05 else "ns"
        ]
    })
    
    # Save analysis results
    analysis_df.to_csv(os.path.join(bp_folder, 'swelling_analysis.csv'), index=False)
    summary_stats.to_csv(os.path.join(bp_folder, 'swelling_summary_statistics.csv'), index=False)
    
    # Print summary statistics
    print("\n=== SWELLING ANALYSIS SUMMARY ===")
    print(f"Total participants: {len(analysis_df)}")
    print(f"Diabetic participants: {analysis_df['is_diabetic'].sum()}")
    print(f"Non-diabetic participants: {(~analysis_df['is_diabetic']).sum()}")
    
    print("\n=== PERCENTAGE CHANGES (Mean ± SD) ===")
    diabetic_data = analysis_df[analysis_df['is_diabetic']]
    non_diabetic_data = analysis_df[~analysis_df['is_diabetic']]
    
    print("\nStart to Swollen Change:")
    print(f"Diabetic: {diabetic_data['start_to_swollen_pct'].mean():.2f} ± {diabetic_data['start_to_swollen_pct'].std():.2f}%")
    print(f"Non-diabetic: {non_diabetic_data['start_to_swollen_pct'].mean():.2f} ± {non_diabetic_data['start_to_swollen_pct'].std():.2f}%")
    
    print("\nSwollen to End Change:")
    print(f"Diabetic: {diabetic_data['swollen_to_end_pct'].mean():.2f} ± {diabetic_data['swollen_to_end_pct'].std():.2f}%")
    print(f"Non-diabetic: {non_diabetic_data['swollen_to_end_pct'].mean():.2f} ± {non_diabetic_data['swollen_to_end_pct'].std():.2f}%")
    
    print("\nTotal Change:")
    print(f"Diabetic: {diabetic_data['total_change_pct'].mean():.2f} ± {diabetic_data['total_change_pct'].std():.2f}%")
    print(f"Non-diabetic: {non_diabetic_data['total_change_pct'].mean():.2f} ± {non_diabetic_data['total_change_pct'].std():.2f}%")
    
    print("\n=== STATISTICAL TEST RESULTS (Mann-Whitney U) ===")
    for test_name, result in stat_results.items():
        significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
        print(f"{test_name.replace('_', ' ').title()}: p = {result['p_value']:.4f} {significance}")
    
    # Create visualization plots
    print("\nGenerating plots...")
    create_swelling_plots(analysis_df, bp_folder)
    create_individual_trajectory_plots(analysis_df, bp_folder)
    create_percentage_trajectory_plot(analysis_df, bp_folder)
    print(f"Plots saved in {bp_folder}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()