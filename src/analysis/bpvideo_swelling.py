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

diabetes_list = ['part78','part79', 'part76', 'part73', 'part74', 'part75', 'part70']

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

def create_swelling_plots(analysis_df: pd.DataFrame, bp_folder: str):
    """Create visualization plots for swelling analysis.
    
    Args:
        analysis_df: DataFrame with swelling analysis results
        bp_folder: Path to save plots
    """
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
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 4.0))
    
    # Plot 1: Start to swollen percentage change
    ax1 = axes[0, 0]
    sns.boxplot(data=analysis_df, x='is_diabetic', y='start_to_swollen_pct', ax=ax1)
    ax1.set_xlabel('Diabetic Status', fontproperties=source_sans if source_sans else None)
    ax1.set_ylabel('Start to Swollen Change (%)', fontproperties=source_sans if source_sans else None)
    ax1.set_xticklabels(['Non-Diabetic', 'Diabetic'])
    
    # Plot 2: Swollen to end percentage change
    ax2 = axes[0, 1]
    sns.boxplot(data=analysis_df, x='is_diabetic', y='swollen_to_end_pct', ax=ax2)
    ax2.set_xlabel('Diabetic Status', fontproperties=source_sans if source_sans else None)
    ax2.set_ylabel('Swollen to End Change (%)', fontproperties=source_sans if source_sans else None)
    ax2.set_xticklabels(['Non-Diabetic', 'Diabetic'])
    
    # Plot 3: Total change percentage
    ax3 = axes[1, 0]
    sns.boxplot(data=analysis_df, x='is_diabetic', y='total_change_pct', ax=ax3)
    ax3.set_xlabel('Diabetic Status', fontproperties=source_sans if source_sans else None)
    ax3.set_ylabel('Total Change (%)', fontproperties=source_sans if source_sans else None)
    ax3.set_xticklabels(['Non-Diabetic', 'Diabetic'])
    
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
                label='Diabetic', marker='o', capsize=3)
    ax4.errorbar(timepoints, non_diabetic_means, yerr=non_diabetic_sems, 
                label='Non-Diabetic', marker='s', capsize=3)
    ax4.set_xlabel('Timepoint', fontproperties=source_sans if source_sans else None)
    ax4.set_ylabel('Mean Mask Area (pixels)', fontproperties=source_sans if source_sans else None)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(bp_folder, 'swelling_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(bp_folder, 'swelling_analysis.pdf'), bbox_inches='tight')
    plt.close()

def main():
    bp_folder = os.path.join(PATHS['cap_flow'], 'data', 'bp-analysis')
    bp_image_folder = os.path.join(bp_folder, 'images')
    bp_mask_folder = os.path.join(bp_folder, 'masks')

    # make a list of all the images in the folder
    bp_image_files = [f for f in os.listdir(bp_image_folder) if f.endswith('.tif')]

    # make a list of all the masks in the folder
    bp_mask_files = [f for f in os.listdir(bp_mask_folder) if f.endswith('.TIF')]

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

    # save the mask dictionary
    mask_df.to_csv(os.path.join(bp_folder, 'bp_mask_df.csv'), index=False)

    # Perform swelling analysis
    print("Performing swelling analysis...")
    analysis_df = analyze_swelling_changes(mask_df)
    
    # Save analysis results
    analysis_df.to_csv(os.path.join(bp_folder, 'swelling_analysis.csv'), index=False)
    
    # Perform statistical tests
    stat_results = perform_statistical_tests(analysis_df)
    
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
    print(f"Plots saved in {bp_folder}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()