"""
Filename: src/analysis/age_histogram.py

File for creating histograms of age data from capillary velocity studies.

This script:
1. Loads the same data as bp_threshold.py
2. Creates histograms of participant age
3. Applies the same styling and font configuration
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from typing import Optional
import argparse

# Import paths from config instead of defining computer paths locally
from src.config import PATHS, load_source_sans

cap_flow_path = PATHS['cap_flow']
source_sans = load_source_sans()

def create_age_histogram(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """
    Creates a histogram of age data.
    
    Args:
        df: DataFrame containing Age column
        output_dir: Directory to save the plot (optional)
    """
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'AgeHistograms')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating age histogram...")
    
    # Standard plot configuration with robust font loading
    sns.set_style("whitegrid")
    
    plt.rcParams.update({
        'pdf.fonttype': 42,  # For editable text in PDFs
        'ps.fonttype': 42,   # For editable text in PostScript
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 5,
        'lines.linewidth': 0.5
    })
    
    # Filter out missing age data
    age_df = df.dropna(subset=['Age']).copy()

    # keep only one row per participant
    age_df = age_df.drop_duplicates(subset=['Participant'])
    
    if len(age_df) == 0:
        print(f"Error: No valid age data found.")
        return
    
    # Create figure
    plt.figure(figsize=(3.5, 2.5))
    
    # Create histogram
    age_values = age_df['Age']
    
    # Use 5-year bins
    age_min = int(age_values.min())
    age_max = int(age_values.max())
    # Create bins every 5 years, starting from the nearest 5-year boundary below min
    bin_start = (age_min // 5) * 5
    bin_end = ((age_max // 5) + 1) * 5
    bins = np.arange(bin_start, bin_end + 5, 5)
    
    plt.hist(age_values, bins=bins, alpha=0.7, color='#1f77b4', 
             edgecolor='black', linewidth=0.5)
    
    # Add statistics to the plot
    mean_age = age_values.mean()
    median_age = age_values.median()
    std_age = age_values.std()
    
    # # Add vertical lines for mean and median
    # plt.axvline(mean_age, color='red', linestyle='--', linewidth=1, 
    #             label=f'Mean: {mean_age:.1f} years')
    # plt.axvline(median_age, color='orange', linestyle='--', linewidth=1,
    #             label=f'Median: {median_age:.1f} years')
    
    # # Add text box with statistics
    # stats_text = f'n = {len(age_values)}\nMean = {mean_age:.1f} ± {std_age:.1f} years\nMedian = {median_age:.1f} years'
    # plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
    #          fontsize=6, verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set title and labels with font handling
    if source_sans:
        plt.title('Age Distribution', 
                 fontproperties=source_sans, fontsize=8)
        plt.xlabel('Age (years)', fontproperties=source_sans)
        plt.ylabel('Frequency', fontproperties=source_sans)
        # plt.legend(prop=source_sans)
    else:
        plt.title('Age Distribution', fontsize=8)
        plt.xlabel('Age (years)')
        plt.ylabel('Frequency')
        # plt.legend()
    
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    filename = 'age_histogram.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram saved to: {os.path.join(output_dir, filename)}")
    print(f"Age statistics:")
    print(f"  Range: {age_values.min():.1f} - {age_values.max():.1f} years")
    print(f"  Mean: {mean_age:.1f} ± {std_age:.1f} years")
    print(f"  Median: {median_age:.1f} years")
    print(f"  Sample size: {len(age_values)}")

def create_age_groups_histogram(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """
    Creates a histogram of age data with age group categories.
    
    Args:
        df: DataFrame containing Age column
        output_dir: Directory to save the plot (optional)
    """
    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'AgeHistograms')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating age groups histogram...")
    
    # Standard plot configuration
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    # Filter out missing age data
    age_df = df.dropna(subset=['Age']).copy()

    # keep only one row per participant
    age_df = age_df.drop_duplicates(subset=['Participant'])
    
    if len(age_df) == 0:
        print("Error: No valid age data found.")
        return
    
    # Create 5-year age groups
    age_min = int(age_df['Age'].min())
    age_max = int(age_df['Age'].max())
    # Create bins every 5 years
    bin_start = (age_min // 5) * 5
    bin_end = ((age_max // 5) + 1) * 5
    bins = list(range(bin_start, bin_end + 5, 5))
    
    # Create labels for 5-year groups
    labels = []
    for i in range(len(bins)-1):
        if i == len(bins)-2:  # Last bin
            labels.append(f'{bins[i]}+')
        else:
            labels.append(f'{bins[i]}-{bins[i+1]-1}')
    
    age_df['Age_Group'] = pd.cut(
        age_df['Age'], 
        bins=bins, 
        labels=labels,
        include_lowest=True
    )
    
    # Create figure
    plt.figure(figsize=(4, 2.5))
    
    # Create histogram by age groups
    age_group_counts = age_df['Age_Group'].value_counts().sort_index()
    
    # Create bar plot
    ax = age_group_counts.plot(kind='bar', color='#1f77b4', alpha=0.7, 
                              edgecolor='black', linewidth=0.5)
    
    # Set title and labels with font handling
    if source_sans:
        ax.set_title('Age Distribution by Groups', fontproperties=source_sans, fontsize=8)
        ax.set_xlabel('Age Group (years)', fontproperties=source_sans)
        ax.set_ylabel('Frequency', fontproperties=source_sans)
    else:
        ax.set_title('Age Distribution by Groups', fontsize=8)
        ax.set_xlabel('Age Group (years)')
        ax.set_ylabel('Frequency')
    
    # Add count labels on bars
    for i, v in enumerate(age_group_counts.values):
        ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=6)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    filename = 'age_groups_histogram.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Age groups histogram saved to: {os.path.join(output_dir, filename)}")
    print(f"Age group distribution:")
    for group, count in age_group_counts.items():
        print(f"  {group}: {count} participants")

def create_age_by_sex_histogram(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """
    Creates a histogram of age data split by sex.
    
    Args:
        df: DataFrame containing Age and Sex columns
        output_dir: Directory to save the plot (optional)
    """
    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'AgeHistograms')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating age by sex histogram...")
    
    # Standard plot configuration
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    # Filter out missing data
    age_sex_df = df.dropna(subset=['Age', 'Sex']).copy()

    # keep only one row per participant
    age_sex_df = age_sex_df.drop_duplicates(subset=['Participant'])
    
    if len(age_sex_df) == 0:
        print("Error: No valid age and sex data found.")
        return
    
    # Create figure
    plt.figure(figsize=(4, 2.5))
    
    # Get age data by sex
    male_ages = age_sex_df[age_sex_df['Sex'] == 'M']['Age']
    female_ages = age_sex_df[age_sex_df['Sex'] == 'F']['Age']
    
    # Use 5-year bins
    all_ages = age_sex_df['Age']
    age_min = int(all_ages.min())
    age_max = int(all_ages.max())
    # Create bins every 5 years
    bin_start = (age_min // 5) * 5
    bin_end = ((age_max // 5) + 1) * 5
    bins = np.arange(bin_start, bin_end + 5, 5)
    
    # Create overlapping histograms
    plt.hist(male_ages, bins=bins, alpha=0.6, color='#1f77b4', 
             label=f'Male (n={len(male_ages)})', edgecolor='black', linewidth=0.5)
    plt.hist(female_ages, bins=bins, alpha=0.6, color='#ff7f0e',
             label=f'Female (n={len(female_ages)})', edgecolor='black', linewidth=0.5)
    
    # Set title and labels with font handling
    if source_sans:
        plt.title('Age Distribution by Sex', fontproperties=source_sans, fontsize=8)
        plt.xlabel('Age (years)', fontproperties=source_sans)
        plt.ylabel('Frequency', fontproperties=source_sans)
        plt.legend(prop=source_sans)
    else:
        plt.title('Age Distribution by Sex', fontsize=8)
        plt.xlabel('Age (years)')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    filename = 'age_by_sex_histogram.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Age by sex histogram saved to: {os.path.join(output_dir, filename)}")
    print(f"Age statistics by sex:")
    if len(male_ages) > 0:
        print(f"  Male - Mean: {male_ages.mean():.1f} ± {male_ages.std():.1f}, Median: {male_ages.median():.1f}")
    if len(female_ages) > 0:
        print(f"  Female - Mean: {female_ages.mean():.1f} ± {female_ages.std():.1f}, Median: {female_ages.median():.1f}")

def main(histogram_type: str = 'all'):
    """
    Main function for creating age histograms.
    
    Args:
        histogram_type: Type of histogram to create ('basic', 'groups', 'sex', or 'all')
    """
    print(f"\nCreating age histogram(s)...")
    
    # Load data - same approach as bp_threshold.py
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Filter for control group - same as bp_threshold.py
    # controls_df = df[df['SET'] == 'set01']
    controls_df = df
    
    # print(f"Using control group data (SET == 'set01'): {len(controls_df)} samples")
    
    # Check if Age column exists
    if 'Age' not in controls_df.columns:
        print("Error: Age column not found in the data.")
        return 1
    
    # Create output directory
    output_dir = os.path.join(cap_flow_path, 'results', 'AgeHistograms')
    
    # Create histograms based on request
    if histogram_type in ['basic', 'all']:
        create_age_histogram(controls_df, output_dir)
    
    if histogram_type in ['groups', 'all']:
        create_age_groups_histogram(controls_df, output_dir)
    
    if histogram_type in ['sex', 'all'] and 'Sex' in controls_df.columns:
        create_age_by_sex_histogram(controls_df, output_dir)
    elif histogram_type == 'sex':
        print("Warning: Sex column not found, skipping sex-based histogram")
    
    print(f"\nAge histogram analysis complete.")
    print(f"Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create histograms of age data.')
    parser.add_argument('--type', choices=['basic', 'groups', 'sex', 'all'], default='all',
                       help='Type of histogram to create (default: all)')
    
    args = parser.parse_args()
    
    main(histogram_type=args.type)
