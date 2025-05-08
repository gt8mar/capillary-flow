"""
Filename: src/analysis/sex_analysis.py

File for analyzing sex-based differences in capillary velocity data.

This script:
1. Compares velocity distributions between male and female participants
2. Reports detailed statistics (median, interquartile range) for each sex
3. Creates visualization of the differences
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.font_manager import FontProperties
from typing import Tuple, List, Dict, Optional
import argparse

# Import paths from config instead of defining computer paths locally
from src.config import PATHS, load_source_sans

cap_flow_path = PATHS['cap_flow']
source_sans = load_source_sans()

def create_sex_cdf_plot(df: pd.DataFrame, output_path: str, ax=None) -> Optional[float]:
    """
    Creates a CDF plot for velocities split by sex.
    
    Args:
        df: DataFrame containing Sex and Video_Median_Velocity columns
        output_path: Path for finding fonts and saving results
        ax: Matplotlib axis object to plot on (optional)
    
    Returns:
        KS statistic measuring the difference between distributions
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.0))
    
    # Group data by sex
    male_group = df[df['Sex'] == 'M']['Video_Median_Velocity']
    female_group = df[df['Sex'] == 'F']['Video_Median_Velocity']
    
    # Check if we have enough data in both groups
    if len(male_group) < 3 or len(female_group) < 3:
        print("Warning: Not enough data for one or both sex groups")
        return None
    
    # Print detailed statistics for each group
    print("\nDetailed statistics by sex:")
    print(f"Male participants (n={len(male_group)}):")
    print(f"  Median velocity: {male_group.median():.2f} mm/s")
    print(f"  Mean velocity: {male_group.mean():.2f} mm/s")
    print(f"  Standard deviation: {male_group.std():.2f} mm/s")
    print(f"  Interquartile range: {male_group.quantile(0.25):.2f} to {male_group.quantile(0.75):.2f} mm/s")
    print(f"  Range: {male_group.min():.2f} to {male_group.max():.2f} mm/s")
    
    print(f"\nFemale participants (n={len(female_group)}):")
    print(f"  Median velocity: {female_group.median():.2f} mm/s")
    print(f"  Mean velocity: {female_group.mean():.2f} mm/s")
    print(f"  Standard deviation: {female_group.std():.2f} mm/s")
    print(f"  Interquartile range: {female_group.quantile(0.25):.2f} to {female_group.quantile(0.75):.2f} mm/s")
    print(f"  Range: {female_group.min():.2f} to {female_group.max():.2f} mm/s")
    
    # Calculate empirical CDFs
    x_male = np.sort(male_group)
    y_male = np.arange(1, len(x_male) + 1) / len(x_male)
    
    x_female = np.sort(female_group)
    y_female = np.arange(1, len(x_female) + 1) / len(x_female)
    
    # Plot CDFs
    ax.plot(x_male, y_male, 'b-', linewidth=1, label=f'Male (n={len(male_group)})')
    ax.plot(x_female, y_female, 'r-', linewidth=1, label=f'Female (n={len(female_group)})')
    
    # Run KS test
    ks_stat, p_value = stats.ks_2samp(male_group, female_group)
    
    # Add KS test result to plot
    p_text = f"p = {p_value:.3f}" if p_value >= 0.001 else "p < 0.001"
    ax.text(0.05, 0.95, f"KS stat: {ks_stat:.3f}\n{p_text}", 
           transform=ax.transAxes, fontsize=6, va='top')
    
    print(f"\nStatistical comparison:")
    print(f"KS statistic: {ks_stat:.3f}")
    print(f"p-value: {p_value:.6f}")
    
    # Run Mann-Whitney U test (for comparing medians)
    u_stat, p_value_mw = stats.mannwhitneyu(male_group, female_group)
    print(f"Mann-Whitney U test p-value: {p_value_mw:.6f}")
    
    percent_diff = ((female_group.median() - male_group.median()) / male_group.median()) * 100
    print(f"Female median is {abs(percent_diff):.1f}% {'higher' if percent_diff > 0 else 'lower'} than male median\n")
    
    # Try to use Source Sans font if available
    try:
        source_sans = FontProperties(fname=os.path.join(PATHS['downloads'], 
                                                       'Source_Sans_3', 'static', 
                                                       'SourceSans3-Regular.ttf'))
        ax.set_xlabel('Video Median Velocity (mm/s)', fontproperties=source_sans)
        ax.set_ylabel('Cumulative Probability', fontproperties=source_sans)
        ax.set_title('Velocity CDF by Sex', fontproperties=source_sans)
        ax.legend(prop=source_sans)
    except:
        # Fall back to default font
        ax.set_xlabel('Video Median Velocity (mm/s)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Velocity CDF by Sex')
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    return ks_stat

def plot_velocity_boxplots(df: pd.DataFrame) -> None:
    """
    Creates boxplots of video median velocities grouped by sex.
    
    Args:
        df: DataFrame containing Sex and Video_Median_Velocity columns
    """
    # Create output directory for plots
    output_dir = os.path.join(cap_flow_path, 'results', 'SexAnalysis')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nCreating velocity boxplots by sex...")
    
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
    
    # Create boxplot by sex
    plt.figure(figsize=(2.4, 2.0))
    
    # Ensure we have clean sex data
    sex_df = df.dropna(subset=['Video_Median_Velocity', 'Sex']).copy()
    
    # Create the boxplot
    ax = sns.boxplot(
        x='Sex',
        y='Video_Median_Velocity',
        data=sex_df,
        color='#1f77b4',
        width=0.6,
        fliersize=3
    )
    
    # Set title and labels
    if source_sans:
        ax.set_title('Video Median Velocity by Sex', fontproperties=source_sans, fontsize=8)
        ax.set_xlabel('Sex', fontproperties=source_sans)
        ax.set_ylabel('Median Velocity (mm/s)', fontproperties=source_sans)
    else:
        ax.set_title('Video Median Velocity by Sex', fontsize=8)
        ax.set_xlabel('Sex')
        ax.set_ylabel('Median Velocity (mm/s)')
    
    # Get counts for each sex group
    counts = sex_df['Sex'].value_counts().sort_index()
    
    # Add counts to the x-tick labels
    xtick_labels = [f"{label.get_text()}\n(n={counts[label.get_text()]})" 
                  if label.get_text() in counts.index else label.get_text()
                  for label in ax.get_xticklabels()]
    ax.set_xticklabels(xtick_labels)
    
    # Add statistical annotation
    groups = sex_df.groupby('Sex')['Video_Median_Velocity'].apply(list).to_dict()
    if len(groups) == 2:
        group_values = list(groups.values())
        stat, p_value = stats.mannwhitneyu(group_values[0], group_values[1])
        
        # Add p-value annotation
        p_text = f"p = {p_value:.3f}" if p_value >= 0.001 else "p < 0.001"
        if source_sans:
            ax.text(0.5, 0.95, p_text, transform=ax.transAxes, 
                   ha='center', fontproperties=source_sans, fontsize=6)
        else:
            ax.text(0.5, 0.95, p_text, transform=ax.transAxes, 
                   ha='center', fontsize=6)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_by_sex.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a violin plot for more distribution detail
    plt.figure(figsize=(2.4, 2.0))
    ax = sns.violinplot(
        x='Sex', 
        y='Video_Median_Velocity', 
        data=sex_df,
        inner='quartile',
        palette="muted"
    )
    
    # Set title and labels
    if source_sans:
        ax.set_title('Velocity Distribution by Sex', fontproperties=source_sans, fontsize=8)
        ax.set_xlabel('Sex', fontproperties=source_sans)
        ax.set_ylabel('Median Velocity (mm/s)', fontproperties=source_sans)
    else:
        ax.set_title('Velocity Distribution by Sex', fontsize=8)
        ax.set_xlabel('Sex')
        ax.set_ylabel('Median Velocity (mm/s)')
    
    # Add counts to the x-tick labels
    ax.set_xticklabels(xtick_labels)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_distribution_by_sex.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def analyze_pressure_specific(df: pd.DataFrame) -> None:
    """
    Analyzes sex differences separately for each applied pressure level.
    
    Args:
        df: DataFrame containing Sex, Pressure, and Video_Median_Velocity columns
    """
    # Create output directory for plots
    output_dir = os.path.join(cap_flow_path, 'results', 'SexAnalysis', 'PressureSpecific')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nAnalyzing pressure-specific sex differences...")
    
    # Check if we have necessary columns
    if not all(col in df.columns for col in ['Sex', 'Pressure', 'Video_Median_Velocity']):
        print("Error: Missing required columns for pressure-specific analysis")
        return
    
    # Get unique pressure levels
    pressures = sorted(df['Pressure'].unique())
    print(f"Found {len(pressures)} pressure levels: {pressures}")
    
    for pressure in pressures:
        print(f"\nAnalyzing sex differences for pressure: {pressure}")
        
        # Filter data for this pressure
        pressure_df = df[df['Pressure'] == pressure].dropna(subset=['Video_Median_Velocity', 'Sex']).copy()
        
        if len(pressure_df) < 10:
            print(f"Skipping pressure {pressure} - insufficient data ({len(pressure_df)} samples)")
            continue
        
        # Print data size
        print(f"Samples for pressure {pressure}: {len(pressure_df)}")
        
        # Group data by sex
        male_group = pressure_df[pressure_df['Sex'] == 'M']['Video_Median_Velocity']
        female_group = pressure_df[pressure_df['Sex'] == 'F']['Video_Median_Velocity']
        
        if len(male_group) < 3 or len(female_group) < 3:
            print(f"Skipping pressure {pressure} - insufficient group sizes")
            continue
        
        # Print statistics for each group
        print(f"Male participants (n={len(male_group)}):")
        print(f"  Median velocity: {male_group.median():.2f} mm/s")
        print(f"  Interquartile range: {male_group.quantile(0.25):.2f} to {male_group.quantile(0.75):.2f} mm/s")
        
        print(f"Female participants (n={len(female_group)}):")
        print(f"  Median velocity: {female_group.median():.2f} mm/s")
        print(f"  Interquartile range: {female_group.quantile(0.25):.2f} to {female_group.quantile(0.75):.2f} mm/s")
        
        # Run statistical test
        stat, p_value = stats.mannwhitneyu(male_group, female_group)
        print(f"Mann-Whitney U test p-value: {p_value:.6f}")
        
        # Create CDF plot for this pressure
        plt.close()
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'pdf.fonttype': 42, 'ps.fonttype': 42,
            'font.size': 7, 'axes.labelsize': 7,
            'xtick.labelsize': 6, 'ytick.labelsize': 6,
            'legend.fontsize': 5, 'lines.linewidth': 0.5
        })
        
        fig, ax = plt.subplots(figsize=(2.4, 2.0))
        
        # Calculate empirical CDFs
        x_male = np.sort(male_group)
        y_male = np.arange(1, len(x_male) + 1) / len(x_male)
        
        x_female = np.sort(female_group)
        y_female = np.arange(1, len(x_female) + 1) / len(x_female)
        
        # Plot CDFs
        ax.plot(x_male, y_male, 'b-', linewidth=1, label=f'Male (n={len(male_group)})')
        ax.plot(x_female, y_female, 'r-', linewidth=1, label=f'Female (n={len(female_group)})')
        
        # Add p-value annotation
        p_text = f"p = {p_value:.3f}" if p_value >= 0.001 else "p < 0.001"
        ax.text(0.05, 0.95, p_text, transform=ax.transAxes, fontsize=6, va='top')
        
        # Set labels and title
        if source_sans:
            ax.set_xlabel('Video Median Velocity (mm/s)', fontproperties=source_sans)
            ax.set_ylabel('Cumulative Probability', fontproperties=source_sans)
            ax.set_title(f'Velocity CDF by Sex (Pressure: {pressure} PSI)', fontproperties=source_sans)
            ax.legend(prop=source_sans)
        else:
            ax.set_xlabel('Video Median Velocity (mm/s)')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title(f'Velocity CDF by Sex (Pressure: {pressure} PSI)')
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'velocity_by_sex_pressure_{pressure}.png'), 
                   dpi=600, bbox_inches='tight')
        plt.close()

def main():
    """Main function for sex-based analysis of capillary velocity data."""
    print("\nRunning sex-based analysis for capillary velocity data...")
    
    # Load data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Filter for control group
    controls_df = df[df['SET'] == 'set01']
    
    # Ensure we have sex data
    if 'Sex' not in controls_df.columns or controls_df['Sex'].isna().all():
        print("Error: Sex data is missing or all null. Cannot create sex-based plots.")
        return 1
    
    # Create output directory for plots
    output_dir = os.path.join(cap_flow_path, 'results', 'SexAnalysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Print basic statistics
    sex_counts = controls_df['Sex'].value_counts()
    print(f"\nData overview:")
    print(f"Total participants: {len(controls_df['Participant'].unique())}")
    print(f"Sex distribution: {sex_counts.to_dict()}")
    
    # Create CDF plot comparing sexes
    plt.close()
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    create_sex_cdf_plot(controls_df, cap_flow_path, ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_cdf_by_sex.png'), 
               dpi=600, bbox_inches='tight')
    plt.close()
    
    # Create boxplots comparing sexes
    plot_velocity_boxplots(controls_df)
    
    # Analyze pressure-specific differences
    analyze_pressure_specific(controls_df)
    
    print("\nSex-based velocity analysis complete.")
    return 0

if __name__ == "__main__":
    main() 