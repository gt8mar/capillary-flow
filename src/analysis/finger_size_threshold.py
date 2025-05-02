"""
Filename: src/analysis/finger_size_threshold.py

File for analyzing finger size thresholds in capillary velocity data.

This script:
1. Determines the optimal finger size threshold for differentiating velocity distributions
2. Creates boxplots and CDF plots of velocities by finger size groups
3. Analyzes pressure-specific finger size thresholds
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.font_manager import FontProperties
from typing import Tuple, List, Dict, Optional

# Import paths from config instead of defining computer paths locally
from src.config import PATHS, load_source_sans

cap_flow_path = PATHS['cap_flow']
source_sans = load_source_sans()

def create_finger_size_cdf_plot(df: pd.DataFrame, threshold: float, 
                               output_path: str, ax=None) -> Optional[float]:
    """
    Creates a CDF plot for velocities split by finger size groups based on the given threshold.
    
    Args:
        df: DataFrame containing FingerSizeBottom and Video_Median_Velocity columns
        threshold: Finger size threshold to split groups (smaller vs larger)
        output_path: Path for finding fonts and saving results
        ax: Matplotlib axis object to plot on (optional)
    
    Returns:
        KS statistic measuring the difference between distributions
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.0))
    
    # Create finger size groups
    df['Size_Group'] = df['FingerSizeBottom'].apply(lambda x: f'<{threshold} mm' if x < threshold else f'≥{threshold} mm')
    
    # Group data
    small_group = df[df['FingerSizeBottom'] < threshold]['Video_Median_Velocity']
    large_group = df[df['FingerSizeBottom'] >= threshold]['Video_Median_Velocity']
    
    # Check if we have enough data in both groups
    if len(small_group) < 3 or len(large_group) < 3:
        print(f"Warning: Not enough data for finger size threshold {threshold}")
        return None
    
    # Calculate empirical CDFs
    x_small = np.sort(small_group)
    y_small = np.arange(1, len(x_small) + 1) / len(x_small)
    
    x_large = np.sort(large_group)
    y_large = np.arange(1, len(x_large) + 1) / len(x_large)
    
    # Plot CDFs
    ax.plot(x_small, y_small, 'b-', linewidth=1, label=f'<{threshold} mm (n={len(small_group)})')
    ax.plot(x_large, y_large, 'r-', linewidth=1, label=f'≥{threshold} mm (n={len(large_group)})')
    
    # Run KS test
    ks_stat, p_value = stats.ks_2samp(small_group, large_group)

    print(f"KS statistic for threshold {threshold}: {ks_stat}")
    print(f"p-value for threshold {threshold}: {p_value}")
    print(f"The median velocity for the small group is {small_group.median()} and the large group is {large_group.median()}")
    print(f"The interquartile range for the small group is {small_group.quantile(0.25)} to {small_group.quantile(0.75)} and the large group is {large_group.quantile(0.25)} to {large_group.quantile(0.75)}")
    
    # Add KS test result to plot
    p_text = f"p = {p_value:.3f}" if p_value >= 0.001 else "p < 0.001"
    ax.text(0.05, 0.95, f"KS stat: {ks_stat:.3f}\n{p_text}", 
           transform=ax.transAxes, fontsize=6, va='top')
    
    # Try to use Source Sans font if available
    if source_sans:
        ax.set_xlabel('Video Median Velocity (mm/s)', fontproperties=source_sans)
        ax.set_ylabel('Cumulative Probability', fontproperties=source_sans)
        ax.set_title(f'Velocity CDF by Finger Size (Threshold: {threshold} mm)', 
                    fontproperties=source_sans)
        ax.legend(prop=source_sans)
    else:
        # Fall back to default font
        ax.set_xlabel('Video Median Velocity (mm/s)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Velocity CDF by Finger Size (Threshold: {threshold} mm)')
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    return ks_stat


def threshold_analysis(df: pd.DataFrame) -> float:
    """
    Analyzes different finger size thresholds to find the one that best differentiates
    velocity distributions.
    
    Args:
        df: DataFrame containing FingerSizeBottom and Video_Median_Velocity columns
    
    Returns:
        The best finger size threshold for differentiating velocity distributions
    """
    # Create output directory for plots
    output_dir = os.path.join(cap_flow_path, 'results', 'FingerSizeThreshold')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nAnalyzing finger size thresholds for velocity distributions...")
    
    # Ensure we have finger size data
    if 'FingerSizeBottom' not in df.columns or df['FingerSizeBottom'].isna().all():
        print("Error: Finger size data is missing or all null. Cannot create finger size-based plots.")
        return None
    
    # Create a copy of the dataframe for finger size analysis
    size_df = df.dropna(subset=['Video_Median_Velocity', 'FingerSizeBottom']).copy()
    
    # Print finger size statistics
    print(f"Finger size range in data: {size_df['FingerSizeBottom'].min():.2f} to {size_df['FingerSizeBottom'].max():.2f} mm")
    print(f"Mean finger size: {size_df['FingerSizeBottom'].mean():.2f} mm")
    print(f"Median finger size: {size_df['FingerSizeBottom'].median():.2f} mm")
    
    # Test different finger size thresholds
    size_min = float(np.floor(size_df['FingerSizeBottom'].min()))
    size_max = float(np.ceil(size_df['FingerSizeBottom'].max()))
    size_range = size_max - size_min
    
    # Create a range of thresholds to test
    # If we have a wide size range, test with larger step sizes
    if size_range > 20:
        step = 2.0
    elif size_range > 10:
        step = 1.0
    else:
        step = 0.5
    
    thresholds = np.arange(size_min + step, size_max - step, step)
    thresholds = [float(round(t, 1)) for t in thresholds]  # Round to 1 decimal place
    
    # Ensure we have at least some thresholds
    if len(thresholds) == 0:
        # If size range is very narrow, just use the median
        thresholds = [float(round(size_df['FingerSizeBottom'].median(), 1))]
    
    print(f"Testing finger size thresholds: {thresholds}")
    
    # Create individual plots for each threshold
    ks_results = {}
    for threshold in thresholds:
        plt.close()
        # Set up style and font
        sns.set_style("whitegrid")
        
        plt.rcParams.update({
            'pdf.fonttype': 42, 'ps.fonttype': 42,
            'font.size': 7, 'axes.labelsize': 7,
            'xtick.labelsize': 6, 'ytick.labelsize': 6,
            'legend.fontsize': 5, 'lines.linewidth': 0.5
        })
        
        fig, ax = plt.subplots(figsize=(2.4, 2.0))
        ks_stat = create_finger_size_cdf_plot(size_df, threshold, cap_flow_path, ax)
        if ks_stat is not None:
            ks_results[threshold] = ks_stat
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'finger_size_velocity_cdf_threshold_{threshold}.png'), 
                   dpi=600, bbox_inches='tight')
        plt.close()
    
    # Find the threshold with the maximum KS statistic (most different distributions)
    if ks_results:
        best_threshold = max(ks_results, key=ks_results.get)
        print(f"\nThreshold with most distinct velocity distributions: {best_threshold:.1f} mm")
        print(f"KS statistic: {ks_results[best_threshold]:.3f}")
        
        # Create a plot showing KS statistic vs threshold
        if len(ks_results) > 1:
            plt.close()
            # Set up style and font
            sns.set_style("whitegrid")
            
            plt.rcParams.update({
                'pdf.fonttype': 42, 'ps.fonttype': 42,
                'font.size': 7, 'axes.labelsize': 7,
                'xtick.labelsize': 6, 'ytick.labelsize': 6,
                'legend.fontsize': 5, 'lines.linewidth': 0.5
            })
            
            fig, ax = plt.subplots(figsize=(2.4, 2.0))
            thresholds_list = list(ks_results.keys())
            ks_stats = list(ks_results.values())
            
            ax.plot(thresholds_list, ks_stats, 'o-', linewidth=0.5)
            ax.axvline(x=best_threshold, color='red', linestyle='--', 
                      label=f'Best threshold: {best_threshold:.1f} mm')
            
            # Try to use Source Sans font if available
            if source_sans:
                ax.set_xlabel('Finger Size Threshold (mm)', fontproperties=source_sans)
                ax.set_ylabel('KS Statistic', fontproperties=source_sans)
                ax.set_title('Capillary Velocity Similarity by Finger Size', fontproperties=source_sans)
                ax.legend(prop=source_sans)
            else:
                ax.set_xlabel('Finger Size Threshold (mm)')
                ax.set_ylabel('KS Statistic')
                ax.set_title('Capillary Velocity Similarity by Finger Size')
                ax.legend()
                
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'velocity_ks_statistic_vs_threshold.png'), 
                       dpi=600, bbox_inches='tight')
            plt.close()
            
        return best_threshold
    
    return None


def plot_velocity_boxplots(df: pd.DataFrame, best_threshold: Optional[float] = None) -> None:
    """
    Creates boxplots of video median velocities grouped by different finger size categories.
    
    Args:
        df: DataFrame containing FingerSizeBottom and Video_Median_Velocity columns
        best_threshold: Optional optimal finger size threshold from threshold_analysis
    """
    # Create output directory for plots
    output_dir = os.path.join(cap_flow_path, 'results', 'FingerSizeThreshold')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nCreating velocity boxplots by finger size groups...")
    
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
    
    # Create size groups
    size_df = df.dropna(subset=['Video_Median_Velocity', 'FingerSizeBottom']).copy()
    
    # 1. Create boxplot by equal-width finger size bins
    plt.figure(figsize=(2.4, 2.0))
    
    # Calculate bin edges for approximately 5 bins
    size_min = size_df['FingerSizeBottom'].min()
    size_max = size_df['FingerSizeBottom'].max()
    bin_width = (size_max - size_min) / 5
    
    # Create custom bins
    bin_edges = [size_min + i * bin_width for i in range(6)]
    bin_edges = [round(edge, 1) for edge in bin_edges]
    
    # Make sure bin edges are unique
    bin_edges = sorted(list(set(bin_edges)))
    
    # Create size groups
    labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
    size_df['Size_Group'] = pd.cut(
        size_df['FingerSizeBottom'], 
        bins=bin_edges, 
        labels=labels,
        include_lowest=True
    )
    
    # Create the boxplot
    ax = sns.boxplot(
        x='Size_Group',
        y='Video_Median_Velocity',
        data=size_df,
        color='#1f77b4',
        width=0.6,
        fliersize=3
    )
    
    # Set title and labels
    if source_sans:
        ax.set_title('Video Median Velocity by Finger Size Group', fontproperties=source_sans, fontsize=8)
        ax.set_xlabel('Finger Size (mm)', fontproperties=source_sans)
        ax.set_ylabel('Median Velocity (mm/s)', fontproperties=source_sans)
    else:
        ax.set_title('Video Median Velocity by Finger Size Group', fontsize=8)
        ax.set_xlabel('Finger Size (mm)')
        ax.set_ylabel('Median Velocity (mm/s)')
    
    # Get counts for each size group
    counts = size_df['Size_Group'].value_counts().sort_index()
    
    # Add counts to the x-tick labels
    xtick_labels = [f"{label.get_text()}\n(n={counts[label.get_text()]})" 
                  if label.get_text() in counts.index else label.get_text()
                  for label in ax.get_xticklabels()]
    ax.set_xticklabels(xtick_labels)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_by_finger_size_groups.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create boxplot by optimal threshold if provided
    if best_threshold is not None:
        plt.figure(figsize=(2.4, 2.0))
        
        # Create binary size groups based on threshold
        size_df['Threshold_Group'] = size_df['FingerSizeBottom'].apply(
            lambda x: f'<{best_threshold:.1f} mm' if x < best_threshold else f'≥{best_threshold:.1f} mm'
        )
        
        # Create the boxplot
        ax = sns.boxplot(
            x='Threshold_Group',
            y='Video_Median_Velocity',
            data=size_df,
            color='#2ca02c',
            width=0.6,
            fliersize=3
        )
        
        # Set title and labels
        if source_sans:
            ax.set_title(f'Video Median Velocity by Finger Size Threshold ({best_threshold:.1f} mm)', 
                        fontproperties=source_sans, fontsize=8)
            ax.set_xlabel('Finger Size Group', fontproperties=source_sans)
            ax.set_ylabel('Median Velocity (mm/s)', fontproperties=source_sans)
        else:
            ax.set_title(f'Video Median Velocity by Finger Size Threshold ({best_threshold:.1f} mm)', fontsize=8)
            ax.set_xlabel('Finger Size Group')
            ax.set_ylabel('Median Velocity (mm/s)')
        
        # Get counts for each threshold group
        counts = size_df['Threshold_Group'].value_counts().sort_index()
        
        # Add counts to the x-tick labels
        xtick_labels = [f"{label.get_text()}\n(n={counts[label.get_text()]})" 
                      if label.get_text() in counts.index else label.get_text()
                      for label in ax.get_xticklabels()]
        ax.set_xticklabels(xtick_labels)
        
        # Add statistical annotation
        groups = size_df.groupby('Threshold_Group')['Video_Median_Velocity'].apply(list).to_dict()
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
        plt.savefig(os.path.join(output_dir, f'velocity_by_threshold_{best_threshold:.1f}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def analyze_pressure_specific_thresholds(df: pd.DataFrame) -> Dict[int, float]:
    """
    Analyzes finger size thresholds separately for each pressure level.
    
    Args:
        df: DataFrame containing FingerSizeBottom, Pressure, and Video_Median_Velocity columns
    
    Returns:
        Dictionary mapping pressure levels to their best finger size thresholds
    """
    # Create output directory for plots
    output_dir = os.path.join(cap_flow_path, 'results', 'FingerSizeThreshold', 'PressureSpecific')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nAnalyzing pressure-specific finger size thresholds...")
    
    # Check if we have necessary columns
    if not all(col in df.columns for col in ['FingerSizeBottom', 'Pressure', 'Video_Median_Velocity']):
        print("Error: Missing required columns for pressure-specific analysis")
        return {}
    
    # Get unique pressure levels
    pressures = sorted(df['Pressure'].unique())
    print(f"Found {len(pressures)} pressure levels: {pressures}")
    
    best_thresholds = {}
    ks_all_pressures = {}
    
    for pressure in pressures:
        print(f"\nAnalyzing threshold for pressure: {pressure}")
        
        # Filter data for this pressure
        pressure_df = df[df['Pressure'] == pressure].dropna(subset=['Video_Median_Velocity', 'FingerSizeBottom']).copy()
        
        if len(pressure_df) < 10:
            print(f"Skipping pressure {pressure} - insufficient data ({len(pressure_df)} samples)")
            continue
            
        # Print data size
        print(f"Samples for pressure {pressure}: {len(pressure_df)}")
        
        # Determine finger size range for thresholds
        size_min = float(np.floor(pressure_df['FingerSizeBottom'].min()))
        size_max = float(np.ceil(pressure_df['FingerSizeBottom'].max()))
        size_range = size_max - size_min
        
        # Create a range of thresholds to test
        if size_range > 20:
            step = 2.0
        elif size_range > 10:
            step = 1.0
        else:
            step = 0.5
        
        thresholds = np.arange(size_min + step, size_max - step, step)
        thresholds = [float(round(t, 1)) for t in thresholds]  # Round to 1 decimal place
        
        # Ensure we have at least some thresholds
        if len(thresholds) == 0:
            thresholds = [float(round(pressure_df['FingerSizeBottom'].median(), 1))]
        
        print(f"Testing thresholds for pressure {pressure}: {thresholds}")
        
        # Test each threshold
        ks_results = {}
        for threshold in thresholds:
            # Create size groups
            pressure_df['Size_Group'] = pressure_df['FingerSizeBottom'].apply(
                lambda x: f'<{threshold}' if x < threshold else f'≥{threshold}'
            )
            
            # Get velocity data for each group
            small_group = pressure_df[pressure_df['FingerSizeBottom'] < threshold]['Video_Median_Velocity']
            large_group = pressure_df[pressure_df['FingerSizeBottom'] >= threshold]['Video_Median_Velocity']
            
            # Skip if not enough data in both groups
            if len(small_group) < 3 or len(large_group) < 3:
                print(f"Skipping threshold {threshold} - insufficient group sizes")
                continue
            
            # Calculate KS statistic
            ks_stat, p_value = stats.ks_2samp(small_group, large_group)
            ks_results[threshold] = ks_stat
            ks_all_pressures[(pressure, threshold)] = ks_stat
            
            # Create CDF plot for this pressure and threshold
            plt.close()
            sns.set_style("whitegrid")
            plt.rcParams.update({
                'pdf.fonttype': 42, 'ps.fonttype': 42,
                'font.size': 7, 'axes.labelsize': 7,
                'xtick.labelsize': 6, 'ytick.labelsize': 6,
                'legend.fontsize': 5, 'lines.linewidth': 0.5
            })
            
            fig, ax = plt.subplots(figsize=(2.4, 2.0))
            create_finger_size_cdf_plot(pressure_df, threshold, cap_flow_path, ax)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'pressure_{pressure}_threshold_{threshold}.png'), 
                       dpi=600, bbox_inches='tight')
            plt.close()
        
        # Find best threshold for this pressure
        if ks_results:
            best_threshold = max(ks_results, key=ks_results.get)
            best_thresholds[pressure] = best_threshold
            print(f"Best threshold for pressure {pressure}: {best_threshold:.1f} mm (KS: {ks_results[best_threshold]:.3f})")
    
    # Create summary plot of best thresholds by pressure
    if best_thresholds:
        plt.close()
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'pdf.fonttype': 42, 'ps.fonttype': 42,
            'font.size': 7, 'axes.labelsize': 7,
            'xtick.labelsize': 6, 'ytick.labelsize': 6,
            'legend.fontsize': 5, 'lines.linewidth': 0.5
        })
        
        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        
        pressures = list(best_thresholds.keys())
        thresholds = [best_thresholds[p] for p in pressures]
        
        ax.plot(pressures, thresholds, 'o-', linewidth=1)
        
        # Try to use Source Sans font if available
        if source_sans:
            ax.set_xlabel('Pressure (PSI)', fontproperties=source_sans)
            ax.set_ylabel('Best Finger Size Threshold (mm)', fontproperties=source_sans)
            ax.set_title('Optimal Finger Size Threshold by Pressure', fontproperties=source_sans)
        else:
            ax.set_xlabel('Pressure (PSI)')
            ax.set_ylabel('Best Finger Size Threshold (mm)')
            ax.set_title('Optimal Finger Size Threshold by Pressure')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'best_threshold_by_pressure.png'), 
                   dpi=600, bbox_inches='tight')
        plt.close()
    
    return best_thresholds


def load_and_prepare_data():
    """
    Loads data from CSV files and prepares the merged dataset for finger size analysis.
    
    Returns:
        DataFrame containing both finger metrics and velocity data
    """
    print("\nLoading and preparing data for finger size threshold analysis...")
    
    # Load velocity data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Standardize finger column names
    df['Finger'] = df['Finger'].str[1:]
    df['Finger'] = df['Finger'].str.lower()
    df['Finger'] = df['Finger'].str.capitalize()
    df['Finger'] = df['Finger'].replace('Mid', 'Middle')
    df['Finger'] = df['Finger'].replace('Index', 'Pointer')
    print(f"Fingers in dataset: {df['Finger'].unique()}")

    # Load finger stats data
    finger_stats_df = pd.read_csv(os.path.join(cap_flow_path, 'finger_stats.csv'))
    
    # Merge with velocity data
    merged_df = pd.merge(df, finger_stats_df, on='Participant', how='left')
    merged_df = merged_df.dropna(subset=['Pointer bottom'])
    
    # Map from 'Finger' string to the column name holding the bottom size
    bottom_col_map = {f: f"{f} bottom" for f in ['Pointer', 'Middle', 'Ring', 'Pinky']}

    # Define a helper function for robust lookup within the row
    def get_finger_size(row, col_map):
        finger = row['Finger']
        col_name = col_map.get(finger)
        # Check if the finger name is valid and the corresponding column exists in the row
        if col_name and col_name in row.index:
            return row[col_name]
        # Return NaN if finger name is invalid or column doesn't exist
        return np.nan

    # Apply the function to get the bottom sizes directly from merged_df columns
    merged_df['FingerSizeBottom'] = merged_df.apply(lambda row: get_finger_size(row, bottom_col_map), axis=1)
    
    # Save merged data for reference
    merged_df.to_csv(os.path.join(cap_flow_path, 'finger_size_threshold_analysis_df.csv'), index=False)
    
    # Check how many participants have both finger and velocity data
    participants_with_data = merged_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity'])
    unique_participants = participants_with_data['Participant'].nunique()
    
    print(f"Found {unique_participants} participants with both finger size and velocity data")
    print(f"Total observations: {len(participants_with_data)}")
    
    return merged_df


def main():
    """Main function for finger size threshold analysis."""
    print("\nRunning finger size threshold analysis for capillary velocity data...")
    
    # Load and prepare data
    merged_df = load_and_prepare_data()
    
    # Optionally filter for control data if needed
    # controls_df = merged_df[merged_df['SET'] == 'set01']
    controls_df = merged_df
    
    # Set up plotting style
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
    
    # Run threshold analysis to find optimal finger size cutoff
    best_threshold = threshold_analysis(controls_df)
    
    # Plot velocity by finger size groups
    plot_velocity_boxplots(controls_df, best_threshold)
    
    # Run pressure-specific threshold analysis
    pressure_thresholds = analyze_pressure_specific_thresholds(controls_df)
    
    print("\nFinger size threshold analysis complete.")
    
    return 0


if __name__ == "__main__":
    main() 