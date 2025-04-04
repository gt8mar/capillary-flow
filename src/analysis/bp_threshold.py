"""
Filename: src/analysis/bp_threshold.py

File for analyzing blood pressure thresholds in capillary velocity data.

This script:
1. Determines the optimal blood pressure threshold for differentiating velocity distributions
2. Creates boxplots of velocities by blood pressure groups
3. Supports both systolic and diastolic blood pressure analysis
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
from src.tools.plotting_utils import plot_CI_multiple_bands

cap_flow_path = PATHS['cap_flow']
source_sans = load_source_sans()

def create_bp_cdf_plot(df: pd.DataFrame, threshold: int, 
                      output_path: str, bp_type: str = 'SYS_BP',
                      ax=None) -> Optional[float]:
    """
    Creates a CDF plot for velocities split by blood pressure groups based on the given threshold.
    
    Args:
        df: DataFrame containing blood pressure and Video_Median_Velocity columns
        threshold: Blood pressure threshold to split groups (lower vs higher)
        output_path: Path for finding fonts and saving results
        bp_type: Type of blood pressure to use ('SYS_BP' or 'DIA_BP')
        ax: Matplotlib axis object to plot on (optional)
    
    Returns:
        KS statistic measuring the difference between distributions
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.0))
    
    # Create BP groups
    bp_label = "Systolic" if bp_type == 'SYS_BP' else "Diastolic"
    df['BP_Group'] = df[bp_type].apply(lambda x: f'<{threshold} mmHg' if x < threshold else f'≥{threshold} mmHg')
    
    # Group data
    low_group = df[df[bp_type] < threshold]['Video_Median_Velocity']
    high_group = df[df[bp_type] >= threshold]['Video_Median_Velocity']
    
    # Check if we have enough data in both groups
    if len(low_group) < 3 or len(high_group) < 3:
        print(f"Warning: Not enough data for {bp_label} BP threshold {threshold}")
        return None
    
    # Calculate empirical CDFs
    x_low = np.sort(low_group)
    y_low = np.arange(1, len(x_low) + 1) / len(x_low)
    
    x_high = np.sort(high_group)
    y_high = np.arange(1, len(x_high) + 1) / len(x_high)
    
    # Plot CDFs
    ax.plot(x_low, y_low, 'b-', linewidth=1, label=f'<{threshold} mmHg (n={len(low_group)})')
    ax.plot(x_high, y_high, 'r-', linewidth=1, label=f'≥{threshold} mmHg (n={len(high_group)})')
    
    # Run KS test
    ks_stat, p_value = stats.ks_2samp(low_group, high_group)
    
    # Add KS test result to plot
    p_text = f"p = {p_value:.3f}" if p_value >= 0.001 else "p < 0.001"
    ax.text(0.05, 0.95, f"KS stat: {ks_stat:.3f}\n{p_text}", 
           transform=ax.transAxes, fontsize=6, va='top')
    
    print(f"KS statistic for threshold {threshold}: {ks_stat}")
    print(f"p-value for threshold {threshold}: {p_value}")
    print(f"The median velocity for the low group is {low_group.median()} and the high group is {high_group.median()}")
    print(f"The standard deviation for the low group is {low_group.std()} and the high group is {high_group.std()}")
            
    
    # Try to use Source Sans font if available
    try:
        source_sans = FontProperties(fname=os.path.join(PATHS['downloads'], 
                                                       'Source_Sans_3', 'static', 
                                                       'SourceSans3-Regular.ttf'))
        ax.set_xlabel('Video Median Velocity (mm/s)', fontproperties=source_sans)
        ax.set_ylabel('Cumulative Probability', fontproperties=source_sans)
        ax.set_title(f'Velocity CDF by {bp_label} BP (Threshold: {threshold})', 
                    fontproperties=source_sans)
        ax.legend(prop=source_sans)
    except:
        # Fall back to default font
        ax.set_xlabel('Video Median Velocity (mm/s)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Velocity CDF by {bp_label} BP (Threshold: {threshold})')
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    return ks_stat 

def threshold_analysis(df: pd.DataFrame, bp_type: str = 'SYS_BP') -> int:
    """
    Analyzes different blood pressure thresholds to find the one that best differentiates
    velocity distributions.
    
    Args:
        df: DataFrame containing blood pressure and Video_Median_Velocity columns
        bp_type: Type of blood pressure to use ('SYS_BP' or 'DIA_BP')
    
    Returns:
        The best blood pressure threshold for differentiating velocity distributions
    """
    bp_label = "Systolic" if bp_type == 'SYS_BP' else "Diastolic"
    
    # Create output directory for plots
    output_dir = os.path.join(cap_flow_path, 'results', f'{bp_label}BPThreshold')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAnalyzing {bp_label.lower()} blood pressure thresholds for velocity distributions...")
    
    # Ensure we have BP data
    if bp_type not in df.columns or df[bp_type].isna().all():
        print(f"Error: {bp_label} BP data is missing or all null. Cannot create BP-based plots.")
        return None
    
    # Create a copy of the dataframe for BP analysis
    bp_df = df.dropna(subset=['Video_Median_Velocity', bp_type]).copy()
    
    # Print BP statistics
    print(f"{bp_label} BP range in data: {bp_df[bp_type].min()} to {bp_df[bp_type].max()} mmHg")
    print(f"Mean {bp_label.lower()} BP: {bp_df[bp_type].mean():.2f} mmHg")
    print(f"Median {bp_label.lower()} BP: {bp_df[bp_type].median():.2f} mmHg")
    
    # Test different BP thresholds
    bp_min = int(np.floor(bp_df[bp_type].min()))
    bp_max = int(np.ceil(bp_df[bp_type].max()))
    
    # Create a range of thresholds to test
    # Standard clinical thresholds for hypertension
    if bp_type == 'SYS_BP':
        # Test around standard systolic thresholds: 120 (normal/elevated), 130-139 (Stage 1), 140+ (Stage 2)
        base_thresholds = [120, 130, 140]
        # Add additional thresholds within data range, every 5 mmHg
        additional_thresholds = list(range(max(110, bp_min), min(160, bp_max), 5))
        thresholds = sorted(list(set(base_thresholds + additional_thresholds)))
    else:  # DIA_BP
        # Test around standard diastolic thresholds: 80 (normal/elevated), 80-89 (Stage 1), 90+ (Stage 2)
        base_thresholds = [80, 90]
        # Add additional thresholds within data range, every 5 mmHg
        additional_thresholds = list(range(max(70, bp_min), min(100, bp_max), 5))
        thresholds = sorted(list(set(base_thresholds + additional_thresholds)))
    
    # Ensure thresholds are within data range
    thresholds = [t for t in thresholds if bp_min < t < bp_max]
    
    # Ensure we have at least some thresholds
    if len(thresholds) == 0:
        # If BP range is very narrow, just use the median
        thresholds = [int(bp_df[bp_type].median())]
    
    print(f"Testing {bp_label.lower()} BP thresholds: {thresholds}")
    
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
        ks_stat = create_bp_cdf_plot(bp_df, threshold, cap_flow_path, bp_type, ax)
        if ks_stat is not None:
            ks_results[threshold] = ks_stat
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{bp_label.lower()}_velocity_cdf_threshold_{threshold}.png'), 
                   dpi=600, bbox_inches='tight')
        plt.close()
    
    # Find the threshold with the maximum KS statistic (most different distributions)
    if ks_results:
        best_threshold = max(ks_results, key=ks_results.get)
        print(f"\nThreshold with most distinct velocity distributions: {best_threshold} mmHg")
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
                      label=f'Best threshold: {best_threshold} mmHg')
            
            # Try to use Source Sans font if available
            try:
                source_sans = FontProperties(fname=os.path.join(PATHS['downloads'], 
                                                               'Source_Sans_3', 'static', 
                                                               'SourceSans3-Regular.ttf'))
                ax.set_xlabel(f'{bp_label} BP Threshold (mmHg)', fontproperties=source_sans)
                ax.set_ylabel('KS Statistic', fontproperties=source_sans)
                ax.set_title(f'Capillary Velocity Similarity by {bp_label} BP', fontproperties=source_sans)
                ax.legend(prop=source_sans)
            except:
                ax.set_xlabel(f'{bp_label} BP Threshold (mmHg)')
                ax.set_ylabel('KS Statistic')
                ax.set_title(f'Capillary Velocity Similarity by {bp_label} BP')
                ax.legend()
                
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{bp_label.lower()}_velocity_ks_statistic_vs_threshold.png'), 
                       dpi=600, bbox_inches='tight')
            plt.close()
            
        return best_threshold
    
    return None 

def plot_velocity_boxplots(df: pd.DataFrame, best_threshold: Optional[int] = None, 
                          bp_type: str = 'SYS_BP') -> None:
    """
    Creates boxplots of video median velocities grouped by different blood pressure categories.
    
    Args:
        df: DataFrame containing blood pressure and Video_Median_Velocity columns
        best_threshold: Optional optimal BP threshold from threshold_analysis
        bp_type: Type of blood pressure to use ('SYS_BP' or 'DIA_BP')
    """
    bp_label = "Systolic" if bp_type == 'SYS_BP' else "Diastolic"
    
    # Create output directory for plots
    output_dir = os.path.join(cap_flow_path, 'results', f'{bp_label}BPThreshold')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating velocity boxplots by {bp_label.lower()} blood pressure groups...")
    
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
    
    # 1. Create boxplot by predefined BP groups
    plt.figure(figsize=(2.4, 2.0))
    
    # Create BP groups based on clinical categories
    bp_df = df.dropna(subset=['Video_Median_Velocity', bp_type]).copy()
    
    if bp_type == 'SYS_BP':
        # Systolic BP categories
        bp_df['BP_Group'] = pd.cut(
            bp_df[bp_type], 
            bins=[0, 120, 130, 140, 180, 300], 
            labels=['Normal (<120)', 'Elevated (120-129)', 'Stage 1 (130-139)', 'Stage 2 (140-179)', 'Crisis (≥180)'],
            include_lowest=True
        )
    else:
        # Diastolic BP categories
        bp_df['BP_Group'] = pd.cut(
            bp_df[bp_type], 
            bins=[0, 80, 90, 120, 300], 
            labels=['Normal (<80)', 'Stage 1 (80-89)', 'Stage 2 (90-119)', 'Crisis (≥120)'],
            include_lowest=True
        )
    
    # Create the boxplot
    ax = sns.boxplot(
        x='BP_Group',
        y='Video_Median_Velocity',
        data=bp_df,
        color='#1f77b4',
        width=0.6,
        fliersize=3
    )
    
    # Set title and labels
    if source_sans:
        ax.set_title(f'Video Median Velocity by {bp_label} BP Group', fontproperties=source_sans, fontsize=8)
        ax.set_xlabel(f'{bp_label} BP Group', fontproperties=source_sans)
        ax.set_ylabel('Median Velocity (mm/s)', fontproperties=source_sans)
    else:
        ax.set_title(f'Video Median Velocity by {bp_label} BP Group', fontsize=8)
        ax.set_xlabel(f'{bp_label} BP Group')
        ax.set_ylabel('Median Velocity (mm/s)')
    
    # Get counts for each BP group
    counts = bp_df['BP_Group'].value_counts().sort_index()
    
    # Add counts to the x-tick labels
    xtick_labels = [f"{label.get_text()}\n(n={counts[label.get_text()]})" 
                  if label.get_text() in counts.index else label.get_text()
                  for label in ax.get_xticklabels()]
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'velocity_by_{bp_label.lower()}_bp_groups.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create boxplot by optimal threshold if provided
    if best_threshold is not None:
        plt.figure(figsize=(2.4, 2.0))
        
        # Create binary BP groups based on threshold
        bp_df['Threshold_Group'] = bp_df[bp_type].apply(
            lambda x: f'<{best_threshold} mmHg' if x < best_threshold else f'≥{best_threshold} mmHg'
        )
        
        # Create the boxplot
        ax = sns.boxplot(
            x='Threshold_Group',
            y='Video_Median_Velocity',
            data=bp_df,
            color='#2ca02c',
            width=0.6,
            fliersize=3
        )
        
        # Set title and labels
        if source_sans:
            ax.set_title(f'Video Median Velocity by {bp_label} BP Threshold ({best_threshold} mmHg)', 
                        fontproperties=source_sans, fontsize=8)
            ax.set_xlabel(f'{bp_label} BP Group', fontproperties=source_sans)
            ax.set_ylabel('Median Velocity (mm/s)', fontproperties=source_sans)
        else:
            ax.set_title(f'Video Median Velocity by {bp_label} BP Threshold ({best_threshold} mmHg)', fontsize=8)
            ax.set_xlabel(f'{bp_label} BP Group')
            ax.set_ylabel('Median Velocity (mm/s)')
        
        # Get counts for each threshold group
        counts = bp_df['Threshold_Group'].value_counts().sort_index()
        
        # Add counts to the x-tick labels
        xtick_labels = [f"{label.get_text()}\n(n={counts[label.get_text()]})" 
                      if label.get_text() in counts.index else label.get_text()
                      for label in ax.get_xticklabels()]
        ax.set_xticklabels(xtick_labels)
        
        # Add statistical annotation
        groups = bp_df.groupby('Threshold_Group')['Video_Median_Velocity'].apply(list).to_dict()
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
        plt.savefig(os.path.join(output_dir, f'velocity_by_{bp_label.lower()}_threshold_{best_threshold}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close() 

def analyze_pressure_specific_thresholds(df: pd.DataFrame, bp_type: str = 'SYS_BP') -> Dict[int, int]:
    """
    Analyzes blood pressure thresholds separately for each applied pressure level.
    
    Args:
        df: DataFrame containing blood pressure, Pressure, and Video_Median_Velocity columns
        bp_type: Type of blood pressure to use ('SYS_BP' or 'DIA_BP')
    
    Returns:
        Dictionary mapping pressure levels to their best BP thresholds
    """
    bp_label = "Systolic" if bp_type == 'SYS_BP' else "Diastolic"
    
    # Create output directory for plots
    output_dir = os.path.join(cap_flow_path, 'results', f'{bp_label}BPThreshold', 'PressureSpecific')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAnalyzing pressure-specific {bp_label.lower()} BP thresholds...")
    
    # Check if we have necessary columns
    if not all(col in df.columns for col in [bp_type, 'Pressure', 'Video_Median_Velocity']):
        print(f"Error: Missing required columns for pressure-specific analysis")
        return {}
    
    # Get unique pressure levels
    pressures = sorted(df['Pressure'].unique())
    print(f"Found {len(pressures)} pressure levels: {pressures}")
    
    best_thresholds = {}
    ks_all_pressures = {}
    
    for pressure in pressures:
        print(f"\nAnalyzing threshold for pressure: {pressure}")
        
        # Filter data for this pressure
        pressure_df = df[df['Pressure'] == pressure].dropna(subset=['Video_Median_Velocity', bp_type]).copy()
        
        if len(pressure_df) < 10:
            print(f"Skipping pressure {pressure} - insufficient data ({len(pressure_df)} samples)")
            continue
            
        # Print data size
        print(f"Samples for pressure {pressure}: {len(pressure_df)}")
        
        # Determine BP range for thresholds
        bp_min = int(np.floor(pressure_df[bp_type].min()))
        bp_max = int(np.ceil(pressure_df[bp_type].max()))
        
        # Create a range of thresholds to test
        if bp_type == 'SYS_BP':
            # Standard clinical thresholds for systolic BP
            base_thresholds = [120, 130, 140]
            additional_thresholds = list(range(max(110, bp_min), min(160, bp_max), 5))
            thresholds = sorted(list(set(base_thresholds + additional_thresholds)))
        else:
            # Standard clinical thresholds for diastolic BP
            base_thresholds = [80, 90]
            additional_thresholds = list(range(max(70, bp_min), min(100, bp_max), 5))
            thresholds = sorted(list(set(base_thresholds + additional_thresholds)))
        
        # Ensure thresholds are within data range
        thresholds = [t for t in thresholds if bp_min < t < bp_max]
        
        # Ensure we have at least some thresholds
        if len(thresholds) == 0:
            thresholds = [int(pressure_df[bp_type].median())]
        
        print(f"Testing {bp_label.lower()} BP thresholds for pressure {pressure}: {thresholds}")
        
        # Test each threshold
        ks_results = {}
        for threshold in thresholds:
            # Create BP groups
            pressure_df['BP_Group'] = pressure_df[bp_type].apply(
                lambda x: f'<{threshold}' if x < threshold else f'≥{threshold}'
            )
            
            # Get velocity data for each group
            low_group = pressure_df[pressure_df[bp_type] < threshold]['Video_Median_Velocity']
            high_group = pressure_df[pressure_df[bp_type] >= threshold]['Video_Median_Velocity']
            
            # Skip if not enough data in both groups
            if len(low_group) < 3 or len(high_group) < 3:
                print(f"Skipping threshold {threshold} - insufficient group sizes")
                continue
            
            # Calculate KS statistic
            ks_stat, p_value = stats.ks_2samp(low_group, high_group)
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
            create_bp_cdf_plot(pressure_df, threshold, cap_flow_path, bp_type, ax)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'pressure_{pressure}_{bp_label.lower()}_threshold_{threshold}.png'), 
                       dpi=600, bbox_inches='tight')
            plt.close()
        
        # Find best threshold for this pressure
        if ks_results:
            best_threshold = max(ks_results, key=ks_results.get)
            best_thresholds[pressure] = best_threshold
            print(f"Best {bp_label.lower()} BP threshold for pressure {pressure}: {best_threshold} mmHg (KS: {ks_results[best_threshold]:.3f})")
    
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
            ax.set_xlabel('Applied Pressure (PSI)', fontproperties=source_sans)
            ax.set_ylabel(f'Best {bp_label} BP Threshold (mmHg)', fontproperties=source_sans)
            ax.set_title(f'Optimal {bp_label} BP Threshold by Applied Pressure', fontproperties=source_sans)
        else:
            ax.set_xlabel('Applied Pressure (PSI)')
            ax.set_ylabel(f'Best {bp_label} BP Threshold (mmHg)')
            ax.set_title(f'Optimal {bp_label} BP Threshold by Applied Pressure')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'best_{bp_label.lower()}_threshold_by_pressure.png'), 
                   dpi=600, bbox_inches='tight')
        plt.close()
    
    return best_thresholds 

def main(use_diastolic: bool = False):
    """
    Main function for blood pressure threshold analysis.
    
    Args:
        use_diastolic: Whether to use diastolic (instead of systolic) blood pressure
    """
    bp_type = 'DIA_BP' if use_diastolic else 'SYS_BP'
    bp_label = "Diastolic" if use_diastolic else "Systolic"
    
    print(f"\nRunning {bp_label.lower()} blood pressure threshold analysis for capillary velocity data...")
    
    # Load data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Ensure blood pressure columns exist
    if 'BP' in df.columns and bp_type not in df.columns:
        print(f"Extracting {bp_type} from BP column...")
        if bp_type == 'SYS_BP':
            df[bp_type] = df['BP'].str.split('/').str[0].astype(float)
        else:  # DIA_BP
            df[bp_type] = df['BP'].str.split('/').str[1].astype(float)
    
    # Filter for control group
    controls_df = df[df['SET'] == 'set01']
    
    # Run threshold analysis to find optimal BP cutoff
    best_threshold = threshold_analysis(controls_df, bp_type)
    
    # Plot velocity by BP groups
    plot_velocity_boxplots(controls_df, best_threshold, bp_type)
    
    # Run pressure-specific threshold analysis
    pressure_thresholds = analyze_pressure_specific_thresholds(controls_df, bp_type)
    
    print(f"\n{bp_label} blood pressure threshold analysis complete.")
    
    # Optional: Plot CI bands
    # plot_CI_multiple_bands(controls_df, thresholds=[best_threshold], variable=bp_type, method='bootstrap', 
    #                       n_iterations=1000, ci_percentile=95, write=True, dimensionless=False, 
    #                       video_median=False, log_scale=False, velocity_variable='Corrected Velocity')

    return 0


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze blood pressure thresholds in capillary velocity data.')
    parser.add_argument('--diastolic', '-d', action='store_true', 
                       help='Use diastolic instead of systolic blood pressure')
    
    args = parser.parse_args()
    
    main(use_diastolic=args.diastolic) 