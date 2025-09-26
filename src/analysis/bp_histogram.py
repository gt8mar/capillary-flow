"""
Filename: src/analysis/bp_histogram.py

File for creating histograms of blood pressure data from capillary velocity studies.

This script:
1. Loads the same data as bp_threshold.py
2. Creates histograms of systolic and diastolic blood pressure
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

def create_bp_histogram(df: pd.DataFrame, bp_type: str = 'SYS_BP', 
                       output_dir: Optional[str] = None) -> None:
    """
    Creates a histogram of blood pressure data.
    
    Args:
        df: DataFrame containing blood pressure columns
        bp_type: Type of blood pressure to plot ('SYS_BP' or 'DIA_BP')
        output_dir: Directory to save the plot (optional)
    """
    bp_label = "Systolic" if bp_type == 'SYS_BP' else "Diastolic"
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'BPHistograms')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating {bp_label.lower()} blood pressure histogram...")
    
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
    
    # Filter out missing BP data
    bp_df = df.dropna(subset=[bp_type]).copy()

    # keep only one row per participant and bp pair
    bp_df = bp_df.drop_duplicates(subset=['Participant', bp_type]) 
    
    if len(bp_df) == 0:
        print(f"Error: No valid {bp_label.lower()} BP data found.")
        return
    
    # Create figure
    plt.figure(figsize=(3.5, 2.5))
    
    # Create histogram
    bp_values = bp_df[bp_type]
    
    # Use automatic binning but ensure reasonable number of bins
    n_bins = min(30, max(10, int(np.sqrt(len(bp_values)))))
    
    plt.hist(bp_values, bins=n_bins, alpha=0.7, color='#1f77b4', 
             edgecolor='black', linewidth=0.5)
    
    # Add statistics to the plot
    mean_bp = bp_values.mean()
    median_bp = bp_values.median()
    std_bp = bp_values.std()
    
    # # Add vertical lines for mean and median
    # plt.axvline(mean_bp, color='red', linestyle='--', linewidth=1, 
    #             label=f'Mean: {mean_bp:.1f} mmHg')
    # plt.axvline(median_bp, color='orange', linestyle='--', linewidth=1,
    #             label=f'Median: {median_bp:.1f} mmHg')
    
    # # Add text box with statistics
    # stats_text = f'n = {len(bp_values)}\nMean = {mean_bp:.1f} ± {std_bp:.1f} mmHg\nMedian = {median_bp:.1f} mmHg'
    # plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
    #          fontsize=6, verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set title and labels with font handling
    if source_sans:
        plt.title(f'{bp_label} Blood Pressure Distribution', 
                 fontproperties=source_sans, fontsize=8)
        plt.xlabel(f'{bp_label} Blood Pressure (mmHg)', fontproperties=source_sans)
        plt.ylabel('Frequency', fontproperties=source_sans)
        plt.legend(prop=source_sans)
    else:
        plt.title(f'{bp_label} Blood Pressure Distribution', fontsize=8)
        plt.xlabel(f'{bp_label} Blood Pressure (mmHg)')
        plt.ylabel('Frequency')
        plt.legend()
    
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    filename = f'{bp_label.lower()}_bp_histogram.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram saved to: {os.path.join(output_dir, filename)}")
    print(f"{bp_label} BP statistics:")
    print(f"  Range: {bp_values.min():.1f} - {bp_values.max():.1f} mmHg")
    print(f"  Mean: {mean_bp:.1f} ± {std_bp:.1f} mmHg")
    print(f"  Median: {median_bp:.1f} mmHg")
    print(f"  Sample size: {len(bp_values)}")

def create_combined_bp_histogram(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """
    Creates a combined histogram showing both systolic and diastolic BP.
    
    Args:
        df: DataFrame containing both SYS_BP and DIA_BP columns
        output_dir: Directory to save the plot (optional)
    """
    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'BPHistograms')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating combined blood pressure histogram...")
    
    # Standard plot configuration
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    # Filter out missing BP data
    sys_bp_df = df.dropna(subset=['SYS_BP'])
    dia_bp_df = df.dropna(subset=['DIA_BP'])

    # keep only one row per participant and bp pair
    sys_bp_df = sys_bp_df.drop_duplicates(subset=['Participant', 'SYS_BP'])
    dia_bp_df = dia_bp_df.drop_duplicates(subset=['Participant', 'DIA_BP'])     
    
    if len(sys_bp_df) == 0 and len(dia_bp_df) == 0:
        print("Error: No valid blood pressure data found.")
        return
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
    
    # Systolic BP histogram
    if len(sys_bp_df) > 0:
        sys_values = sys_bp_df['SYS_BP']
        n_bins_sys = min(25, max(10, int(np.sqrt(len(sys_values)))))
        
        ax1.hist(sys_values, bins=n_bins_sys, alpha=0.7, color='#1f77b4',
                edgecolor='black', linewidth=0.5)
        
        # Add statistics
        mean_sys = sys_values.mean()
        median_sys = sys_values.median()
        
        ax1.axvline(mean_sys, color='red', linestyle='--', linewidth=1)
        ax1.axvline(median_sys, color='orange', linestyle='--', linewidth=1)
        
        
        if source_sans:
            ax1.set_title('Systolic Blood Pressure', fontproperties=source_sans, fontsize=8)
            ax1.set_xlabel('Systolic BP (mmHg)', fontproperties=source_sans)
            ax1.set_ylabel('Frequency', fontproperties=source_sans)
        else:
            ax1.set_title('Systolic Blood Pressure', fontsize=8)
            ax1.set_xlabel('Systolic BP (mmHg)')
            ax1.set_ylabel('Frequency')
        
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'n = {len(sys_values)}\nMean = {mean_sys:.1f}\nMedian = {median_sys:.1f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=6, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Diastolic BP histogram
    if len(dia_bp_df) > 0:
        dia_values = dia_bp_df['DIA_BP']
        n_bins_dia = min(25, max(10, int(np.sqrt(len(dia_values)))))
        
        ax2.hist(dia_values, bins=n_bins_dia, alpha=0.7, color='#ff7f0e',
                edgecolor='black', linewidth=0.5)
        
        # Add statistics
        mean_dia = dia_values.mean()
        median_dia = dia_values.median()
        
        ax2.axvline(mean_dia, color='red', linestyle='--', linewidth=1)
        ax2.axvline(median_dia, color='orange', linestyle='--', linewidth=1)
        
        
        if source_sans:
            ax2.set_title('Diastolic Blood Pressure', fontproperties=source_sans, fontsize=8)
            ax2.set_xlabel('Diastolic BP (mmHg)', fontproperties=source_sans)
            ax2.set_ylabel('Frequency', fontproperties=source_sans)
        else:
            ax2.set_title('Diastolic Blood Pressure', fontsize=8)
            ax2.set_xlabel('Diastolic BP (mmHg)')
            ax2.set_ylabel('Frequency')
        
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'n = {len(dia_values)}\nMean = {mean_dia:.1f}\nMedian = {median_dia:.1f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                fontsize=6, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the figure
    filename = 'combined_bp_histogram.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined histogram saved to: {os.path.join(output_dir, filename)}")

def main(bp_type: str = 'both', show_combined: bool = True):
    """
    Main function for creating blood pressure histograms.
    
    Args:
        bp_type: Type of blood pressure to plot ('SYS_BP', 'DIA_BP', or 'both')
        show_combined: Whether to create a combined histogram plot
    """
    print(f"\nCreating blood pressure histogram(s)...")
    
    # Load data - same approach as bp_threshold.py
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Ensure blood pressure columns exist
    if 'BP' in df.columns:
        if 'SYS_BP' not in df.columns:
            print(f"Extracting SYS_BP from BP column...")
            df['SYS_BP'] = df['BP'].str.split('/').str[0].astype(float)
        if 'DIA_BP' not in df.columns:
            print(f"Extracting DIA_BP from BP column...")
            df['DIA_BP'] = df['BP'].str.split('/').str[1].astype(float)
    
    # Filter for control group - same as bp_threshold.py
    controls_df = df[df['SET'] == 'set01']
    
    print(f"Using control group data (SET == 'set01'): {len(controls_df)} samples")
    
    # Create output directory
    output_dir = os.path.join(cap_flow_path, 'results', 'BPHistograms')
    
    # Create histograms based on request
    if bp_type == 'both' or bp_type == 'SYS_BP':
        create_bp_histogram(controls_df, 'SYS_BP', output_dir)
    
    if bp_type == 'both' or bp_type == 'DIA_BP':
        create_bp_histogram(controls_df, 'DIA_BP', output_dir)
    
    # Create combined histogram if requested
    if show_combined and bp_type == 'both':
        create_combined_bp_histogram(controls_df, output_dir)
    
    print(f"\nBlood pressure histogram analysis complete.")
    print(f"Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create histograms of blood pressure data.')
    parser.add_argument('--bp-type', choices=['SYS_BP', 'DIA_BP', 'both'], default='both',
                       help='Type of blood pressure to plot (default: both)')
    parser.add_argument('--no-combined', action='store_true',
                       help='Skip creating combined histogram plot')
    
    args = parser.parse_args()
    
    main(bp_type=args.bp_type, show_combined=not args.no_combined)
