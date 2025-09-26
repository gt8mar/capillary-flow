"""
Filename: src/analysis/participant27_velocity_histogram.py
----------------------------------------------------------

This script creates a publishable quality histogram of velocities 
for participant 27, following the coding standards for plot styling.

By: Assistant
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from src.config import PATHS

def get_source_sans_font():
    """Safely load the SourceSans font with fallback to default font."""
    try:
        font_path = os.path.join(PATHS['downloads'], 'Source_Sans_3', 'static', 'SourceSans3-Regular.ttf')
        if os.path.exists(font_path):
            return FontProperties(fname=font_path)
        print("Warning: SourceSans3-Regular.ttf not found, using default font")
        return None
    except Exception as e:
        print(f"Warning: Error loading font: {e}")
        return None

def setup_plot_style():
    """Setup the standard plot configuration according to coding standards."""
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

def plot_participant27_cdf(df):
    """
    Create a publishable quality CDF plot of velocities for participant 27.
    
    Args:
        df: DataFrame containing velocity data
        
    Returns:
        0 if successful, 1 if error occurred
    """
    # Filter for participant 27
    part27_data = df[df['Participant'] == 'part27']
    
    if part27_data.empty:
        print("Error: No data found for participant 27")
        return 1
    
    print(f"Found {len(part27_data)} velocity measurements for participant 27")
    
    # Get velocities
    velocities = part27_data['Corrected Velocity'].dropna()
    
    if velocities.empty:
        print("Error: No valid velocity data found for participant 27")
        return 1
    
    print(f"Creating CDF for {len(velocities)} velocity measurements")
    
    # Setup plotting style
    setup_plot_style()
    source_sans = get_source_sans_font()
    
    # Create figure with standard dimensions
    plt.figure(figsize=(2.4, 2.0))
    
    # Sort velocities for CDF
    sorted_velocities = np.sort(velocities)
    # Calculate cumulative probabilities
    y_values = np.arange(1, len(sorted_velocities) + 1) / len(sorted_velocities)
    
    # Plot CDF
    plt.plot(sorted_velocities, y_values, color='#1f77b4', linewidth=1.0)
    
    # Set labels and title with font handling
    if source_sans:
        plt.xlabel('Corrected Velocity (μm/s)', fontproperties=source_sans)
        plt.ylabel('Cumulative Probability', fontproperties=source_sans)
        plt.title('Participant 27 - Velocity CDF', fontproperties=source_sans)
    else:
        plt.xlabel('Corrected Velocity (μm/s)')
        plt.ylabel('Cumulative Probability')
        plt.title('Participant 27 - Velocity CDF')
    
    # Set y-axis limits
    plt.ylim(0, 1)
    
    # Add sample size annotation
    plt.text(0.95, 0.05, f'n = {len(velocities)}', 
             transform=plt.gca().transAxes, ha='right', va='bottom',
             fontsize=5, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'participant_histograms')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'participant27_velocity_cdf.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"CDF plot saved to: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = os.path.join(output_dir, 'participant27_velocity_cdf.pdf')
    plt.savefig(pdf_path, dpi=600, bbox_inches='tight')
    print(f"CDF PDF saved to: {pdf_path}")
    
    plt.show()
    return 0

def plot_participant27_histogram(df):
    """
    Create a publishable quality histogram of velocities for participant 27.
    
    Args:
        df: DataFrame containing velocity data
        
    Returns:
        0 if successful, 1 if error occurred
    """
    # Filter for participant 27
    part27_data = df[df['Participant'] == 'part27']
    
    if part27_data.empty:
        print("Error: No data found for participant 27")
        return 1
    
    print(f"Found {len(part27_data)} velocity measurements for participant 27")
    
    # Get velocities and basic statistics
    velocities = part27_data['Corrected Velocity'].dropna()
    
    if velocities.empty:
        print("Error: No valid velocity data found for participant 27")
        return 1
    
    mean_vel = velocities.mean()
    median_vel = velocities.median()
    std_vel = velocities.std()
    
    print(f"Velocity statistics for participant 27:")
    print(f"  Mean: {mean_vel:.2f} μm/s")
    print(f"  Median: {median_vel:.2f} μm/s")
    print(f"  Std Dev: {std_vel:.2f} μm/s")
    print(f"  Range: {velocities.min():.2f} - {velocities.max():.2f} μm/s")
    
    # Setup plotting style
    setup_plot_style()
    source_sans = get_source_sans_font()
    
    # Create figure with standard dimensions
    plt.figure(figsize=(2.4, 2.0))
    
    # Create histogram with standard blue color
    plt.hist(velocities, bins=15, density=True, alpha=0.7, 
             color='#1f77b4', edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for mean and median
    plt.axvline(x=mean_vel, color='#d62728', linestyle='-', linewidth=1.0,
                label=f'Mean: {mean_vel:.1f} μm/s')
    plt.axvline(x=median_vel, color='#2ca02c', linestyle='--', linewidth=1.0,
                label=f'Median: {median_vel:.1f} μm/s')
    
    # Set labels and title with font handling
    if source_sans:
        plt.xlabel('Corrected Velocity (μm/s)', fontproperties=source_sans)
        plt.ylabel('Density', fontproperties=source_sans)
        plt.title('Participant 27 - Velocity Distribution', fontproperties=source_sans)
    else:
        plt.xlabel('Corrected Velocity (μm/s)')
        plt.ylabel('Density')
        plt.title('Participant 27 - Velocity Distribution')
    
    # Add legend
    plt.legend(prop={'size': 5})
    
    # Add sample size annotation
    plt.text(0.95, 0.95, f'n = {len(velocities)}', 
             transform=plt.gca().transAxes, ha='right', va='top',
             fontsize=5, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'participant_distributions')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'participant27_velocity_histogram.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = os.path.join(output_dir, 'participant27_velocity_histogram.pdf')
    plt.savefig(pdf_path, dpi=600, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    return 0

def main():
    """Main function to load data and create both histogram and CDF plots."""
    print('Generating velocity histogram and CDF for participant 27...')
    
    # Load dataset (same path as other analyses)
    data_fp = os.path.join(PATHS['cap_flow'], 'summary_df_nhp_video_stats.csv')
    
    if not os.path.exists(data_fp):
        print(f"Error: Data file not found at {data_fp}")
        return 1
    
    df = pd.read_csv(data_fp)
    
    # Basic cleaning (mirrors other scripts)
    df = df.dropna(subset=['Age', 'Corrected Velocity'])
    df = df.drop_duplicates(subset=['Participant', 'Video'])
    
    print(f"Data loaded: {len(df)} rows, {len(df['Participant'].unique())} unique participants")
    
    # Check if participant 27 exists
    if 'part27' not in df['Participant'].unique():
        print("Available participants:", sorted(df['Participant'].unique()))
        print("Error: Participant 27 not found in dataset")
        return 1
    
    # Create the histogram
    hist_result = plot_participant27_histogram(df)
    
    if hist_result == 0:
        print("Histogram generation completed successfully!")
    else:
        print("Error occurred during histogram generation.")
    
    # Create the CDF
    cdf_result = plot_participant27_cdf(df)
    
    if cdf_result == 0:
        print("CDF generation completed successfully!")
    else:
        print("Error occurred during CDF generation.")
    
    # Return 0 if both succeeded, 1 if either failed
    return 0 if (hist_result == 0 and cdf_result == 0) else 1

if __name__ == '__main__':
    main()
