"""
Filename: temporal_resolution.py
---------------------------------

Analyzes the temporal resolution of our capillary microscope.

By: Marcus Forst
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.font_manager import FontProperties

from src.config import PATHS, load_source_sans

source_sans = load_source_sans()

PIX_UM = 2.44

def main():
    # Load data
    df_path = os.path.join(PATHS['cap_flow'], 'results', 'diameter_analysis_df.csv')
    summary_df_path = os.path.join(PATHS['cap_flow'], 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(df_path)
    summary_df = pd.read_csv(summary_df_path)
    # print all columns with "centerline" in the name
    centerline_cols = df.columns[df.columns.str.contains('Centerline')]

    # add FPS values from summary_df to df using participant and video columns
    df = df.merge(summary_df[['Participant','Video', 'FPS']], on=['Participant','Video'], how='left')
    # rename FPS_x to FPS
    df['FPS'] = df['FPS_x']
    df = df.drop(columns=['FPS_x', 'FPS_y'])

    # # Scatterplot of 'Corrected Velocity' vs 'Centerline_Length'
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x='Centerline_Length', y='Corrected_Velocity', data=df)
    # plt.show()

    # Calculate average frames in view for each capillary but cap the max at 1000
    df['Frames_in_view'] = df['Centerline_Length'] / df['Corrected_Velocity'] * df['FPS'] / PIX_UM
    df['Frames_in_view'] = df['Frames_in_view'].clip(upper=1000)
    print(df['Frames_in_view'].describe())

    # Calculate number of Frames_in_view under 5:
    df_under_5 = df[df['Frames_in_view'] < 5]
    under_5 = len(df_under_5)
    print(f'the number of Frames_in_view below 5 is: {under_5}')

    # Calculate number of Frames_in_view under 3:
    df_under_3 = df[df['Frames_in_view'] < 3]
    under_3 = len(df_under_3)
    print(f'the number of Frames_in_view below 3 is: {under_3}')

    # # Scatterplot of 'Frames_in_view' vs 'Corrected_Velocity'
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x='Frames_in_view', y='Corrected_Velocity', data=df)
    # plt.show()

    # Histogram of 'Frames_in_view'
    # Apply standard plot configuration
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
    
    # Create figure with standard dimensions
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    
    # Create histogram with improved aesthetics
    sns.histplot(
        data=df, 
        x='Frames_in_view', 
        bins=40, 
        binrange=(0, 200),
        color='#1f77b4',
        alpha=0.8,
        ax=ax
    )
    
    # Apply font to labels if available
    if source_sans:
        ax.set_xlabel('Frames in View', fontproperties=source_sans)
        ax.set_ylabel('Count', fontproperties=source_sans)
        ax.set_title('Distribution of Frames in View', fontproperties=source_sans)
    else:
        ax.set_xlabel('Frames in View')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Frames in View')
    
    # Add vertical lines for reference
    ax.axvline(x=3, color='#d62728', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(x=5, color='#ff7f0e', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Add text annotations
    ax.text(3.5, ax.get_ylim()[1]*0.9, f'{under_3} caps < 3 frames', 
            fontsize=5, color='#d62728')
    ax.text(5.5, ax.get_ylim()[1]*0.8, f'{under_5} caps < 5 frames', 
            fontsize=5, color='#ff7f0e')
    
    # Save the figure to results folder
    save_path = os.path.join(PATHS['cap_flow'], 'results', 'frames_in_view_histogram.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Histogram saved to: {save_path}")
    
    # Display the plot
    # plt.show()
    plt.close()

    # Print the number of total cap measurements in the histogram
    print(f'the number of total cap measurements in the histogram is: {len(df)}')
    return 0

if __name__ == '__main__':
    main()