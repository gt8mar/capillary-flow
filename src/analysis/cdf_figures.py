"""
Filename: src/analysis/cdf_figures.py
--------------------------------------------------

This script plots the CDF figures for the ATVB paper.

By: Marcus Forst
"""
import os
import pandas as pd
from src.config import PATHS
from src.tools.plotting_utils import plot_cdf_old

def main():
    print('Generating CDF figures...')
    
    # Load dataset (same path as confidence-interval analyses)
    data_fp = os.path.join(PATHS['cap_flow'], 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_fp)
    
    # Basic cleaning (mirrors previous scripts)
    df = df.dropna(subset=['Age', 'Corrected Velocity'])
    df = df.drop_duplicates(subset=['Participant', 'Video'])
    
    # choose set01
    df = df[df['SET'] == 'set01']

    # Plot CDF figures
    plot_cdf_old(df['Corrected Velocity'], subsets=[df[df['Age'] > 50]['Corrected Velocity'], df[df['Age'] <= 50]['Corrected Velocity']], labels=['Entire Dataset', '>50', '≤50'], title = 'CDF Comparison of velocities by Age')

if __name__ == '__main__':
    main()
