"""
Filename: src/analysis/median_group_velocity.py
--------------------------------------------------

This script loads data the same way as age_score_roc.py and 
calculates median velocity and IQR for different age groups. 
Default threshold is 59 and younger vs 60 and older.

By: Marcus Forst
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict

# Import paths from config
from src.config import PATHS


def load_and_preprocess_data() -> pd.DataFrame:
    """
    Loads and preprocesses the data for velocity analysis.
    Uses the same loading pattern as age_score_roc.py.
    
    Returns:
        DataFrame containing the preprocessed data
    """
    print("\nLoading and preprocessing data...")
    
    # Load data (same path as other analyses)
    data_filepath = os.path.join(PATHS['cap_flow'], 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Basic cleaning - same as age_score_roc.py
    df = df.dropna(subset=['SET','Age', 'Corrected Velocity'])
    df = df.drop_duplicates(subset=['Participant', 'Video'])

    # choose set01
    df = df[df['SET'] == 'set01']
    
    print(f"Loaded {len(df)} records")
    return df


def split_groups_by_age_threshold(df: pd.DataFrame, age_threshold: int = 59) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into two age groups based on the specified threshold.
    
    Args:
        df: DataFrame containing the data with 'Age' and 'Corrected Velocity' columns
        age_threshold: Age threshold for splitting groups (default: 59)
        
    Returns:
        Tuple of (younger_group, older_group) DataFrames
    """
    younger_group = df[df['Age'] <= age_threshold].copy()
    older_group = df[df['Age'] > age_threshold].copy()
    
    print(f"\nSplit data by age threshold {age_threshold}:")
    print(f"  Younger group (<={age_threshold}): {len(younger_group)} records")
    print(f"  Older group (>{age_threshold}): {len(older_group)} records")
    
    return younger_group, older_group


def calculate_median_and_iqr(df: pd.DataFrame, group_name: str) -> Dict[str, float]:
    """
    Calculate median velocity and IQR for a given group.
    
    Args:
        df: DataFrame containing velocity data
        group_name: Name of the group for display purposes
        
    Returns:
        Dictionary containing median, Q1, Q3, and IQR values
    """
    velocities = df['Corrected Velocity'].dropna()
    
    if len(velocities) == 0:
        print(f"Warning: No valid velocity data for {group_name}")
        return {
            'median': np.nan,
            'q1': np.nan,
            'q3': np.nan,
            'iqr': np.nan,
            'count': 0
        }
    
    median_vel = np.median(velocities)
    q1 = np.percentile(velocities, 25)
    q3 = np.percentile(velocities, 75)
    iqr = q3 - q1
    
    return {
        'median': median_vel,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'count': len(velocities)
    }


def analyze_velocity_by_age_groups(df: pd.DataFrame, age_threshold: int = 59) -> Dict[str, Dict[str, float]]:
    """
    Analyze velocity statistics for different age groups.
    
    Args:
        df: DataFrame containing the data
        age_threshold: Age threshold for splitting groups (default: 59)
        
    Returns:
        Dictionary containing statistics for each age group
    """
    print(f"\n{'='*60}")
    print(f"VELOCITY ANALYSIS BY AGE GROUPS (Threshold: {age_threshold})")
    print(f"{'='*60}")
    
    # Split into age groups
    younger_group, older_group = split_groups_by_age_threshold(df, age_threshold)
    
    # Calculate statistics for each group
    younger_stats = calculate_median_and_iqr(younger_group, f"<={age_threshold} years")
    older_stats = calculate_median_and_iqr(older_group, f">{age_threshold} years")
    
    # Print results
    print(f"\nVELOCITY STATISTICS:")
    print(f"{'-'*40}")
    
    print(f"\nYounger Group (<={age_threshold} years):")
    print(f"  Count: {younger_stats['count']}")
    print(f"  Median Velocity: {younger_stats['median']:.2f} um/s")
    print(f"  Q1 (25th percentile): {younger_stats['q1']:.2f} um/s")
    print(f"  Q3 (75th percentile): {younger_stats['q3']:.2f} um/s")
    print(f"  IQR: {younger_stats['iqr']:.2f} um/s")
    
    print(f"\nOlder Group (>{age_threshold} years):")
    print(f"  Count: {older_stats['count']}")
    print(f"  Median Velocity: {older_stats['median']:.2f} um/s")
    print(f"  Q1 (25th percentile): {older_stats['q1']:.2f} um/s")
    print(f"  Q3 (75th percentile): {older_stats['q3']:.2f} um/s")
    print(f"  IQR: {older_stats['iqr']:.2f} um/s")
    
    # Calculate difference
    if not (np.isnan(younger_stats['median']) or np.isnan(older_stats['median'])):
        median_diff = younger_stats['median'] - older_stats['median']
        iqr_diff = younger_stats['iqr'] - older_stats['iqr']
        
        print(f"\nGROUP COMPARISONS:")
        print(f"{'-'*40}")
        print(f"  Median Velocity Difference (Younger - Older): {median_diff:.2f} um/s")
        print(f"  IQR Difference (Younger - Older): {iqr_diff:.2f} um/s")
        
        if median_diff > 0:
            print(f"  -> Younger group has higher median velocity")
        elif median_diff < 0:
            print(f"  -> Older group has higher median velocity")
        else:
            print(f"  -> Groups have equal median velocity")
    
    return {
        f'younger_group_le_{age_threshold}': younger_stats,
        f'older_group_gt_{age_threshold}': older_stats
    }


def main():
    """Main function to run median group velocity analysis."""
    print("Starting median group velocity analysis...")
    
    # Load data using the same method as age_score_roc.py
    df = load_and_preprocess_data()
    
    # Analyze velocity by age groups (default threshold: 59)
    results = analyze_velocity_by_age_groups(df, age_threshold=59)
    
    # Additional analysis with different thresholds if desired
    print(f"\n{'='*60}")
    print("ADDITIONAL THRESHOLD ANALYSIS")
    print(f"{'='*60}")
    
    # Test with other common age thresholds
    other_thresholds = [50, 65]
    for threshold in other_thresholds:
        print(f"\n--- Age Threshold: {threshold} ---")
        analyze_velocity_by_age_groups(df, age_threshold=threshold)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
