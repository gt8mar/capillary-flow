"""
Filename: src/analysis/stiffness_coeff.py

File for calculating stiffness coefficients and stopping/restart pressures
from capillary flow velocity data.
By: Marcus Forst
"""

import os
import pandas as pd
import numpy as np
from scipy.integrate import trapz
from typing import Dict, Tuple, Optional

# Import paths from config
from src.config import PATHS

cap_flow_path = PATHS['cap_flow']

# Threshold for zero flow (in um/s)
ZERO_FLOW_THRESHOLD = 5.0  # um/s

# Pressure range for stiffness coefficient calculation
STIFFNESS_PRESSURE_MIN = 0.4  # psi
STIFFNESS_PRESSURE_MAX = 1.2  # psi


def get_up_down_curves(participant_df: pd.DataFrame, 
                      velocity_column: str = 'Video_Median_Velocity') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract up and down velocity curves for a participant.
    
    Args:
        participant_df: DataFrame containing data for a single participant
        velocity_column: Name of the velocity column to use (default: 'Video_Median_Velocity')
    
    Returns:
        Tuple containing:
            - up_pressures: Array of pressures for up curve (sorted)
            - up_velocities: Array of mean velocities for up curve
            - down_pressures: Array of pressures for down curve (sorted)
            - down_velocities: Array of mean velocities for down curve
    """
    # Get up curve data
    up_data = participant_df[participant_df['UpDown'] == 'U'].copy()
    up_grouped = up_data.groupby('Pressure')[velocity_column].mean()
    up_pressures = up_grouped.index.values
    up_velocities = up_grouped.values
    
    # Sort by pressure (ascending for up curve)
    sort_idx = np.argsort(up_pressures)
    up_pressures = up_pressures[sort_idx]
    up_velocities = up_velocities[sort_idx]
    
    # Get down curve data
    down_data = participant_df[participant_df['UpDown'] == 'D'].copy()
    down_grouped = down_data.groupby('Pressure')[velocity_column].mean()
    down_pressures = down_grouped.index.values
    down_velocities = down_grouped.values
    
    # Sort by pressure (ascending for down curve)
    sort_idx = np.argsort(down_pressures)
    down_pressures = down_pressures[sort_idx]
    down_velocities = down_velocities[sort_idx]
    
    return up_pressures, up_velocities, down_pressures, down_velocities


def calculate_stiffness_coefficient_up(up_pressures: np.ndarray, 
                                       up_velocities: np.ndarray,
                                       pressure_min: float = STIFFNESS_PRESSURE_MIN,
                                       pressure_max: float = STIFFNESS_PRESSURE_MAX) -> float:
    """Calculate stiffness coefficient as area under the up curve between pressure_min and pressure_max.
    
    Args:
        up_pressures: Array of pressures for up curve
        up_velocities: Array of velocities for up curve
        pressure_min: Minimum pressure for integration (default: 0.4 psi)
        pressure_max: Maximum pressure for integration (default: 1.2 psi)
    
    Returns:
        Area under the up curve between pressure_min and pressure_max (or np.nan if insufficient data)
    """
    # Filter to pressure range
    mask = (up_pressures >= pressure_min) & (up_pressures <= pressure_max)
    
    if np.sum(mask) < 2:
        return np.nan
    
    filtered_pressures = up_pressures[mask]
    filtered_velocities = up_velocities[mask]
    
    # Calculate area using trapezoidal rule
    area = trapz(filtered_velocities, filtered_pressures)
    
    return area


def calculate_stiffness_coefficient_averaged(up_pressures: np.ndarray,
                                            up_velocities: np.ndarray,
                                            down_pressures: np.ndarray,
                                            down_velocities: np.ndarray,
                                            pressure_min: float = STIFFNESS_PRESSURE_MIN,
                                            pressure_max: float = STIFFNESS_PRESSURE_MAX) -> float:
    """Calculate stiffness coefficient as area under the averaged up and down curves.
    
    Args:
        up_pressures: Array of pressures for up curve
        up_velocities: Array of velocities for up curve
        down_pressures: Array of pressures for down curve
        down_velocities: Array of velocities for down curve
        pressure_min: Minimum pressure for integration (default: 0.4 psi)
        pressure_max: Maximum pressure for integration (default: 1.2 psi)
    
    Returns:
        Area under the averaged curve between pressure_min and pressure_max (or np.nan if insufficient data)
    """
    # Get all unique pressures in the range from both curves
    all_pressures = np.unique(np.concatenate([up_pressures, down_pressures]))
    pressure_mask = (all_pressures >= pressure_min) & (all_pressures <= pressure_max)
    filtered_pressures = all_pressures[pressure_mask]
    
    if len(filtered_pressures) < 2:
        return np.nan
    
    # Interpolate velocities at common pressure points
    up_interp = np.interp(filtered_pressures, up_pressures, up_velocities, 
                          left=np.nan, right=np.nan)
    down_interp = np.interp(filtered_pressures, down_pressures, down_velocities,
                           left=np.nan, right=np.nan)
    
    # Average the velocities
    averaged_velocities = np.nanmean([up_interp, down_interp], axis=0)
    
    # Remove any NaN values
    valid_mask = ~np.isnan(averaged_velocities)
    if np.sum(valid_mask) < 2:
        return np.nan
    
    final_pressures = filtered_pressures[valid_mask]
    final_velocities = averaged_velocities[valid_mask]
    
    # Calculate area using trapezoidal rule
    area = trapz(final_velocities, final_pressures)
    
    return area


def calculate_stopping_pressure(up_pressures: np.ndarray,
                                up_velocities: np.ndarray,
                                threshold: float = ZERO_FLOW_THRESHOLD) -> Optional[float]:
    """Calculate stopping pressure: first pressure where flow goes to zero in up curve.
    
    The stopping pressure is the first pressure (going from low to high) where
    velocity drops below the threshold.
    
    Args:
        up_pressures: Array of pressures for up curve (should be sorted ascending)
        up_velocities: Array of velocities for up curve
        threshold: Velocity threshold for zero flow (default: 5.0 um/s)
    
    Returns:
        Stopping pressure in psi, or None if flow never stops
    """
    # Find first index where velocity drops below threshold
    below_threshold = up_velocities < threshold
    
    if not np.any(below_threshold):
        return None
    
    first_stop_idx = np.where(below_threshold)[0][0]
    stopping_pressure = up_pressures[first_stop_idx]
    
    return float(stopping_pressure)


def calculate_restart_pressure(down_pressures: np.ndarray,
                              down_velocities: np.ndarray,
                              threshold: float = ZERO_FLOW_THRESHOLD) -> Optional[float]:
    """Calculate restart pressure: first pressure where flow returns above zero in down curve.
    
    The restart pressure is the first pressure (going from high to low) where
    velocity rises above the threshold.
    
    Args:
        down_pressures: Array of pressures for down curve (should be sorted ascending)
        down_velocities: Array of velocities for down curve
        threshold: Velocity threshold for zero flow (default: 5.0 um/s)
    
    Returns:
        Restart pressure in psi, or None if flow never restarts
    """
    # Reverse arrays to go from high to low pressure
    reversed_pressures = down_pressures[::-1]
    reversed_velocities = down_velocities[::-1]
    
    # Find first index where velocity rises above threshold
    above_threshold = reversed_velocities >= threshold
    
    if not np.any(above_threshold):
        return None
    
    first_restart_idx = np.where(above_threshold)[0][0]
    restart_pressure = reversed_pressures[first_restart_idx]
    
    return float(restart_pressure)


def calculate_stiffness_metrics(df: pd.DataFrame,
                               velocity_column: str = 'Video_Median_Velocity') -> pd.DataFrame:
    """Calculate stiffness coefficients and stopping/restart pressures for all participants.
    
    Args:
        df: DataFrame containing participant data with Pressure, UpDown, and velocity columns
        velocity_column: Name of the velocity column to use (default: 'Video_Median_Velocity')
    
    Returns:
        DataFrame with one row per participant containing:
            - Participant: Participant ID
            - stiffness_coeff_up: Area under up curve from 0.4 to 1.2 psi
            - stiffness_coeff_averaged: Area under averaged curve from 0.4 to 1.2 psi
            - stopping_pressure: Pressure where flow stops in up curve (psi)
            - restart_pressure: Pressure where flow restarts in down curve (psi)
            - Additional columns from original data (Age, Diabetes, Hypertension, etc.)
    """
    participant_data = []
    
    for participant in df['Participant'].unique():
        participant_df = df[df['Participant'] == participant].copy()
        
        # Get up and down curves
        up_pressures, up_velocities, down_pressures, down_velocities = get_up_down_curves(
            participant_df, velocity_column
        )
        
        # Calculate stiffness coefficients
        stiffness_coeff_up = calculate_stiffness_coefficient_up(
            up_pressures, up_velocities
        )
        stiffness_coeff_averaged = calculate_stiffness_coefficient_averaged(
            up_pressures, up_velocities, down_pressures, down_velocities
        )
        
        # Calculate stopping and restart pressures
        stopping_pressure = calculate_stopping_pressure(up_pressures, up_velocities)
        restart_pressure = calculate_restart_pressure(down_pressures, down_velocities)
        
        # Gather participant information
        result = {
            'Participant': participant,
            'stiffness_coeff_up': stiffness_coeff_up,
            'stiffness_coeff_averaged': stiffness_coeff_averaged,
            'stopping_pressure': stopping_pressure,
            'restart_pressure': restart_pressure
        }
        
        # Add demographic/health information if available
        for col in ['Age', 'Diabetes', 'Hypertension', 'SET', 'Sex', 'SYS_BP', 'DIA_BP']:
            if col in participant_df.columns:
                result[col] = participant_df[col].iloc[0]
        
        # Convert Diabetes to boolean if it's a string
        if 'Diabetes' in result:
            if isinstance(result['Diabetes'], str):
                result['Diabetes'] = str(result['Diabetes']).upper() == 'TRUE'
        
        # Add is_healthy flag if SET column is available
        if 'SET' in result:
            result['is_healthy'] = str(result['SET']).startswith('set01')
        
        participant_data.append(result)
    
    results_df = pd.DataFrame(participant_data)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("STIFFNESS COEFFICIENT SUMMARY")
    print("="*70)
    print(f"\nTotal participants: {len(results_df)}")
    print(f"\nStiffness Coefficient (Up Curve):")
    print(f"  Mean: {results_df['stiffness_coeff_up'].mean():.3f}")
    print(f"  Median: {results_df['stiffness_coeff_up'].median():.3f}")
    print(f"  Range: {results_df['stiffness_coeff_up'].min():.3f} to {results_df['stiffness_coeff_up'].max():.3f}")
    print(f"  Valid values: {results_df['stiffness_coeff_up'].notna().sum()}")
    
    print(f"\nStiffness Coefficient (Averaged):")
    print(f"  Mean: {results_df['stiffness_coeff_averaged'].mean():.3f}")
    print(f"  Median: {results_df['stiffness_coeff_averaged'].median():.3f}")
    print(f"  Range: {results_df['stiffness_coeff_averaged'].min():.3f} to {results_df['stiffness_coeff_averaged'].max():.3f}")
    print(f"  Valid values: {results_df['stiffness_coeff_averaged'].notna().sum()}")
    
    print(f"\nStopping Pressure:")
    valid_stopping = results_df['stopping_pressure'].notna()
    if valid_stopping.sum() > 0:
        print(f"  Mean: {results_df.loc[valid_stopping, 'stopping_pressure'].mean():.3f} psi")
        print(f"  Median: {results_df.loc[valid_stopping, 'stopping_pressure'].median():.3f} psi")
        print(f"  Range: {results_df.loc[valid_stopping, 'stopping_pressure'].min():.3f} to {results_df.loc[valid_stopping, 'stopping_pressure'].max():.3f} psi")
    print(f"  Participants with stopping: {valid_stopping.sum()}")
    
    print(f"\nRestart Pressure:")
    valid_restart = results_df['restart_pressure'].notna()
    if valid_restart.sum() > 0:
        print(f"  Mean: {results_df.loc[valid_restart, 'restart_pressure'].mean():.3f} psi")
        print(f"  Median: {results_df.loc[valid_restart, 'restart_pressure'].median():.3f} psi")
        print(f"  Range: {results_df.loc[valid_restart, 'restart_pressure'].min():.3f} to {results_df.loc[valid_restart, 'restart_pressure'].max():.3f} psi")
    print(f"  Participants with restart: {valid_restart.sum()}")
    print("="*70 + "\n")
    
    return results_df


def main():
    """Main function to calculate stiffness coefficients and stopping/restart pressures."""
    print("\nStarting stiffness coefficient calculation...")
    
    # Load data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    if not os.path.exists(data_filepath):
        print(f"Error: Data file not found at {data_filepath}")
        return 1
    
    df = pd.read_csv(data_filepath)
    print(f"Loaded data with {len(df)} rows")
    
    # Calculate stiffness metrics
    results_df = calculate_stiffness_metrics(df, velocity_column='Video_Median_Velocity')
    
    # Create output directory
    output_dir = os.path.join(cap_flow_path, 'results', 'Stiffness')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_filepath = os.path.join(output_dir, 'stiffness_coefficients.csv')
    results_df.to_csv(output_filepath, index=False)
    print(f"\nResults saved to: {output_filepath}")
    
    # Also calculate for log velocity if available
    if 'Log_Video_Median_Velocity' in df.columns:
        print("\nCalculating stiffness coefficients for log velocity...")
        results_df_log = calculate_stiffness_metrics(df, velocity_column='Log_Video_Median_Velocity')
        
        # Rename columns to indicate log velocity
        results_df_log = results_df_log.rename(columns={
            'stiffness_coeff_up': 'stiffness_coeff_up_log',
            'stiffness_coeff_averaged': 'stiffness_coeff_averaged_log',
            'stopping_pressure': 'stopping_pressure_log',
            'restart_pressure': 'restart_pressure_log'
        })
        
        # Save log velocity results
        output_filepath_log = os.path.join(output_dir, 'stiffness_coefficients_log.csv')
        results_df_log.to_csv(output_filepath_log, index=False)
        print(f"Log velocity results saved to: {output_filepath_log}")
    
    print("\nAnalysis complete.")
    return 0


if __name__ == "__main__":
    main()

