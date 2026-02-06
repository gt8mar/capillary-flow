"""
Filename: src/analysis/stiffness_coeff.py

File for calculating stiffness coefficients and stopping/restart pressures
from capillary flow velocity data.
By: Marcus Forst
"""

import os
import json
import pandas as pd
import numpy as np
from scipy.integrate import trapz
from scipy import stats
from typing import Dict, Tuple, Optional
import statsmodels.api as sm
from statsmodels.formula.api import ols

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


def calculate_p50(up_pressures: np.ndarray,
                 up_velocities: np.ndarray) -> Optional[float]:
    """Calculate P50: pressure at which velocity reaches 50% of maximum.
    
    P50 is a mechanical readout metric indicating the pressure at which
    the capillary reaches half of its maximum velocity response.
    
    Args:
        up_pressures: Array of pressures for up curve (sorted ascending)
        up_velocities: Array of velocities for up curve
    
    Returns:
        P50 pressure in psi, or None if cannot be determined
    """
    if len(up_velocities) == 0:
        return None
    
    max_velocity = np.nanmax(up_velocities)
    if np.isnan(max_velocity) or max_velocity <= 0:
        return None
    
    target_velocity = 0.5 * max_velocity
    
    # Find where velocity crosses 50% of maximum
    # Look for first pressure where velocity >= target_velocity
    above_target = up_velocities >= target_velocity
    
    if not np.any(above_target):
        return None
    
    # Interpolate to get exact pressure
    first_above_idx = np.where(above_target)[0][0]
    
    if first_above_idx == 0:
        return float(up_pressures[0])
    
    # Linear interpolation between points
    v_before = up_velocities[first_above_idx - 1]
    v_after = up_velocities[first_above_idx]
    p_before = up_pressures[first_above_idx - 1]
    p_after = up_pressures[first_above_idx]
    
    if v_after == v_before:
        return float(p_after)
    
    # Interpolate
    fraction = (target_velocity - v_before) / (v_after - v_before)
    p50 = p_before + fraction * (p_after - p_before)
    
    return float(p50)


def calculate_ev_lin(up_pressures: np.ndarray,
                     up_velocities: np.ndarray,
                     pressure_range: Tuple[float, float] = (0.2, 0.8)) -> Optional[float]:
    """Calculate EV_lin: linear elastic modulus from pressure-velocity relationship.
    
    EV_lin is a mechanical readout metric calculated as the slope of the
    linear portion of the pressure-velocity curve.
    
    Args:
        up_pressures: Array of pressures for up curve (sorted ascending)
        up_velocities: Array of velocities for up curve
        pressure_range: Tuple of (min, max) pressure for linear fit (default: 0.2-0.8 psi)
    
    Returns:
        EV_lin (slope) in (um/s)/psi, or None if cannot be determined
    """
    # Filter to pressure range
    mask = (up_pressures >= pressure_range[0]) & (up_pressures <= pressure_range[1])
    
    if np.sum(mask) < 2:
        return None
    
    filtered_pressures = up_pressures[mask]
    filtered_velocities = up_velocities[mask]
    
    # Remove NaN values
    valid = ~(np.isnan(filtered_pressures) | np.isnan(filtered_velocities))
    if np.sum(valid) < 2:
        return None
    
    filtered_pressures = filtered_pressures[valid]
    filtered_velocities = filtered_velocities[valid]
    
    # Linear fit: velocity = slope * pressure + intercept
    slope, intercept = np.polyfit(filtered_pressures, filtered_velocities, 1)
    
    return float(slope)


def calculate_stiffness_metrics(df: pd.DataFrame,
                               velocity_column: str = 'Video_Median_Velocity') -> pd.DataFrame:
    """Calculate stiffness coefficients and stopping/restart pressures for all participants.
    
    Args:
        df: DataFrame containing participant data with Pressure, UpDown, and velocity columns
        velocity_column: Name of the velocity column to use (default: 'Video_Median_Velocity')
    
    Returns:
        DataFrame with one row per participant containing:
            - Participant: Participant ID
            - stiffness_coeff_up_04_12: Area under up curve from 0.4 to 1.2 psi
            - stiffness_coeff_up_02_12: Area under up curve from 0.2 to 1.2 psi
            - stiffness_coeff_averaged_04_12: Area under averaged curve from 0.4 to 1.2 psi
            - stiffness_coeff_averaged_02_12: Area under averaged curve from 0.2 to 1.2 psi
            - stopping_pressure: Pressure where flow stops in up curve (psi)
            - restart_pressure: Pressure where flow restarts in down curve (psi)
            - MAP: Mean arterial pressure (mmHg)
            - Additional columns from original data (Age, Diabetes, Hypertension, etc.)
    """
    participant_data = []
    
    for participant in df['Participant'].unique():
        participant_df = df[df['Participant'] == participant].copy()
        
        # Get up and down curves
        up_pressures, up_velocities, down_pressures, down_velocities = get_up_down_curves(
            participant_df, velocity_column
        )
        
        # Calculate stiffness coefficients for both ranges
        stiffness_coeff_up_04_12 = calculate_stiffness_coefficient_up(
            up_pressures, up_velocities, pressure_min=0.4, pressure_max=1.2
        )
        stiffness_coeff_up_02_12 = calculate_stiffness_coefficient_up(
            up_pressures, up_velocities, pressure_min=0.2, pressure_max=1.2
        )
        stiffness_coeff_averaged_04_12 = calculate_stiffness_coefficient_averaged(
            up_pressures, up_velocities, down_pressures, down_velocities,
            pressure_min=0.4, pressure_max=1.2
        )
        stiffness_coeff_averaged_02_12 = calculate_stiffness_coefficient_averaged(
            up_pressures, up_velocities, down_pressures, down_velocities,
            pressure_min=0.2, pressure_max=1.2
        )
        
        # Calculate stopping and restart pressures
        stopping_pressure = calculate_stopping_pressure(up_pressures, up_velocities)
        restart_pressure = calculate_restart_pressure(down_pressures, down_velocities)
        
        # Calculate secondary mechanical metrics
        p50 = calculate_p50(up_pressures, up_velocities)
        ev_lin = calculate_ev_lin(up_pressures, up_velocities)
        
        # Calculate hysteresis (up-down difference)
        up_mean = np.mean(up_velocities) if len(up_velocities) > 0 else np.nan
        down_mean = np.mean(down_velocities) if len(down_velocities) > 0 else np.nan
        hysteresis = up_mean - down_mean if not (np.isnan(up_mean) or np.isnan(down_mean)) else np.nan
        
        # Get velocities at specific pressures (for composite score)
        velocity_04 = np.nan
        velocity_12 = np.nan
        if len(up_pressures) > 0:
            # Interpolate to get velocities at 0.4 and 1.2 psi
            if 0.4 in up_pressures:
                idx = np.where(up_pressures == 0.4)[0][0]
                velocity_04 = up_velocities[idx]
            elif len(up_pressures) > 1:
                velocity_04 = np.interp(0.4, up_pressures, up_velocities)
            
            if 1.2 in up_pressures:
                idx = np.where(up_pressures == 1.2)[0][0]
                velocity_12 = up_velocities[idx]
            elif len(up_pressures) > 1:
                velocity_12 = np.interp(1.2, up_pressures, up_velocities)
        
        # Gather participant information
        result = {
            'Participant': participant,
            'stiffness_coeff_up_04_12': stiffness_coeff_up_04_12,
            'stiffness_coeff_up_02_12': stiffness_coeff_up_02_12,
            'stiffness_coeff_averaged_04_12': stiffness_coeff_averaged_04_12,
            'stiffness_coeff_averaged_02_12': stiffness_coeff_averaged_02_12,
            'stopping_pressure': stopping_pressure,
            'restart_pressure': restart_pressure,
            'P50': p50,
            'EV_lin': ev_lin,
            'hysteresis': hysteresis,
            'velocity_04': velocity_04,
            'velocity_12': velocity_12
        }
        
        # Add demographic/health information if available
        for col in ['Age', 'Diabetes', 'Hypertension', 'SET', 'Sex', 'SYS_BP', 'DIA_BP']:
            if col in participant_df.columns:
                result[col] = participant_df[col].iloc[0]
        
        # Calculate MAP if SYS_BP and DIA_BP are available
        if 'SYS_BP' in result and 'DIA_BP' in result:
            if pd.notna(result['SYS_BP']) and pd.notna(result['DIA_BP']):
                result['MAP'] = result['DIA_BP'] + (result['SYS_BP'] - result['DIA_BP']) / 3
            else:
                result['MAP'] = np.nan
        else:
            result['MAP'] = np.nan
        
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
    print(f"\nStiffness Coefficient (Up Curve, 0.4-1.2 psi):")
    print(f"  Mean: {results_df['stiffness_coeff_up_04_12'].mean():.3f}")
    print(f"  Median: {results_df['stiffness_coeff_up_04_12'].median():.3f}")
    print(f"  Range: {results_df['stiffness_coeff_up_04_12'].min():.3f} to {results_df['stiffness_coeff_up_04_12'].max():.3f}")
    print(f"  Valid values: {results_df['stiffness_coeff_up_04_12'].notna().sum()}")
    
    print(f"\nStiffness Coefficient (Up Curve, 0.2-1.2 psi):")
    print(f"  Mean: {results_df['stiffness_coeff_up_02_12'].mean():.3f}")
    print(f"  Median: {results_df['stiffness_coeff_up_02_12'].median():.3f}")
    print(f"  Range: {results_df['stiffness_coeff_up_02_12'].min():.3f} to {results_df['stiffness_coeff_up_02_12'].max():.3f}")
    print(f"  Valid values: {results_df['stiffness_coeff_up_02_12'].notna().sum()}")
    
    print(f"\nStiffness Coefficient (Averaged, 0.4-1.2 psi):")
    print(f"  Mean: {results_df['stiffness_coeff_averaged_04_12'].mean():.3f}")
    print(f"  Median: {results_df['stiffness_coeff_averaged_04_12'].median():.3f}")
    print(f"  Range: {results_df['stiffness_coeff_averaged_04_12'].min():.3f} to {results_df['stiffness_coeff_averaged_04_12'].max():.3f}")
    print(f"  Valid values: {results_df['stiffness_coeff_averaged_04_12'].notna().sum()}")
    
    print(f"\nStiffness Coefficient (Averaged, 0.2-1.2 psi):")
    print(f"  Mean: {results_df['stiffness_coeff_averaged_02_12'].mean():.3f}")
    print(f"  Median: {results_df['stiffness_coeff_averaged_02_12'].median():.3f}")
    print(f"  Range: {results_df['stiffness_coeff_averaged_02_12'].min():.3f} to {results_df['stiffness_coeff_averaged_02_12'].max():.3f}")
    print(f"  Valid values: {results_df['stiffness_coeff_averaged_02_12'].notna().sum()}")
    
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


def get_classifier_weights(df: pd.DataFrame, target: str = 'Diabetes') -> Dict[str, float]:
    """Get feature importance weights from Random Forest classifier.
    
    Trains a classifier to predict health condition and extracts weights
    for velocities at 0.4 psi, 1.2 psi, and hysteresis.
    
    Args:
        df: DataFrame containing participant data
        target: Target variable ('Diabetes', 'Hypertension', or 'is_healthy')
    
    Returns:
        Dictionary with weights for 'velocity_04', 'velocity_12', and 'hysteresis'
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Prepare features
    participant_data = []
    for participant in df['Participant'].unique():
        participant_df = df[df['Participant'] == participant].copy()
        
        # Get velocities at specific pressures
        up_data = participant_df[participant_df['UpDown'] == 'U'].copy()
        down_data = participant_df[participant_df['UpDown'] == 'D'].copy()
        
        # Get mean velocities at 0.4 and 1.2 psi
        vel_04 = up_data[up_data['Pressure'] == 0.4]['Video_Median_Velocity'].mean() if len(up_data[up_data['Pressure'] == 0.4]) > 0 else np.nan
        vel_12 = up_data[up_data['Pressure'] == 1.2]['Video_Median_Velocity'].mean() if len(up_data[up_data['Pressure'] == 1.2]) > 0 else np.nan
        
        # Interpolate if exact pressure not available
        if pd.isna(vel_04) and len(up_data) > 1:
            pressures = up_data['Pressure'].values
            velocities = up_data['Video_Median_Velocity'].values
            if len(pressures) > 1:
                vel_04 = np.interp(0.4, pressures, velocities)
        
        if pd.isna(vel_12) and len(up_data) > 1:
            pressures = up_data['Pressure'].values
            velocities = up_data['Video_Median_Velocity'].values
            if len(pressures) > 1:
                vel_12 = np.interp(1.2, pressures, velocities)
        
        # Calculate hysteresis
        up_mean = up_data['Video_Median_Velocity'].mean() if len(up_data) > 0 else np.nan
        down_mean = down_data['Video_Median_Velocity'].mean() if len(down_data) > 0 else np.nan
        hyst = up_mean - down_mean if not (pd.isna(up_mean) or pd.isna(down_mean)) else np.nan
        
        # Get target variable
        if target == 'is_healthy':
            target_val = str(participant_df['SET'].iloc[0]).startswith('set01') if 'SET' in participant_df.columns else False
        else:
            target_val = str(participant_df[target].iloc[0]).upper() == 'TRUE' if target in participant_df.columns else False
        
        participant_data.append({
            'Participant': participant,
            'velocity_04': vel_04,
            'velocity_12': vel_12,
            'hysteresis': hyst,
            target: target_val
        })
    
    feature_df = pd.DataFrame(participant_data)
    
    # Remove rows with missing values
    feature_df = feature_df.dropna(subset=['velocity_04', 'velocity_12', 'hysteresis', target])
    
    if len(feature_df) < 10:
        print(f"Warning: Not enough data for classifier. Using default weights.")
        return {'velocity_04': 0.33, 'velocity_12': 0.33, 'hysteresis': 0.34}
    
    # Prepare features and target
    X = feature_df[['velocity_04', 'velocity_12', 'hysteresis']].values
    y = feature_df[target].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    weights = {
        'velocity_04': float(importances[0]),
        'velocity_12': float(importances[1]),
        'hysteresis': float(importances[2])
    }
    
    # Normalize weights to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    
    print(f"\nClassifier weights for {target}:")
    print(f"  velocity_04: {weights['velocity_04']:.3f}")
    print(f"  velocity_12: {weights['velocity_12']:.3f}")
    print(f"  hysteresis: {weights['hysteresis']:.3f}")
    
    return weights


def calculate_composite_stiffness(results_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Calculate composite stiffness score using classifier weights.
    
    Composite Stiffness = w1 * V(0.4) + w2 * V(1.2) + w3 * H
    
    Args:
        results_df: DataFrame with stiffness metrics
        weights: Dictionary with weights for velocity_04, velocity_12, and hysteresis
    
    Returns:
        Series with composite stiffness scores
    """
    composite = (
        weights['velocity_04'] * results_df['velocity_04'].fillna(0) +
        weights['velocity_12'] * results_df['velocity_12'].fillna(0) +
        weights['hysteresis'] * results_df['hysteresis'].fillna(0)
    )
    
    return composite


def age_adjusted_analysis(results_df: pd.DataFrame, stiffness_col: str, 
                          group_col: str = 'Diabetes') -> Dict:
    """Perform age-adjusted analysis using ANCOVA/linear regression.
    
    SI ~ Group + Age
    
    Args:
        results_df: DataFrame with stiffness metrics
        stiffness_col: Name of stiffness column to analyze
        group_col: Name of group column (default: 'Diabetes')
    
    Returns:
        Dictionary with regression results including p-values
    """
    # Filter to valid data
    df_clean = results_df[[stiffness_col, group_col, 'Age']].dropna()
    
    if len(df_clean) < 10:
        return {'error': 'Insufficient data for analysis'}
    
    # Create binary group variable
    if group_col == 'Diabetes':
        df_clean['Group'] = df_clean[group_col].astype(int)  # 1 = Diabetic, 0 = Control
    else:
        df_clean['Group'] = df_clean[group_col].astype(int)
    
    # Fit linear regression: SI ~ Group + Age
    formula = f'{stiffness_col} ~ Group + Age'
    model = ols(formula, data=df_clean).fit()
    
    # Extract results
    results = {
        'formula': formula,
        'n': len(df_clean),
        'group_coef': model.params['Group'],
        'group_pvalue': model.pvalues['Group'],
        'age_coef': model.params['Age'],
        'age_pvalue': model.pvalues['Age'],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'model_summary': str(model.summary())
    }
    
    return results


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
    
    # Calculate composite stiffness using classifier weights
    print("\nCalculating composite stiffness scores...")
    weights = get_classifier_weights(df, target='Diabetes')
    results_df['composite_stiffness'] = calculate_composite_stiffness(results_df, weights)
    
    # Calculate log-transformed SI_AUC (log(SI_AUC + 1))
    for col in ['stiffness_coeff_up_04_12', 'stiffness_coeff_up_02_12',
                'stiffness_coeff_averaged_04_12', 'stiffness_coeff_averaged_02_12']:
        if col in results_df.columns:
            results_df[f'log_{col}'] = np.log1p(results_df[col])  # log1p = log(1 + x)
    
    # Also calculate log for composite stiffness
    results_df['log_composite_stiffness'] = np.log1p(results_df['composite_stiffness'])
    
    # Create output directory
    output_dir = os.path.join(cap_flow_path, 'results', 'Stiffness')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_filepath = os.path.join(output_dir, 'stiffness_coefficients.csv')
    results_df.to_csv(output_filepath, index=False)
    print(f"\nResults saved to: {output_filepath}")
    
    # Perform age-adjusted analysis
    print("\nPerforming age-adjusted analysis...")
    age_adjusted_results = {}
    for stiffness_col in ['stiffness_coeff_averaged_04_12', 'composite_stiffness']:
        if stiffness_col in results_df.columns:
            age_results = age_adjusted_analysis(results_df, stiffness_col, group_col='Diabetes')
            age_adjusted_results[stiffness_col] = age_results
            if 'error' not in age_results:
                print(f"\n{stiffness_col} - Age-adjusted analysis:")
                print(f"  Group coefficient: {age_results['group_coef']:.4f}")
                print(f"  Group p-value: {age_results['group_pvalue']:.4f}")
                print(f"  Age coefficient: {age_results['age_coef']:.4f}")
                print(f"  Age p-value: {age_results['age_pvalue']:.4f}")
                print(f"  R-squared: {age_results['r_squared']:.4f}")
    
    # Save age-adjusted results
    if age_adjusted_results:
        age_filepath = os.path.join(output_dir, 'age_adjusted_analysis.json')
        # Convert numpy types to Python types for JSON serialization
        age_adjusted_json = {}
        for key, value in age_adjusted_results.items():
            if isinstance(value, dict) and 'error' not in value:
                age_adjusted_json[key] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in value.items() if k != 'model_summary'
                }
        with open(age_filepath, 'w') as f:
            json.dump(age_adjusted_json, f, indent=2)
        print(f"\nAge-adjusted analysis saved to: {age_filepath}")
    
    # Also calculate for log velocity if available
    if 'Log_Video_Median_Velocity' in df.columns:
        print("\nCalculating stiffness coefficients for log velocity...")
        results_df_log = calculate_stiffness_metrics(df, velocity_column='Log_Video_Median_Velocity')
        
        # Calculate composite stiffness for log velocity
        weights_log = get_classifier_weights(df, target='Diabetes')
        results_df_log['composite_stiffness'] = calculate_composite_stiffness(results_df_log, weights_log)
        
        # Calculate log-transformed SI_AUC for log velocity
        for col in ['stiffness_coeff_up_04_12', 'stiffness_coeff_up_02_12',
                    'stiffness_coeff_averaged_04_12', 'stiffness_coeff_averaged_02_12']:
            if col in results_df_log.columns:
                results_df_log[f'log_{col}'] = np.log1p(results_df_log[col])
        
        results_df_log['log_composite_stiffness'] = np.log1p(results_df_log['composite_stiffness'])
        
        # Rename columns to indicate log velocity
        results_df_log = results_df_log.rename(columns={
            'stiffness_coeff_up_04_12': 'stiffness_coeff_up_04_12_log',
            'stiffness_coeff_up_02_12': 'stiffness_coeff_up_02_12_log',
            'stiffness_coeff_averaged_04_12': 'stiffness_coeff_averaged_04_12_log',
            'stiffness_coeff_averaged_02_12': 'stiffness_coeff_averaged_02_12_log',
            'stopping_pressure': 'stopping_pressure_log',
            'restart_pressure': 'restart_pressure_log',
            'composite_stiffness': 'composite_stiffness_log',
            'log_composite_stiffness': 'log_composite_stiffness_log'
        })
        
        # Save log velocity results
        output_filepath_log = os.path.join(output_dir, 'stiffness_coefficients_log.csv')
        results_df_log.to_csv(output_filepath_log, index=False)
        print(f"Log velocity results saved to: {output_filepath_log}")
    
    print("\nAnalysis complete.")
    return 0


if __name__ == "__main__":
    main()

