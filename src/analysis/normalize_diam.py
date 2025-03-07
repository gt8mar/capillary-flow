"""
Filename: normalize_diam.py
------------------------------------------------------
This script normalizes capillary velocities by diameter, calculates fluid dynamics 
parameters (Reynolds number, shear rate, shear stress), and creates a dimensionless 
pressure-flow index. It also provides functions to plot these metrics by different 
demographic and clinical groups.

By: Marcus Forst
"""

# Standard library imports
import os
import platform
from typing import Dict, List, Tuple, Union, Optional, Callable

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from scipy import stats

# Local imports
from src.tools.parse_filename import parse_filename

# Get the hostname and set paths
hostname = platform.node()
computer_paths = {
    "LAPTOP-I5KTBOR3": {
        'cap_flow': 'C:\\Users\\gt8ma\\capillary-flow',
        'downloads': 'C:\\Users\\gt8ma\\Downloads'
    },
    "Quake-Blood": {
        'cap_flow': "C:\\Users\\gt8mar\\capillary-flow",
        'downloads': 'C:\\Users\\gt8mar\\Downloads'
    },
}
default_paths = {
    'cap_flow': "/hpc/projects/capillary-flow",
    'downloads': "/home/downloads"
}

# Determine paths based on hostname
paths = computer_paths.get(hostname, default_paths)
cap_flow_path = paths['cap_flow']
downloads_path = paths['downloads']

# Set up font and plot styling
try:
    source_sans = FontProperties(fname=os.path.join(downloads_path, 'Source_Sans_3', 'static', 'SourceSans3-Regular.ttf'))
    
    # Standard plot configuration
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
except Exception as e:
    print(f"Warning: Could not set up font: {e}")
    source_sans = None

# Base colors for different categories
base_colors = {
    'default': '#1f77b4',
    'diabetes': '#ff7f0e',
    'hypertension': '#2ca02c',
    'heart_disease': '#d62728',
    'young': '#1f77b4',
    'old': '#ff7f0e',
    'male': '#1f77b4',
    'female': '#ff7f0e',
    'normal_bp': '#1f77b4',
    'high_bp': '#ff7f0e'
}

def to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def create_monochromatic_palette(base_color, n_colors=5):
    """Creates a monochromatic palette based on the given color."""
    import matplotlib.colors as mcolors
    
    # Convert hex to RGB if needed
    if isinstance(base_color, str) and base_color.startswith('#'):
        base_rgb = to_rgb(base_color)
    else:
        base_rgb = base_color
        
    # Convert to HSV
    base_hsv = mcolors.rgb_to_hsv(base_rgb)
    
    # Create variations by adjusting saturation and value
    colors = []
    for i in range(n_colors):
        # Adjust saturation and value, keeping hue constant
        sat = max(0.4, base_hsv[1] - 0.15 * (i - n_colors//2))
        val = min(0.9, base_hsv[2] + 0.15 * (i - n_colors//2))
        
        # Convert back to RGB and then to hex
        new_rgb = mcolors.hsv_to_rgb((base_hsv[0], sat, val))
        colors.append(mcolors.rgb2hex(new_rgb))
        
    return colors

def load_and_clean_data(velocity_file_path: str = None, metadata_file_path: str = None) -> pd.DataFrame:
    """
    Load and clean the capillary diameter and velocity data.
    
    Args:
        velocity_file_path: Path to the velocity data file
        metadata_file_path: Path to the participant metadata file (optional)
        
    Returns:
        Cleaned DataFrame with capillary measurements
    """
    # Set default file paths if not provided
    if velocity_file_path is None:
        velocity_file_path = os.path.join(cap_flow_path, 'results', 'cap_diameters_areas_merged_with_velocity.csv')
    
    print(f"Loading velocity data from: {velocity_file_path}")
    
    # Load the velocity data
    try:
        df = pd.read_csv(velocity_file_path)
        print(f"Successfully loaded velocity data with {len(df)} entries")
    except Exception as e:
        print(f"Error loading velocity data: {e}")
        return pd.DataFrame()
    
    # Rename _x columns without the _x
    x_columns = [col for col in df.columns if col.endswith('_x')]
    rename_dict = {col: col[:-2] for col in x_columns}
    df_clean = df.rename(columns=rename_dict)

    # Remove rows with empty Area or empty Corrected_Velocity
    df_clean = df_clean.dropna(subset=['Area', 'Corrected_Velocity'])
    print(f"Removed {len(df) - len(df_clean)} rows with missing Area or Corrected_Velocity")

    
    # Filter out videos with 'bp' in their names
    # First, check if any videos in the dataframe have 'bp' in their names
    bp_videos = df_clean[df_clean['Video'].astype(str).str.contains('bp', case=False)]
    if len(bp_videos) > 0:
        print(f"Found {len(bp_videos)} entries with 'bp' in Video name")
        df_clean = df_clean[~df_clean['Video'].astype(str).str.contains('bp', case=False)]
        print(f"Removed {len(bp_videos)} entries with 'bp' in Video name")
    
    # If metadata file is provided, use it to identify additional 'bp' videos
    if metadata_file_path:
        try:
            metadata = pd.read_csv(metadata_file_path)
            print(f"Successfully loaded metadata with {len(metadata)} entries")
            
            # Find videos in metadata that have 'bp' in their names
            bp_videos_in_metadata = metadata[metadata['Video'].astype(str).str.contains('bp', case=False)]
            
            if len(bp_videos_in_metadata) > 0:
                print(f"Found {len(bp_videos_in_metadata)} videos with 'bp' in metadata")
                
                # Create a set of (Participant, Date, Location, Video) tuples to filter out
                bp_video_keys = set()
                for _, row in bp_videos_in_metadata.iterrows():
                    # Remove 'bp' from the video name to match with velocity data
                    clean_video = str(row['Video']).replace('bp', '')
                    bp_video_keys.add((row['Participant'], row['Date'], row['Location'], clean_video))
                
                # Filter out these videos from the velocity data
                initial_count = len(df_clean)
                df_clean = df_clean[~df_clean.apply(
                    lambda row: (row['Participant'], row['Date'], row['Location'], row['Video']) in bp_video_keys, 
                    axis=1
                )]
                
                removed_count = initial_count - len(df_clean)
                print(f"Removed {removed_count} additional entries based on metadata 'bp' videos")
            else:
                print("No videos with 'bp' found in metadata")
                
        except Exception as e:
            print(f"Error loading metadata: {e}")
            print("Continuing with filtering based only on velocity data")
    
    print(f"Final cleaned dataset has {len(df_clean)} entries")
    return df_clean

def normalize_velocities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize corrected velocities by diameter and diameter squared.
    
    Args:
        df: DataFrame containing Corrected_Velocity and Mean_Diameter columns
        
    Returns:
        DataFrame with added normalized velocity columns
    """
    # Make a copy to avoid modifying the original
    df_norm = df.copy()
    
    # 1) Normalize by diameter (V/D)
    df_norm['V_over_D'] = df_norm['Corrected_Velocity'] / df_norm['Mean_Diameter']
    
    # 2) Normalize by diameter squared (V/D²)
    df_norm['V_over_D2'] = df_norm['Corrected_Velocity'] / (df_norm['Mean_Diameter'] ** 2)
    
    return df_norm

def calculate_secomb_viscosity(radius_um: float, discharge_hematocrit: float = 0.45) -> float:
    """
    Calculate blood viscosity using Secomb's model that accounts for the
    Fahraeus-Lindqvist effect (viscosity dependence on vessel diameter).
    
    Args:
        radius_um: Vessel radius in micrometers or pixels
        discharge_hematocrit: Blood hematocrit (default 0.45)
        
    Returns:
        Apparent viscosity in cP (centipoise)
    """
    # Conversion factor: 2.44 pixels per micrometer
    PIX_UM = 2.44
    
    # Convert from pixels to micrometers if needed
    if radius_um > 100:  # If input is likely in pixels
        radius_um = radius_um / PIX_UM  # Convert pixels to micrometers
    
    # Parameters for Secomb's model
    plasma_viscosity = 1.2  # cP
    
    # Minimum apparent viscosity occurs around 7-8 μm diameter
    if radius_um < 2.0:
        # For very small vessels, viscosity decreases (plasma skimming)
        relative_viscosity = 1.0 + 2.0 * (radius_um / 2.0) * discharge_hematocrit
    elif radius_um < 4.0:
        # Fahraeus-Lindqvist effect is strongest in this range
        relative_viscosity = 1.0 + (radius_um / 4.0) * discharge_hematocrit
    else:
        # For larger vessels, approach bulk blood viscosity
        # Asymptotic value of ~3.5-4.0 times plasma viscosity for normal hematocrit
        max_relative_viscosity = 3.5
        relative_viscosity = max_relative_viscosity - (max_relative_viscosity - 1.0) * np.exp(-(radius_um - 4.0) / 10.0)
    
    return plasma_viscosity * relative_viscosity

def calculate_fluid_dynamics(
    df: pd.DataFrame, 
    use_secomb_viscosity: bool = False,
    hematocrit: float = 0.45,
    viscosity_function: Optional[Callable] = None
) -> pd.DataFrame:
    """
    Compute Reynolds number, shear rate, and shear stress for each capillary.
    
    Args:
        df: DataFrame with velocity and diameter data
        use_secomb_viscosity: Whether to use Secomb's viscosity model
        hematocrit: Blood hematocrit (default 0.45)
        viscosity_function: Optional custom function to calculate viscosity
        
    Returns:
        DataFrame with added fluid dynamics columns
    """
    # Make a copy to avoid modifying the original
    df_fluid = df.copy()
    
    # Constants
    rho = 1060  # Blood density (kg/m³)
    default_viscosity = 3.5  # Blood viscosity (cP)
    
    # Conversion factors
    PIX_UM = 2.44  # 2.44 pixels per micrometer
    UM_M = 1e-6    # 1 micrometer = 1e-6 meters
    
    # Calculate diameter and radius in meters
    # First convert from pixels to micrometers, then from micrometers to meters
    df_fluid['Diameter_um'] = df_fluid['Mean_Diameter'] / PIX_UM
    df_fluid['Diameter_m'] = df_fluid['Diameter_um'] * UM_M
    df_fluid['Radius_m'] = df_fluid['Diameter_m'] / 2
    
    # Calculate velocity in m/s (convert from pixels/s to m/s)
    df_fluid['Velocity_m_s'] = df_fluid['Corrected_Velocity'] / PIX_UM * UM_M
    
    # Calculate viscosity based on the selected method
    if viscosity_function is not None:
        # Use custom viscosity function if provided
        df_fluid['Viscosity_Pa_s'] = df_fluid.apply(
            lambda row: viscosity_function(row['Mean_Diameter'] / 2, hematocrit) * 1e-3,  # Convert cP to Pa·s
            axis=1
        )
    elif use_secomb_viscosity:
        # Use Secomb's viscosity model
        df_fluid['Viscosity_Pa_s'] = df_fluid.apply(
            lambda row: calculate_secomb_viscosity(row['Mean_Diameter'] / 2, hematocrit) * 1e-3,  # Convert cP to Pa·s
            axis=1
        )
    else:
        # Use default constant viscosity
        df_fluid['Viscosity_Pa_s'] = default_viscosity * 1e-3  # Convert cP to Pa·s
    
    # Calculate Reynolds number: Re = ρvD/μ
    df_fluid['Reynolds_Number'] = (rho * df_fluid['Velocity_m_s'] * df_fluid['Diameter_m']) / df_fluid['Viscosity_Pa_s']
    
    # Calculate shear rate: γ = 8v/D (for Newtonian fluid in a tube)
    df_fluid['Shear_Rate_s'] = 8 * df_fluid['Velocity_m_s'] / df_fluid['Diameter_m']
    
    # Calculate shear stress: τ = μγ
    df_fluid['Shear_Stress_Pa'] = df_fluid['Viscosity_Pa_s'] * df_fluid['Shear_Rate_s']
    
    return df_fluid

def calculate_pressure_flow_index(
    df: pd.DataFrame,
    pressure_method: str = 'capillary',
    constant_pressure: float = None
) -> pd.DataFrame:
    """
    Calculate dimensionless pressure-flow index Nv = (v/D²)(μ/ΔP).
    
    Args:
        df: DataFrame with velocity, diameter, and blood pressure data
        pressure_method: Method to calculate pressure drop:
            - 'capillary': Uses physiological capillary pressure drop (~20 mmHg)
            - 'SYS': Uses systolic pressure minus venous pressure
            - 'DIA': Uses diastolic pressure minus venous pressure
            - 'MAP': Uses mean arterial pressure minus venous pressure
            - 'constant': Uses a constant pressure value
        constant_pressure: Value to use if pressure_method is 'constant'
        
    Returns:
        DataFrame with added pressure-flow index
    """
    # Make a copy to avoid modifying the original
    df_pressure = df.copy()
    
    # Calculate pressure drop based on selected method
    if pressure_method == 'capillary':
        # Use physiological capillary pressure drop
        # Arterial end of capillary: ~30 mmHg, Venous end: ~10 mmHg
        # This is more accurate for nailfold capillaries than using systemic pressures
        df_pressure['Delta_P_mmHg'] = 20.0  # Fixed 20 mmHg pressure drop across capillaries
        print("Using physiological capillary pressure drop of 20 mmHg")
    
    elif pressure_method == 'capillary_scaled':
        # Scale capillary pressure based on MAP but maintain physiological range
        # This assumes capillary pressure correlates with systemic pressure but with appropriate scaling
        if 'SYS_BP' in df_pressure.columns and 'DIA_BP' in df_pressure.columns:
            # Calculate MAP
            df_pressure['MAP_mmHg'] = df_pressure['DIA_BP'] + (df_pressure['SYS_BP'] - df_pressure['DIA_BP']) / 3
            
            # Scale capillary pressure: baseline 20 mmHg at MAP of 93 mmHg
            # For every 10 mmHg change in MAP, capillary pressure changes by ~2 mmHg
            baseline_map = 93.0
            df_pressure['Delta_P_mmHg'] = 20.0 + 0.2 * (df_pressure['MAP_mmHg'] - baseline_map)
            
            # Ensure pressure drop stays in physiological range (15-30 mmHg)
            df_pressure['Delta_P_mmHg'] = df_pressure['Delta_P_mmHg'].clip(15.0, 30.0)
            
            print("Using scaled capillary pressure drop based on MAP")
        else:
            print("Warning: Blood pressure columns not found. Using default capillary pressure drop.")
            df_pressure['Delta_P_mmHg'] = 20.0
    
    elif pressure_method == 'SYS':
        # Use systolic pressure
        if 'SYS_BP' in df_pressure.columns:
            df_pressure['Delta_P_mmHg'] = df_pressure['SYS_BP'] - 10  # Assuming venous pressure is 10 mmHg
        else:
            print("Warning: SYS_BP column not found. Using default pressure.")
            df_pressure['Delta_P_mmHg'] = 120 - 10
    
    elif pressure_method == 'DIA':
        # Use diastolic pressure
        if 'DIA_BP' in df_pressure.columns:
            df_pressure['Delta_P_mmHg'] = df_pressure['DIA_BP'] - 10
        else:
            print("Warning: DIA_BP column not found. Using default pressure.")
            df_pressure['Delta_P_mmHg'] = 80 - 10
    
    elif pressure_method == 'MAP':
        # Use mean arterial pressure: MAP ≈ DIA + (SYS - DIA)/3
        if 'SYS_BP' in df_pressure.columns and 'DIA_BP' in df_pressure.columns:
            df_pressure['MAP_mmHg'] = df_pressure['DIA_BP'] + (df_pressure['SYS_BP'] - df_pressure['DIA_BP']) / 3
            df_pressure['Delta_P_mmHg'] = df_pressure['MAP_mmHg'] - 10
        else:
            print("Warning: Blood pressure columns not found. Using default MAP.")
            df_pressure['Delta_P_mmHg'] = 93 - 10  # Default MAP of 93 mmHg
    
    elif pressure_method == 'constant':
        # Use a constant pressure drop
        if constant_pressure is not None:
            df_pressure['Delta_P_mmHg'] = constant_pressure
        else:
            print("Warning: No constant pressure provided. Using default 20 mmHg.")
            df_pressure['Delta_P_mmHg'] = 20
    
    else:
        print(f"Warning: Unknown pressure method '{pressure_method}'. Using capillary pressure drop.")
        df_pressure['Delta_P_mmHg'] = 20.0
    
    # Convert pressure from mmHg to Pa
    mmHg_to_Pa = 133.322
    df_pressure['Delta_P_Pa'] = df_pressure['Delta_P_mmHg'] * mmHg_to_Pa
    
    # Calculate dimensionless pressure-flow index: Nv = (v/D²)(μ/ΔP)
    df_pressure['Pressure_Flow_Index'] = (df_pressure['V_over_D2'] * df_pressure['Viscosity_Pa_s']) / df_pressure['Delta_P_Pa']
    
    return df_pressure

def calculate_stats(group, ci_percentile=95, dimensionless=False):
    """
    Calculate statistics for a group of data.
    
    Args:
        group: Group data to analyze
        ci_percentile: Confidence interval percentile
        dimensionless: Whether to use dimensionless values
        
    Returns:
        Dictionary of statistics
    """
    # Remove NaN values
    group = group.dropna()
    
    if len(group) == 0:
        return {
            'median': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n': 0
        }
    
    # Calculate statistics
    median = np.median(group)
    mean = np.mean(group)
    std = np.std(group)
    n = len(group)
    
    # Calculate confidence interval
    alpha = 1 - (ci_percentile / 100)
    ci_lower = np.percentile(group, alpha/2 * 100)
    ci_upper = np.percentile(group, (1 - alpha/2) * 100)
    
    return {
        'median': median,
        'mean': mean,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n
    }

def plot_CI(
    df: pd.DataFrame,
    variable: str = 'Age',
    method: str = 'bootstrap',
    n_iterations: int = 1000,
    ci_percentile: float = 99.5,
    write: bool = True,
    dimensionless: bool = False,
    use_fractional: bool = True,
    baseline_pressure: float = 0.2,
    baseline_method: str = 'median',
    velocity_type: str = 'Corrected_Velocity',
    log_scale: bool = False,
    figsize: Tuple[float, float] = (2.4, 2.0)
):
    """
    Plot confidence intervals for velocity vs pressure by group.
    
    Args:
        df: DataFrame with velocity and pressure data
        variable: Variable to group by ('Age', 'Diabetes', 'Sex', 'BP', 'Hypertension')
        method: Method for calculating confidence intervals
        n_iterations: Number of bootstrap iterations
        ci_percentile: Confidence interval percentile
        write: Whether to save the plot
        dimensionless: Whether to use dimensionless values
        use_fractional: Whether to use fractional velocities (relative to baseline)
        baseline_pressure: Pressure value to use as baseline
        baseline_method: Method to calculate baseline ('median' or 'mean')
        # velocity_type: Type of velocity to use ('Corrected_Velocity', 'V_over_D', 'V_over_D2', 'Pressure_Flow_Index') # TODO: should we add reynolds number?
        log_scale: Whether to use log scale for y-axis
        figsize: Figure size
        
    Returns:
        0 if successful, 1 if error occurred
    """
    # Create a copy of the dataframe
    plot_df = df.copy()
    
    # Define groups based on variable
    if variable == 'Age':
        # Define age threshold (e.g., 50 years)
        age_threshold = 50
        plot_df['Group'] = np.where(plot_df['Age'] <= age_threshold, 'Young', 'Old')
        group_colors = [base_colors['young'], base_colors['old']]
        group_labels = [f'Age ≤ {age_threshold}', f'Age > {age_threshold}']
    
    elif variable == 'Diabetes':
        plot_df['Group'] = np.where(plot_df['Diabetes'] == True, 'Diabetes', 'No Diabetes')
        group_colors = [base_colors['default'], base_colors['diabetes']]
        group_labels = ['No Diabetes', 'Diabetes']
    
    elif variable == 'Sex':
        plot_df['Group'] = plot_df['Sex']
        group_colors = [base_colors['male'], base_colors['female']]
        group_labels = ['Male', 'Female']
    
    elif variable == 'BP' or variable == 'Hypertension':
        # Define BP threshold (e.g., 130 mmHg systolic)
        bp_threshold = 130
        if 'SYS_BP' in plot_df.columns:
            plot_df['Group'] = np.where(plot_df['SYS_BP'] <= bp_threshold, 'Normal BP', 'High BP')
        elif 'Hypertension' in plot_df.columns:
            plot_df['Group'] = np.where(plot_df['Hypertension'] == True, 'High BP', 'Normal BP')
        else:
            print("Warning: Neither SYS_BP nor Hypertension columns found.")
            return 1
        
        group_colors = [base_colors['normal_bp'], base_colors['high_bp']]
        group_labels = [f'SYS ≤ {bp_threshold}', f'SYS > {bp_threshold}']
    
    else:
        print(f"Warning: Unknown variable '{variable}'. Using Age as default.")
        age_threshold = 50
        plot_df['Group'] = np.where(plot_df['Age'] <= age_threshold, 'Young', 'Old')
        group_colors = [base_colors['young'], base_colors['old']]
        group_labels = [f'Age ≤ {age_threshold}', f'Age > {age_threshold}']
    
    # Calculate baseline velocities for each participant if using fractional velocities
    if use_fractional:
        # Group by participant and group
        baseline_groups = plot_df[plot_df['Pressure'] == baseline_pressure].groupby(['Participant', 'Group'])
        
        # Calculate baseline velocity for each participant
        baseline_velocities = {}
        for (participant, group), data in baseline_groups:
            if baseline_method == 'median':
                baseline_vel = data[velocity_type].median()
            else:  # mean
                baseline_vel = data[velocity_type].mean()
            
            baseline_velocities[(participant, group)] = baseline_vel
        
        # Apply baseline normalization
        def normalize_by_baseline(row):
            key = (row['Participant'], row['Group'])
            if key in baseline_velocities and baseline_velocities[key] > 0:
                return row[velocity_type] / baseline_velocities[key]
            return np.nan
        
        plot_df['Normalized_Velocity'] = plot_df.apply(normalize_by_baseline, axis=1)
        
        # Use normalized velocity for plotting
        plot_column = 'Normalized_Velocity'
        y_label = f'Fractional {velocity_type.replace("_", " ")}'
    else:
        # Use the selected velocity type directly
        plot_column = velocity_type
        y_label = velocity_type.replace("_", " ")
    
    # Get unique pressures and sort them
    pressures = sorted(plot_df['Pressure'].unique())
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot for each group
    for i, group_name in enumerate(plot_df['Group'].unique()):
        group_data = plot_df[plot_df['Group'] == group_name]
        
        # Calculate statistics for each pressure
        pressure_stats = []
        for pressure in pressures:
            pressure_data = group_data[group_data['Pressure'] == pressure][plot_column]
            stats = calculate_stats(pressure_data, ci_percentile, dimensionless)
            pressure_stats.append((pressure, stats))
        
        # Extract data for plotting
        x_values = [p for p, _ in pressure_stats]
        y_values = [s['median'] for _, s in pressure_stats]
        ci_lower = [s['ci_lower'] for _, s in pressure_stats]
        ci_upper = [s['ci_upper'] for _, s in pressure_stats]
        
        # Plot median line
        plt.plot(x_values, y_values, '-', color=group_colors[i], label=f"{group_labels[i]} (n={len(group_data['Participant'].unique())})")
        
        # Plot confidence interval
        plt.fill_between(x_values, ci_lower, ci_upper, color=group_colors[i], alpha=0.2)
    
    # Set axis labels and title
    plt.xlabel('Applied Pressure (psi)')
    plt.ylabel(y_label)
    
    if use_fractional:
        plt.title(f'Fractional {velocity_type.replace("_", " ")} vs Pressure by {variable}')
        # Set y-axis to start at 0 for fractional values
        plt.ylim(bottom=0)
    else:
        plt.title(f'{velocity_type.replace("_", " ")} vs Pressure by {variable}')
    
    # Apply log scale if requested
    if log_scale:
        plt.yscale('log')
    
    # Add legend
    plt.legend(loc='best', frameon=True)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if requested
    if write:
        # Create directory if it doesn't exist
        save_dir = os.path.join(cap_flow_path, 'results', 'normalized_velocities')
        os.makedirs(save_dir, exist_ok=True)
        
        # Create filename
        fractional_str = 'fractional_' if use_fractional else ''
        log_str = 'log_' if log_scale else ''
        filename = f"{fractional_str}{log_str}{velocity_type}_{variable}.png"
        
        # Save figure
        plt.savefig(os.path.join(save_dir, filename), dpi=600, bbox_inches='tight')
        print(f"Figure saved to: {os.path.join(save_dir, filename)}")
    
    # Show figure
    plt.show()
    
    return 0

def main():
    """Main function to run the analysis."""
    # Load and clean data
    df = load_and_clean_data()
    
    if len(df) == 0:
        print("Error: No data to analyze.")
        return 1
    
    # Normalize velocities
    df = normalize_velocities(df)
    
    # Calculate fluid dynamics parameters
    df = calculate_fluid_dynamics(df, use_secomb_viscosity=True)
    
    # Calculate pressure-flow index
    df = calculate_pressure_flow_index(df, pressure_method='MAP')
    
    # Plot results
    plot_CI(df, variable='Age', use_fractional=True, velocity_type='Corrected_Velocity')
    plot_CI(df, variable='Age', use_fractional=False, velocity_type='V_over_D')
    plot_CI(df, variable='Age', use_fractional=False, velocity_type='V_over_D2')
    
    # Plot for other variables
    for variable in ['Diabetes', 'Sex', 'BP']:
        plot_CI(df, variable=variable, use_fractional=True, velocity_type='Corrected_Velocity')
    
    # Plot pressure-flow index
    plot_CI(df, variable='Age', use_fractional=False, velocity_type='Pressure_Flow_Index')
    
    # Plot fluid dynamics parameters for different demographic groups
    plot_fluid_dynamics_by_group(df)
    
    return 0

def plot_fluid_dynamics_by_group(df: pd.DataFrame):
    """
    Plot fluid dynamics parameters (Reynolds number, shear rate, shear stress) 
    for different demographic groups.
    
    Args:
        df: DataFrame with calculated fluid dynamics parameters
    """
    # Check if fluid dynamics parameters have been calculated
    required_columns = ['Reynolds_Number', 'Shear_Rate_s', 'Shear_Stress_Pa']
    if not all(col in df.columns for col in required_columns):
        print("Error: Fluid dynamics parameters not found in DataFrame.")
        return
    
    print("Plotting fluid dynamics parameters by demographic groups...")
    
    # Plot Reynolds number for different groups
    for variable in ['Age', 'Diabetes', 'Sex', 'BP']:
        # Reynolds number
        plot_CI(df, variable=variable, use_fractional=False, velocity_type='Reynolds_Number',
                log_scale=True, figsize=(3.0, 2.4))
        
        # Shear rate
        plot_CI(df, variable=variable, use_fractional=False, velocity_type='Shear_Rate_s',
                log_scale=True, figsize=(3.0, 2.4))
        
        # Shear stress
        plot_CI(df, variable=variable, use_fractional=False, velocity_type='Shear_Stress_Pa',
                log_scale=True, figsize=(3.0, 2.4))
    
    # Create scatter plots of diameter vs fluid dynamics parameters
    plot_diameter_vs_fluid_dynamics(df)
    
    print("Fluid dynamics plots completed.")

def plot_diameter_vs_fluid_dynamics(df: pd.DataFrame):
    """
    Create scatter plots showing the relationship between capillary diameter
    and fluid dynamics parameters.
    
    Args:
        df: DataFrame with calculated fluid dynamics parameters
    """
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    
    # Plot diameter vs Reynolds number
    sns.scatterplot(x='Diameter_um', y='Reynolds_Number', hue='Age', 
                   data=df, alpha=0.6, s=15, ax=axes[0])
    axes[0].set_xlabel('Diameter (μm)')
    axes[0].set_ylabel('Reynolds Number')
    axes[0].set_title('Diameter vs Reynolds Number')
    axes[0].set_yscale('log')
    
    # Plot diameter vs shear rate
    sns.scatterplot(x='Diameter_um', y='Shear_Rate_s', hue='Age', 
                   data=df, alpha=0.6, s=15, ax=axes[1])
    axes[1].set_xlabel('Diameter (μm)')
    axes[1].set_ylabel('Shear Rate (s⁻¹)')
    axes[1].set_title('Diameter vs Shear Rate')
    axes[1].set_yscale('log')
    
    # Plot diameter vs shear stress
    sns.scatterplot(x='Diameter_um', y='Shear_Stress_Pa', hue='Age', 
                   data=df, alpha=0.6, s=15, ax=axes[2])
    axes[2].set_xlabel('Diameter (μm)')
    axes[2].set_ylabel('Shear Stress (Pa)')
    axes[2].set_title('Diameter vs Shear Stress')
    axes[2].set_yscale('log')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_dir = os.path.join(cap_flow_path, 'results', 'fluid_dynamics')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'diameter_vs_fluid_dynamics.png'), dpi=600, bbox_inches='tight')
    
    # Create pressure-specific plots
    for pressure in sorted(df['Pressure'].unique()):
        pressure_df = df[df['Pressure'] == pressure]
        if len(pressure_df) < 10:  # Skip if too few data points
            continue
            
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
        
        # Plot diameter vs Reynolds number for this pressure
        sns.scatterplot(x='Diameter_um', y='Reynolds_Number', hue='Age', 
                       data=pressure_df, alpha=0.6, s=15, ax=axes[0])
        axes[0].set_xlabel('Diameter (μm)')
        axes[0].set_ylabel('Reynolds Number')
        axes[0].set_title(f'Pressure = {pressure} psi')
        axes[0].set_yscale('log')
        
        # Plot diameter vs shear rate for this pressure
        sns.scatterplot(x='Diameter_um', y='Shear_Rate_s', hue='Age', 
                       data=pressure_df, alpha=0.6, s=15, ax=axes[1])
        axes[1].set_xlabel('Diameter (μm)')
        axes[1].set_ylabel('Shear Rate (s⁻¹)')
        axes[1].set_title(f'Pressure = {pressure} psi')
        axes[1].set_yscale('log')
        
        # Plot diameter vs shear stress for this pressure
        sns.scatterplot(x='Diameter_um', y='Shear_Stress_Pa', hue='Age', 
                       data=pressure_df, alpha=0.6, s=15, ax=axes[2])
        axes[2].set_xlabel('Diameter (μm)')
        axes[2].set_ylabel('Shear Stress (Pa)')
        axes[2].set_title(f'Pressure = {pressure} psi')
        axes[2].set_yscale('log')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'diameter_vs_fluid_dynamics_pressure_{pressure}.png'), 
                   dpi=600, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()
