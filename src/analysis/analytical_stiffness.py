import platform
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from matplotlib.font_manager import FontProperties
import time

# Get the hostname of the computer
hostname = platform.node()

# Dictionary mapping hostnames to folder paths and font paths
computer_paths = {
    "LAPTOP-I5KTBOR3": {
        'cap_flow': 'C:\\Users\\gt8ma\\capillary-flow',
        'font': 'C:\\Users\\gt8ma\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf'
    },
    "Quake-Blood": {
        'cap_flow': "C:\\Users\\gt8mar\\capillary-flow",
        'font': 'C:\\Users\\gt8mar\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf'
    },
    "ComputerName3": {
        'cap_flow': "C:\\Users\\ejerison\\capillary-flow",
        'font': 'path/to/font/SourceSans3-Regular.ttf'  # Update this path
    }
}

# Set default paths
default_paths = {
    'cap_flow': "/hpc/projects/capillary-flow",
    'font': 'path/to/font/SourceSans3-Regular.ttf'  # Update this path
}

# Get the paths for the current computer
paths = computer_paths.get(hostname, default_paths)
cap_flow_path = paths['cap_flow']
font_path = paths['font']

def calculate_hysteresis(velocity_profiles):
    """
    Calculate hysteresis and total area from velocity profiles
    
    Parameters:
    -----------
    velocity_profiles : pandas.DataFrame
        DataFrame containing velocity profiles with 'U' and 'D' columns for up/down directions
        
    Returns:
    --------
    tuple : (hysteresis, total_area, up_velocities, down_velocities)
        hysteresis: difference between up and down areas
        total_area: average of up and down areas
        up_velocities: Series containing up velocities
        down_velocities: Series containing down velocities
    """
    # Check if we have necessary directions
    if not all(direction in velocity_profiles.columns for direction in ['U', 'D']):
        return np.nan, np.nan, None, None
    
    # Sort by pressure to ensure correct area calculation
    velocity_profiles = velocity_profiles.sort_index()
    
    # Get the top point (if it exists)
    top_point = None
    if 'T' in velocity_profiles.columns:
        top_values = velocity_profiles['T'].dropna()
        if not top_values.empty:
            top_pressure = top_values.index[0]
            top_velocity = top_values.iloc[0]
            top_point = (top_pressure, top_velocity)
    
    # Get up and down curves
    up_pressures = velocity_profiles.index
    up_velocities = velocity_profiles['U']
    down_pressures = velocity_profiles.index
    down_velocities = velocity_profiles['D']
    
    # Add top point to arrays if it exists
    if top_point:
        # Insert top point at the correct pressure position
        up_idx = up_pressures.get_loc(top_point[0])
        up_velocities.iloc[up_idx] = top_point[1]
        
        down_idx = down_pressures.get_loc(top_point[0])
        down_velocities.iloc[down_idx] = top_point[1]
    
    # Calculate areas under curves using trapezoidal rule
    up_area = np.trapz(up_velocities.values, up_pressures.values)
    down_area = np.trapz(down_velocities.values, down_pressures.values)
    
    # Calculate hysteresis (up - down area)
    hysteresis = up_area - down_area
    
    # Calculate total area (average of up and down)
    total_area = (up_area + down_area) / 2
    
    return hysteresis, total_area, up_velocities, down_velocities

def print_diagnostic_velocity_profile(participant, participant_df, profiles):
    """Print diagnostic information for velocity profiles"""
    # Only show diagnostics for part09-part27
    participant_num = int(participant.replace('part', ''))
    if not (9 <= participant_num <= 27):
        return
        
    print(f"\nParticipant {participant}:")
    print("Sample of raw data:")
    print(participant_df[['Pressure', 'Video_Median_Velocity', 'UpDown']].head())
    print("\nResulting velocity profile:")
    print(profiles)

def print_diagnostic_plotting(participant_id, velocity_profiles, output_path=None):
    """Print diagnostic information for plotting"""
    # Only show diagnostics for part09-part27
    participant_num = int(participant_id.replace('part', ''))
    if not (9 <= participant_num <= 27):
        return
        
    print(f"\nPlotting {participant_id}:")
    print("Velocity profiles shape:", velocity_profiles.shape)
    print("Velocity profiles data:")
    print(velocity_profiles)
    if output_path:
        print(f"Saved plot to: {output_path}")

def print_diagnostic_summary(df, velocity_profiles_dict, plotted_participants, skipped_plots):
    """Print summary diagnostic information"""
    # Filter participants for part09-part27
    early_participants = [p for p in df['Participant'].unique() 
                         if 9 <= int(p.replace('part', '')) <= 27]
    early_profiles = {k: v for k, v in velocity_profiles_dict.items() 
                     if 9 <= int(k.replace('part', '')) <= 27}
    early_plotted = [p for p in plotted_participants 
                    if 9 <= int(p.replace('part', '')) <= 27]
    early_skipped = [(p, r) for p, r in skipped_plots 
                    if 9 <= int(p.replace('part', '')) <= 27]
    
    print("\nEarly participants in dataset (part09-part27):")
    print(sorted(early_participants))
    print(f"Number of early participants: {len(early_participants)}")
    
    print("\nEarly participants with velocity profiles:")
    print(sorted(early_profiles.keys()))
    print(f"Number of early participants with profiles: {len(early_profiles)}")
    
    print("\nSuccessfully plotted early participants:")
    print(sorted(early_plotted))
    print(f"Number of early participants plotted: {len(early_plotted)}")
    
    if early_skipped:
        print("\nEarly participants skipped during plotting:")
        for participant, reason in early_skipped:
            print(f"Participant {participant}: {reason}")

def calculate_velocity_profiles(df):
    """Calculate velocity profiles for all participants"""
    velocity_profiles_dict = {}
    for participant in df['Participant'].unique():
        participant_df = df[df['Participant'] == participant]
        
        # Calculate velocity profiles
        profiles = participant_df.groupby(['Pressure', 'UpDown'])['Video_Median_Velocity'].mean().unstack()
        
        # Uncomment for diagnostics:
        print_diagnostic_velocity_profile(participant, participant_df, profiles)
        
        # Drop the 'T' column if it exists and contains only NaN values
        if 'T' in profiles.columns and profiles['T'].isna().all():
            profiles = profiles.drop('T', axis=1)
            
        velocity_profiles_dict[participant] = profiles
    
    return velocity_profiles_dict

def calculate_participant_stiffness(df, participant_id, velocity_profiles):
    """
    Calculate stiffness parameters for a single participant
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing participant data
    participant_id : str
        Identifier for the participant
    velocity_profiles : pandas.DataFrame
        Pre-calculated velocity profiles with up/down curves
    """
    try:
        # Get participant's age
        participant_age = df['Age'].iloc[0]
        
        # Calculate hysteresis and total area
        hysteresis, total_area, _, _ = calculate_hysteresis(velocity_profiles)
        
        # Fit linear model with just Pressure
        model = smf.ols('Video_Median_Velocity ~ Pressure', data=df).fit()
        
        # Calculate other metrics
        stiffness_index = -model.params['Pressure']
        baseline_velocity = model.params['Intercept']
        compliance = 1 / stiffness_index if abs(stiffness_index) > 1e-10 else float('inf')
        pressure_range = df['Pressure'].max() - df['Pressure'].min()
        velocity_range = df['Video_Median_Velocity'].max() - df['Video_Median_Velocity'].min()
        sensitivity = velocity_range / pressure_range if pressure_range > 0 else np.nan

        # Handle potential division by zero in R-squared calculation
        centered_tss = model.centered_tss
        r_squared = 1 - model.ssr/centered_tss if abs(centered_tss) > 1e-10 else np.nan
        
        return {
            'Participant': participant_id,
            'Age': participant_age,
            'Stiffness_Index': stiffness_index,
            'Compliance': compliance,
            'Baseline_Velocity': baseline_velocity,
            'Pressure_Sensitivity': sensitivity,
            'Model_R_Squared': r_squared,
            'N_Observations': len(df),
            'Velocity_Std': df['Video_Median_Velocity'].std(),
            'Pressure_Range': pressure_range,
            'Hysteresis': hysteresis,
            'Total_Area': total_area
        }
        
    except Exception as e:
        print(f"\nWarning: Could not calculate stiffness for {participant_id}: {str(e)}")
        return {
            'Participant': participant_id,
            'Age': participant_age if 'participant_age' in locals() else np.nan,
            'Stiffness_Index': np.nan,
            'Compliance': np.nan,
            'Baseline_Velocity': np.nan,
            'Pressure_Sensitivity': np.nan,
            'Model_R_Squared': np.nan,
            'N_Observations': len(df),
            'Velocity_Std': df['Video_Median_Velocity'].std(),
            'Pressure_Range': df['Pressure'].max() - df['Pressure'].min(),
            'Hysteresis': np.nan,
            'Total_Area': np.nan
        }

def plot_velocity_curves(df, participant_id, output_dir):
    """Plot velocity vs pressure curves for a single participant"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up style and font
    sns.set_style("whitegrid")
    source_sans = FontProperties(fname=font_path)
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    participant_df = df[df['Participant'] == participant_id]
    velocity_profiles = participant_df.groupby(['Pressure', 'UpDown'])['Video_Median_Velocity'].mean().unstack()
    
    # Uncomment for diagnostics:
    print_diagnostic_plotting(participant_id, velocity_profiles)
    
    hysteresis, _, up_velocities, down_velocities = calculate_hysteresis(velocity_profiles)
    
    if up_velocities is not None and down_velocities is not None:
        plt.figure(figsize=(5, 3))
        
        plt.plot(velocity_profiles.index, up_velocities, '.-', color='#1f77b4', 
                label='Up', alpha=0.7, markersize=3, linewidth=0.5)
        plt.plot(velocity_profiles.index, down_velocities, '.-', color='#ff7f0e', 
                label='Down', alpha=0.7, markersize=3, linewidth=0.5)
        
        plt.title(f'Velocity Profile: {participant_id}\nHysteresis: {hysteresis:.2f}', 
                 fontproperties=source_sans, fontsize=7)
        plt.xlabel('Pressure (psi)', fontproperties=source_sans, fontsize=7)
        plt.ylabel('Velocity (μm/s)', fontproperties=source_sans, fontsize=7)
        plt.legend(prop=source_sans)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'velocity_profile_{participant_id}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        # Uncomment for diagnostics:
        print_diagnostic_plotting(participant_id, velocity_profiles, output_path)
        plt.close()

def plot_participant_comparisons(df, output_dir):
    """
    Create plots comparing stiffness parameters across participants and against age
    """
    # Set up style and font
    sns.set_style("whitegrid")
    source_sans = FontProperties(fname=font_path)
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    # Suppress specific warnings
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Parameters to plot
    params_to_plot = {
        'Stiffness_Index': 'Capillary Stiffness Index',
        'Compliance': 'Capillary Compliance',
        'Baseline_Velocity': 'Baseline Blood Velocity',
        'Pressure_Sensitivity': 'Pressure-Velocity Sensitivity',
        'Model_R_Squared': 'Model Fit (R²)',
        'Hysteresis': 'Hysteresis',
        'Total_Area': 'Total Area Under Curve'
    }
    
    for param, title in params_to_plot.items():
        try:
            # 1. Parameter vs Age scatter plot
            plt.figure(figsize=(5, 3))
            sns.regplot(data=df, x='Age', y=param, scatter=True, 
                       scatter_kws={'alpha':0.5, 's':20},
                       line_kws={'color': '#ff7f0e', 'linewidth': 1})
            
            correlation = df[['Age', param]].corr().iloc[0, 1]
            plt.title(f'{title} vs Age\nr = {correlation:.3f}', 
                     fontproperties=source_sans, fontsize=7)
            plt.xlabel('Age (years)', fontproperties=source_sans, fontsize=7)
            plt.ylabel(title, fontproperties=source_sans, fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param.lower()}_vs_age.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Parameter distribution across participants
            plt.figure(figsize=(5, 3))
            box_props = dict(color='#1f77b4', facecolor='white')
            sns.boxplot(data=df, y=param, boxprops=box_props, showfliers=False)
            sns.stripplot(data=df, y=param, color='#1f77b4', 
                         size=5, alpha=0.5, jitter=0.2)
            
            plt.title(f'Distribution of {title}', fontproperties=source_sans, fontsize=7)
            plt.ylabel(title, fontproperties=source_sans, fontsize=7)
            plt.xticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param.lower()}_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Parameter values by participant (ordered by age)
            plt.figure(figsize=(8, 4))
            participant_order = df.sort_values('Age')['Participant'].values
            sns.barplot(data=df, x='Participant', y=param, 
                       order=participant_order, color='#1f77b4', alpha=0.7)
            plt.xticks(rotation=45, ha='right', fontproperties=source_sans, fontsize=6)
            plt.title(f'{title} by Participant', fontproperties=source_sans, fontsize=7)
            plt.ylabel(title, fontproperties=source_sans, fontsize=7)
            plt.xlabel('Participant', fontproperties=source_sans, fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param.lower()}_by_participant.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"\nError plotting {param}: {str(e)}")
            continue

def load_metadata():
    """Load metadata and create UpDown column"""
    if platform.system() == 'Windows':
        metadata_folder = os.path.join(cap_flow_path, 'metadata')
    else:
        metadata_folder = '/hpc/projects/capillary-flow/metadata'
    
    # Read all metadata Excel files
    metadata_files = [f for f in os.listdir(metadata_folder) if f.endswith('.xlsx')]
    metadata_dfs = [pd.read_excel(os.path.join(metadata_folder, f)) for f in metadata_files]
    metadata_df = pd.concat(metadata_dfs)
    
    # Add 'loc' and leading zero to location column
    metadata_df['Location'] = 'loc' + metadata_df['Location'].astype(str).str.zfill(2)
    
    # Create UpDown column based on pressure sequence
    metadata_df = metadata_df.sort_values(['Participant', 'Location', 'Video'])
    
    # Group by participant and location to find pressure sequences
    for (participant, location), group in metadata_df.groupby(['Participant', 'Location']):
        max_pressure_idx = group['Pressure'].idxmax()
        metadata_df.loc[group.index[:max_pressure_idx], 'UpDown'] = 'U'
        metadata_df.loc[group.index[max_pressure_idx:], 'UpDown'] = 'D'
    
    # Select only necessary columns
    metadata_df = metadata_df[['Participant', 'Video', 'UpDown']]
    
    return metadata_df

def main():
    start_total = time.time()
    
    print("\nStarting data processing...")
    start = time.time()
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_medians.csv')
    output_dir = os.path.join(cap_flow_path, 'results')
    df = pd.read_csv(data_filepath)
    
    # Load metadata and merge UpDown column
    metadata_df = load_metadata()
    df = pd.merge(df, metadata_df, on=['Participant', 'Video'], how='left')
    
    velocity_profiles_dict = calculate_velocity_profiles(df)
    
    participant_metrics = []
    skipped_participants = []
    plotted_participants = []
    skipped_plots = []
    
    for participant in df['Participant'].unique():
        try:
            participant_df = df[df['Participant'] == participant]
            metrics = calculate_participant_stiffness(participant_df, participant, velocity_profiles_dict[participant])
            participant_metrics.append(metrics)
            
            plot_velocity_curves(df, participant, output_dir)
            plotted_participants.append(participant)
        except Exception as e:
            skipped_plots.append((participant, str(e)))
    
    # Uncomment for diagnostics:
    print_diagnostic_summary(df, velocity_profiles_dict, plotted_participants, skipped_plots)
    
    metrics_df = pd.DataFrame(participant_metrics)
    plot_participant_comparisons(metrics_df, output_dir)
    metrics_df.to_csv(os.path.join(output_dir, 'participant_stiffness_metrics.csv'), index=False)
    
    print(f"\nTotal execution time: {time.time() - start_total:.2f} seconds")

if __name__ == "__main__":
    main()