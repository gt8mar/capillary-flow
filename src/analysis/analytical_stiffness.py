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
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

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
        # print_diagnostic_velocity_profile(participant, participant_df, profiles)
        
        # Drop the 'T' column if it exists and contains only NaN values
        if 'T' in profiles.columns and profiles['T'].isna().all():
            profiles = profiles.drop('T', axis=1)
            
        velocity_profiles_dict[participant] = profiles
    
    return velocity_profiles_dict

def calculate_health_score(participant_df):
    """Calculate health score based on various metrics"""
    score = 1.0  # Start with perfect health
    
    # Blood pressure component (if available)
    if 'DIA_BP' in participant_df.columns and not participant_df['DIA_BP'].isna().all():
        dia_bp = participant_df['DIA_BP'].median()
        # More nuanced BP scoring
        if dia_bp > 90:  # High
            score *= 0.8
        elif dia_bp > 85:  # Borderline high
            score *= 0.9
        elif dia_bp < 60:  # Low
            score *= 0.9
        elif dia_bp < 65:  # Borderline low
            score *= 0.95
    
    # Age component - slight reduction for older age
    if 'Age' in participant_df.columns:
        age = participant_df['Age'].iloc[0]
        if age > 65:
            score *= 0.95
        elif age > 55:
            score *= 0.98
    
    # Disease states
    if 'Hypertension' in participant_df.columns:
        if participant_df['Hypertension'].iloc[0] == True:
            score *= 0.8
    if 'Diabetes' in participant_df.columns:
        if participant_df['Diabetes'].iloc[0] == True:
            score *= 0.8
        elif participant_df['Diabetes'].iloc[0] == 'PRE':
            score *= 0.9
    if 'HeartDisease' in participant_df.columns:
        if participant_df['HeartDisease'].iloc[0] == True:
            score *= 0.7
    
    # BMI component (if height and weight are available)
    if all(col in participant_df.columns for col in ['Height', 'Weight']):
        height_m = participant_df['Height'].iloc[0] / 100  # Convert cm to m
        weight_kg = participant_df['Weight'].iloc[0]
        bmi = weight_kg / (height_m ** 2)
        if bmi > 30:  # Obese
            score *= 0.85
        elif bmi > 25:  # Overweight
            score *= 0.95
        elif bmi < 18.5:  # Underweight
            score *= 0.95
    
    return score

def calculate_participant_stiffness(participant_df, participant_id, velocity_profiles, use_log=False):
    """
    Calculate stiffness parameters for a single participant
    
    Parameters:
    -----------
    participant_df : pandas.DataFrame
        DataFrame containing participant data
    participant_id : str
        Participant identifier
    velocity_profiles : pandas.DataFrame
        DataFrame containing velocity profiles
    use_log : bool, optional
        Whether to use log-transformed velocity values (default: False)
    """
    try:
        # Get the SET value and age without binning
        set_value = participant_df['SET'].iloc[0] if 'SET' in participant_df.columns else None
        participant_age = participant_df['Age'].iloc[0] if 'Age' in participant_df.columns else None
        
        # Calculate health score
        health_score = calculate_health_score(participant_df)
        
        # Calculate hysteresis and total area
        hysteresis, total_area, _, _ = calculate_hysteresis(velocity_profiles)
        
        # Use log-transformed velocity if specified
        velocity_col = 'Log_Video_Median_Velocity' if use_log else 'Video_Median_Velocity'
        
        # Add log-transformed velocity if needed and not already present
        if use_log and 'Log_Video_Median_Velocity' not in participant_df.columns:
            participant_df['Log_Video_Median_Velocity'] = np.log(participant_df['Video_Median_Velocity'])
        
        # Fit linear model with just Pressure
        model = smf.ols(f'{velocity_col} ~ Pressure', data=participant_df).fit()
        
        # Calculate metrics with error checking
        stiffness_index = -model.params['Pressure']
        baseline_velocity = model.params['Intercept']
        compliance = 1 / stiffness_index if abs(stiffness_index) > 1e-10 else np.nan
        pressure_range = participant_df['Pressure'].max() - participant_df['Pressure'].min()
        velocity_range = participant_df[velocity_col].max() - participant_df[velocity_col].min()
        sensitivity = velocity_range / pressure_range if pressure_range > 0 else np.nan
        
        # Handle potential division by zero in R-squared calculation
        centered_tss = model.centered_tss
        r_squared = 1 - model.ssr/centered_tss if abs(centered_tss) > 1e-10 else np.nan
        
        metrics = {
            'Participant': participant_id,
            'SET': set_value,
            'Age': participant_age,
            'Health_Score': health_score,
            'Stiffness_Index': stiffness_index,
            'Compliance': compliance,
            'Baseline_Velocity': baseline_velocity,
            'Pressure_Sensitivity': sensitivity,
            'Model_R_Squared': r_squared,
            'N_Observations': len(participant_df),
            'Velocity_Std': participant_df[velocity_col].std(),
            'Pressure_Range': pressure_range,
            'Hysteresis': hysteresis,
            'Total_Area': total_area,
            'Log_Transform_Used': use_log
        }
        
        return metrics
        
    except Exception as e:
        print(f"\nWarning: Could not calculate stiffness for {participant_id}: {str(e)}")
        return None

def plot_velocity_curves(df, participant_id, results_dir, use_log=False):
    """Plot velocity vs pressure curves for a single participant"""
    # Save to velocity_profiles directory
    output_dir = os.path.join(results_dir, 'analytical', 'velocity_profiles')
    transform_suffix = '_log' if use_log else ''
    output_path = os.path.join(output_dir, f'velocity_profile_{participant_id}{transform_suffix}.png')
    
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
    
    # Use log-transformed velocity if specified
    velocity_col = 'Log_Video_Median_Velocity' if use_log else 'Video_Median_Velocity'
    
    # Add log-transformed velocity if needed
    if use_log and 'Log_Video_Median_Velocity' not in participant_df.columns:
        participant_df['Log_Video_Median_Velocity'] = np.log(participant_df['Video_Median_Velocity'])
    
    velocity_profiles = participant_df.groupby(['Pressure', 'UpDown'])[velocity_col].mean().unstack()
    
    hysteresis, _, up_velocities, down_velocities = calculate_hysteresis(velocity_profiles)
    
    if up_velocities is not None and down_velocities is not None:
        plt.figure(figsize=(5, 3))
        
        plt.plot(velocity_profiles.index, up_velocities, '.-', color='#1f77b4', 
                label='Up', alpha=0.7, markersize=3, linewidth=0.5)
        plt.plot(velocity_profiles.index, down_velocities, '.-', color='#ff7f0e', 
                label='Down', alpha=0.7, markersize=3, linewidth=0.5)
        
        ylabel = 'Log Velocity (log μm/s)' if use_log else 'Velocity (μm/s)'
        plt.title(f'Velocity Profile: {participant_id}\nHysteresis: {hysteresis:.2f}', 
                 fontproperties=source_sans, fontsize=7)
        plt.xlabel('Pressure (psi)', fontproperties=source_sans, fontsize=7)
        plt.ylabel(ylabel, fontproperties=source_sans, fontsize=7)
        plt.legend(prop=source_sans)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    """Load metadata and merge UpDown column"""
    print("\nLoading metadata...")
    
    if platform.system() == 'Windows':
        metadata_folder = os.path.join(cap_flow_path, 'metadata')
    else:
        metadata_folder = '/hpc/projects/capillary-flow/metadata'
    
    # Read all metadata Excel files
    metadata_files = [f for f in os.listdir(metadata_folder) if f.endswith('.xlsx')]
    metadata_dfs = [pd.read_excel(os.path.join(metadata_folder, f)) for f in metadata_files]
    metadata_df = pd.concat(metadata_dfs)
    
    # Print debug info
    print(f"\nFound {len(metadata_files)} metadata files")
    print("Columns in metadata:", metadata_df.columns.tolist())
    print("\nExample BP values from metadata:")
    if 'BP' in metadata_df.columns:
        print(metadata_df[['Participant', 'Video', 'BP']].head())
    else:
        print("Warning: 'BP' column not found in metadata")
    
    return metadata_df

def run_umap_analysis(metrics_df, df, results_dir):
    """Run UMAP analysis on the metrics"""
    # Save to umap directory
    output_dir = os.path.join(results_dir, 'stats', 'umap')
    
    # Separate numeric and non-numeric columns
    numeric_cols = metrics_df.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_cols = metrics_df.select_dtypes(exclude=['float64', 'int64']).columns
    
    # Remove Pressure_Range from feature columns since it's constant
    feature_cols = [col for col in numeric_cols 
                   if col not in ['SET', 'Participant', 'Age', 'Pressure_Range']]
    
    # Print feature info before handling NaN
    print("\nFeatures used in UMAP clustering (before NaN handling):")
    for col in feature_cols:
        print(f"- {col}")
        print(f"  Range: {metrics_df[col].min():.2f} to {metrics_df[col].max():.2f}")
        print(f"  Missing values: {metrics_df[col].isna().sum()}")
    
    # Handle missing values by filling with median
    X = metrics_df[feature_cols].copy()
    for col in feature_cols:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"\nFilled {X[col].isna().sum()} missing values in {col} with median: {median_val:.2f}")
    
    # Run UMAP with different n_neighbors values
    n_neighbors_values = [5, 15, 30]
    
    for n_neighbors in n_neighbors_values:
        print(f"\nRunning UMAP with n_neighbors={n_neighbors}")
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
        embedding = reducer.fit_transform(X.values)
        
        # Add UMAP coordinates to metrics_df
        metrics_df[f'UMAP1_{n_neighbors}'] = embedding[:, 0]
        metrics_df[f'UMAP2_{n_neighbors}'] = embedding[:, 1]
    
    # Save UMAP plots
    plt.savefig(os.path.join(output_dir, 'umap_stability_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'), dpi=300)
    
    return metrics_df  # Return the updated DataFrame

def impute_height_weight(df):
    """
    Impute missing height and weight values based on age and sex averages
    """
    # Create age groups (e.g., 5-year bins)
    df['Age_Group'] = pd.qcut(df['Age'], q=5)
    
    # Calculate average height/weight by age group and sex
    height_means = df.groupby(['Age_Group', 'Sex'])['Height'].transform('mean')
    weight_means = df.groupby(['Age_Group', 'Sex'])['Weight'].transform('mean')
    
    # If still missing (due to missing age or sex), use overall means
    height_overall_mean = df['Height'].mean()
    weight_overall_mean = df['Weight'].mean()
    
    # Impute missing values
    df['Height'] = df['Height'].fillna(height_means).fillna(height_overall_mean)
    df['Weight'] = df['Weight'].fillna(weight_means).fillna(weight_overall_mean)
    
    return df

def analyze_pressure_importance(df, participant_df, results_dir):
    """Analyze which pressures are most informative using PCA"""
    # Save to pca directory
    output_dir = os.path.join(results_dir, 'stats', 'pca')
    
    # Create pressure-velocity matrix
    pressure_matrix = df.pivot_table(
        index='Participant',
        columns='Pressure',
        values='Video_Median_Velocity',
        aggfunc='median'
    )
    
    # Print the pressures being analyzed
    print("\nPressures included in analysis:")
    for pressure in pressure_matrix.columns:
        print(f"- {pressure:.1f} psi")
    
    # Replace inf values with NaN and then fill with column means
    features = pressure_matrix.copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.mean())
    
    # Print summary of data cleaning
    print(f"\nShape of pressure matrix: {features.shape}")
    print(f"Number of participants: {len(features)}")
    print(f"Number of pressure points: {len(features.columns)}")
    
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Run PCA
    pca = PCA()
    pca_result = pca.fit_transform(features_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    print("\nExplained variance ratio per component:")
    for i, var in enumerate(explained_variance):
        print(f"PC{i+1}: {var:.3f}")
    
    # Calculate correlation between PCs and health score
    pc_correlations = []
    for i in range(pca.n_components_):
        correlation = np.corrcoef(pca_result[:, i], 
                                participant_df.loc[features.index, 'Health_Score'])[0, 1]
        pc_correlations.append(abs(correlation))
    
    print("\nCorrelations between PCs and Health Score:")
    for i, corr in enumerate(pc_correlations):
        print(f"PC{i+1}: {corr:.3f}")
    
    # Get pressure importance scores
    pressure_importance = pd.DataFrame(
        np.abs(pca.components_[0:3]) * np.array(pc_correlations[0:3])[:, np.newaxis],
        columns=features.columns
    ).sum(axis=0)
    
    # Plot explained variance
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    plt.title('Explained Variance by Principal Component')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot pressure importance scores
    plt.figure(figsize=(8, 4))
    pressure_importance.sort_values().plot(kind='bar')
    plt.xlabel('Pressure (psi)')
    plt.ylabel('Importance Score')
    plt.title('Pressure Importance Scores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pressure_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot first two PCs colored by health score
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=participant_df.loc[features.index, 'Health_Score'],
                         cmap='viridis')
    plt.colorbar(scatter, label='Health Score')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Results Colored by Health Score')
    
    # Add participant labels
    for idx, row in features.iterrows():
        plt.annotate(idx, 
                    (pca_result[features.index.get_loc(idx), 0],
                     pca_result[features.index.get_loc(idx), 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=6, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_health_score.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return pressure_importance.sort_values(ascending=False)

def analyze_pressure_importance_rf(df, participant_df, output_dir):
    """Analyze which pressures are most informative using Random Forest"""
    print("\nAnalyzing pressure importance using Random Forest...")
    
    # Create pressure-velocity matrix
    pressure_matrix = df.pivot_table(
        index='Participant',
        columns='Pressure',
        values='Video_Median_Velocity',
        aggfunc='median'
    )
    
    # Print the pressures being analyzed
    print("\nPressures included in analysis:")
    for pressure in pressure_matrix.columns:
        print(f"- {pressure:.1f} psi")
    
    # Replace inf values with NaN and then fill with column means
    features = pressure_matrix.copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.mean())
    
    # Get health scores for matching participants and handle missing values
    health_scores = participant_df.loc[features.index, 'Health_Score']
    
    # Print diagnostic information
    print("\nHealth scores shape:", health_scores.shape)
    print("Number of NaN health scores:", health_scores.isna().sum())
    print("Number of NaN feature values:", features.isna().sum().sum())
    
    # Remove any rows with NaN values
    valid_mask = ~health_scores.isna()
    features_clean = features[valid_mask]
    health_scores_clean = health_scores[valid_mask]
    
    print(f"\nAfter removing NaN values:")
    print(f"Features shape: {features_clean.shape}")
    print(f"Health scores shape: {health_scores_clean.shape}")
    
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_clean)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(features_scaled, health_scores_clean)
    
    # Get feature importance scores
    importance_scores = pd.Series(
        rf.feature_importances_,
        index=features_clean.columns,
        name='Importance'
    ).sort_values(ascending=False)
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf, features_scaled, health_scores_clean, cv=5)
    print(f"\nCross-validation R² scores: {cv_scores}")
    print(f"Mean R² score: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
    
    # Plot feature importance with error bars from cross-validation
    importances = []
    for train_idx, test_idx in KFold(n_splits=5, shuffle=True, random_state=42).split(features_scaled):
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(features_scaled[train_idx], health_scores_clean[train_idx])
        importances.append(rf.feature_importances_)
    
    importance_df = pd.DataFrame(importances, columns=features_clean.columns)
    
    plt.figure(figsize=(8, 4))
    plt.errorbar(x=range(len(features_clean.columns)),
                y=importance_df.mean(),
                yerr=importance_df.std() * 2,
                fmt='o', capsize=5)
    plt.xticks(range(len(features_clean.columns)), features_clean.columns, rotation=45)
    plt.xlabel('Pressure (psi)')
    plt.ylabel('Random Forest Importance')
    plt.title('Pressure Importance from Random Forest\n(with 95% confidence intervals)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rf_pressure_importance_with_errors.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compare with PCA importance
    pca_importance = analyze_pressure_importance(df, participant_df, output_dir)
    
    # Create comparison plot
    plt.figure(figsize=(10, 5))
    comparison_df = pd.DataFrame({
        'Random Forest': importance_scores,
        'PCA': pca_importance
    })
    comparison_df.plot(kind='bar')
    plt.xlabel('Pressure (psi)')
    plt.ylabel('Importance Score')
    plt.title('Pressure Importance: Random Forest vs PCA')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pressure_importance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_scores, comparison_df

def audit_health_score_inputs(df):
    """Check and print all entries in columns used for health score calculation"""
    print("\nAuditing health score input data...")
    
    # Detailed DIA_BP analysis
    print("\nDiastolic BP Analysis:")
    missing_dia_bp = df[df['DIA_BP'].isna()]
    print(f"Number of missing DIA_BP values: {len(missing_dia_bp)}")
    print("\nParticipants with missing DIA_BP:")
    print(missing_dia_bp['Participant'].value_counts())
    
    # Hypertension analysis
    print("\nHypertension Analysis:")
    print("Data type:", df['Hypertension'].dtype)
    print("Unique values with types:")
    for val in df['Hypertension'].unique():
        print(f"Value: {val}, Type: {type(val)}")
    
    # Original audit code...
    health_columns = [
        'Participant', 'Hypertension', 'Diabetes', 'HeartDisease',
        'SYS_BP', 'DIA_BP', 'Age', 'Height', 'Weight', 'Sex'
    ]
    
    # Check which columns exist
    missing_cols = [col for col in health_columns if col not in df.columns]
    if missing_cols:
        print("\nWarning: Missing columns:", missing_cols)
    
    existing_cols = [col for col in health_columns if col in df.columns]
    
    # Get unique values for each column
    print("\nUnique values in each column:")
    for col in existing_cols:
        unique_vals = df[col].unique()
        n_unique = len(unique_vals)
        n_missing = df[col].isna().sum()
        print(f"\n{col}:")
        print(f"Number of unique values: {n_unique}")
        print(f"Number of missing values: {n_missing}")
        print("Unique values:", sorted(unique_vals) if n_unique < 20 else f"{n_unique} values (too many to display)")
        
        # For numerical columns, show summary statistics
        if df[col].dtype in ['int64', 'float64']:
            print("Summary statistics:")
            print(df[col].describe())
        
        # For categorical columns, show value counts
        else:
            print("Value counts:")
            print(df[col].value_counts())
    
    # Check for potential data quality issues
    print("\nPotential data quality issues:")
    
    # Check age range
    if 'Age' in df.columns:
        age_min, age_max = df['Age'].min(), df['Age'].max()
        if age_min < 0 or age_max > 120:
            print(f"Warning: Age range ({age_min}, {age_max}) seems unusual")
    
    # Check height range (in cm)
    if 'Height' in df.columns:
        height_min, height_max = df['Height'].min(), df['Height'].max()
        if height_min < 100 or height_max > 250:
            print(f"Warning: Height range ({height_min}, {height_max}) seems unusual")
    
    # Check weight range (in kg)
    if 'Weight' in df.columns:
        weight_min, weight_max = df['Weight'].min(), df['Weight'].max()
        if weight_min < 30 or weight_max > 250:
            print(f"Warning: Weight range ({weight_min}, {weight_max}) seems unusual")
    
    # Check blood pressure ranges
    if 'SYS_BP' in df.columns:
        sys_min, sys_max = df['SYS_BP'].min(), df['SYS_BP'].max()
        if sys_min < 70 or sys_max > 220:
            print(f"Warning: Systolic BP range ({sys_min}, {sys_max}) seems unusual")
    
    if 'DIA_BP' in df.columns:
        dia_min, dia_max = df['DIA_BP'].min(), df['DIA_BP'].max()
        if dia_min < 40 or dia_max > 120:
            print(f"Warning: Diastolic BP range ({dia_min}, {dia_max}) seems unusual")
    
    # Update the boolean column check in audit_health_score_inputs
    bool_columns = ['Hypertension', 'Diabetes', 'HeartDisease']
    for col in bool_columns:
        if col in df.columns:
            if col == 'Diabetes':
                invalid_values = df[~df[col].isin(['True', 'False', 'PRE', np.nan])][col].unique()
            else:
                invalid_values = df[~df[col].isin([True, False, 'Yes', 'No', np.nan])][col].unique()
            if len(invalid_values) > 0:
                print(f"Warning: Invalid values in {col}: {invalid_values}")
    
    print("\nAudit complete.")

def fix_diastolic_bp(df):
    """Extract and fill missing DIA_BP values from BP column"""
    print("\nFixing missing DIA_BP values...")
    
    def extract_diastolic(bp_str):
        if pd.isna(bp_str):
            return np.nan
        try:
            return float(bp_str.split('/')[1].strip())
        except (IndexError, ValueError, AttributeError):
            return np.nan
    
    # Convert DIA_BP column to float64 if it exists
    if 'DIA_BP' in df.columns:
        df['DIA_BP'] = pd.to_numeric(df['DIA_BP'], errors='coerce')
    
    # Extract DIA_BP from BP string
    if 'BP' in df.columns:
        bp_values = df['BP'].apply(extract_diastolic)
        df['DIA_BP'] = df['DIA_BP'].fillna(bp_values)
    
    # Fill remaining missing values with participant medians
    df['DIA_BP'] = df.groupby('Participant')['DIA_BP'].transform(
        lambda x: x.fillna(x.median())
    )
    
    return df

def analyze_bp_extraction(df):
    """Analyze the quality of BP extraction"""
    print("\nAnalyzing BP extraction...")
    
    # Check BP and DIA_BP distributions
    print("\nBP format examples:")
    print(df[['Participant', 'BP', 'DIA_BP']].head(10))
    
    # Analyze DIA_BP distribution
    print("\nDIA_BP Statistics:")
    print(df['DIA_BP'].describe())
    
    # Check for unusual values
    unusual = df[
        (df['DIA_BP'] < 40) | 
        (df['DIA_BP'] > 120)
    ][['Participant', 'BP', 'DIA_BP']]
    
    if not unusual.empty:
        print("\nUnusual DIA_BP values found:")
        print(unusual)
    
    # Group by participant
    participant_bp = df.groupby('Participant').agg({
        'DIA_BP': ['mean', 'std', 'count']
    }).round(2)
    
    print("\nParticipant BP Summary:")
    print(participant_bp)
    
    return participant_bp

def analyze_velocity_profiles(velocity_profiles_dict, output_dir):
    """Analyze the quality of velocity profiles"""
    print("\nAnalyzing velocity profiles...")
    
    # Collect statistics for each profile
    profile_stats = []
    for participant, profiles in velocity_profiles_dict.items():
        stats = {
            'Participant': participant,
            'Num_Points': len(profiles),
            'Pressure_Range': profiles.index.max() - profiles.index.min(),
            'Max_Velocity': profiles.max().max(),
            'Has_Up': 'U' in profiles.columns,
            'Has_Down': 'D' in profiles.columns,
            'Missing_Values': profiles.isna().sum().sum()
        }
        profile_stats.append(stats)
    
    profile_df = pd.DataFrame(profile_stats)
    
    print("\nVelocity Profile Summary:")
    print(profile_df.describe())
    
    # Plot profile quality metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Number of points
    sns.histplot(data=profile_df, x='Num_Points', ax=axes[0,0])
    axes[0,0].set_title('Distribution of Profile Points')
    
    # Pressure range
    sns.histplot(data=profile_df, x='Pressure_Range', ax=axes[0,1])
    axes[0,1].set_title('Distribution of Pressure Ranges')
    
    # Max velocity
    sns.histplot(data=profile_df, x='Max_Velocity', ax=axes[1,0])
    axes[1,0].set_title('Distribution of Max Velocities')
    
    # Missing values
    sns.histplot(data=profile_df, x='Missing_Values', ax=axes[1,1])
    axes[1,1].set_title('Distribution of Missing Values')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_profile_diagnostics.png'))
    plt.close()
    
    return profile_df

def analyze_health_scores(metrics_df, output_dir):
    """Analyze the distribution and relationships in health scores"""
    print("\nAnalyzing health scores...")
    
    # Basic statistics
    print("\nHealth Score Statistics:")
    print(metrics_df['Health_Score'].describe())
    
    # Create correlation matrix for all numeric columns
    numeric_cols = metrics_df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = metrics_df[numeric_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Health Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'health_score_correlations.png'))
    plt.close()
    
    # Plot health score distribution by SET
    if 'SET' in metrics_df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=metrics_df, x='SET', y='Health_Score')
        sns.swarmplot(data=metrics_df, x='SET', y='Health_Score', color='0.25', alpha=0.5)
        plt.title('Health Score Distribution by SET')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'health_score_by_set.png'))
        plt.close()
    
    return corr_matrix

def analyze_clustering_results(metrics_df, output_dir):
    """Analyze the quality of UMAP clustering"""
    print("\nAnalyzing clustering results...")
    
    # Get UMAP columns for different n_neighbors
    umap_cols = [col for col in metrics_df.columns if col.startswith('UMAP')]
    n_neighbors_list = sorted(set([int(col.split('_')[1]) for col in umap_cols if col.endswith('1')]))
    
    # Check if we have any UMAP results to analyze
    if not n_neighbors_list:
        print("Warning: No UMAP results found to analyze")
        return
    
    # Calculate silhouette scores if SET is available
    if 'SET' in metrics_df.columns:
        from sklearn.metrics import silhouette_score
        print("\nSilhouette Scores by n_neighbors:")
        for n in n_neighbors_list:
            umap_data = metrics_df[[f'UMAP1_{n}', f'UMAP2_{n}']].values
            score = silhouette_score(umap_data, metrics_df['SET'])
            print(f"n_neighbors={n}: {score:.3f}")
    
    # Plot stability comparison
    if len(n_neighbors_list) == 1:
        # Handle single n_neighbors case
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            metrics_df[f'UMAP1_{n_neighbors_list[0]}'],
            metrics_df[f'UMAP2_{n_neighbors_list[0]}'],
            c=metrics_df['Health_Score'] if 'Health_Score' in metrics_df else None,
            cmap='viridis'
        )
        plt.title(f'UMAP Projection (n_neighbors={n_neighbors_list[0]})')
        plt.colorbar(scatter, label='Health Score')
        plt.tight_layout()
    elif len(n_neighbors_list) > 1:
        # Multiple n_neighbors case
        fig, axes = plt.subplots(1, len(n_neighbors_list), figsize=(15, 5))
        for i, n in enumerate(n_neighbors_list):
            scatter = axes[i].scatter(
                metrics_df[f'UMAP1_{n}'],
                metrics_df[f'UMAP2_{n}'],
                c=metrics_df['Health_Score'] if 'Health_Score' in metrics_df else None,
                cmap='viridis'
            )
            axes[i].set_title(f'n_neighbors={n}')
            plt.colorbar(scatter, ax=axes[i], label='Health Score')
        plt.tight_layout()
    
    if n_neighbors_list:  # Only save if we created a plot
        plt.savefig(os.path.join(output_dir, 'umap_stability_comparison.png'))
        plt.close()

def compare_velocity_classifications(df, metrics_df_normal, metrics_df_log, output_dir):
    """Compare classification performance between normal and log-transformed velocities"""
    print("\nComparing velocity classification performance...")
    
    # Features to use for classification
    features = ['Stiffness_Index', 'Compliance', 'Baseline_Velocity', 
                'Pressure_Sensitivity', 'Model_R_Squared', 'Hysteresis', 
                'Total_Area', 'Velocity_Std']
    
    # Target variables to predict
    targets = ['SET']  # Add more targets if available, e.g., 'Hypertension', 'Diabetes'
    
    results = []
    
    for target in targets:
        print(f"\nClassification results for {target}:")
        
        # Prepare data for both normal and log versions
        data_versions = {
            'Normal': metrics_df_normal,
            'Log-transformed': metrics_df_log
        }
        
        for version_name, metrics_df in data_versions.items():
            # Skip if target is not in the data
            if target not in metrics_df.columns:
                print(f"Warning: {target} not found in {version_name} data")
                continue
                
            # Prepare features and target
            X = metrics_df[features].copy()
            y = metrics_df[target]
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Initialize classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Perform cross-validation
            cv_scores = cross_val_score(clf, X_scaled, y, cv=5)
            
            print(f"\n{version_name} Velocity Results:")
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV score: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
            
            # Train on full dataset for feature importance
            clf.fit(X_scaled, y)
            
            # Get feature importance
            importance = pd.DataFrame({
                'Feature': features,
                'Importance': clf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nFeature Importance:")
            print(importance)
            
            # Store results
            results.append({
                'Target': target,
                'Version': version_name,
                'Mean_CV_Score': cv_scores.mean(),
                'CV_Score_Std': cv_scores.std(),
                'Feature_Importance': importance
            })
            
            # Plot feature importance
            plt.figure(figsize=(8, 4))
            sns.barplot(data=importance, x='Importance', y='Feature')
            plt.title(f'Feature Importance for {target} Classification\n({version_name} Velocity)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 
                       f'feature_importance_{target}_{version_name.lower()}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Compare versions side by side
        if len(results) >= 2:
            comparison_data = pd.DataFrame([
                {
                    'Version': r['Version'],
                    'Mean_CV_Score': r['Mean_CV_Score'],
                    'CV_Score_Std': r['CV_Score_Std']
                }
                for r in results if r['Target'] == target
            ])
            
            plt.figure(figsize=(6, 4))
            # Modified barplot code to correctly handle error bars
            bars = plt.bar(comparison_data['Version'], 
                          comparison_data['Mean_CV_Score'],
                          yerr=comparison_data['CV_Score_Std']*2,
                          capsize=5)
            
            plt.title(f'Classification Performance Comparison for {target}')
            plt.ylabel('Mean Cross-Validation Score')
            plt.ylim(0, 1)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'classification_comparison_{target}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    return results

def compare_health_classifications(df, metrics_df_normal, metrics_df_log, results_dir):
    """Compare classification performance for health outcomes"""
    print("\nComparing health classification performance...")
    
    # Features to use for classification
    features = ['Stiffness_Index', 'Compliance', 'Baseline_Velocity', 
                'Pressure_Sensitivity', 'Model_R_Squared', 'Hysteresis', 
                'Total_Area', 'Velocity_Std']
    
    # Health-related targets to predict
    targets = ['Hypertension', 'Diabetes', 'HeartDisease']
    
    results = []
    
    for target in targets:
        print(f"\nClassification results for {target}:")
        
        # Prepare data for both normal and log versions
        data_versions = {
            'Normal': metrics_df_normal,
            'Log-transformed': metrics_df_log
        }
        
        target_results = []
        
        for version_name, metrics_df in data_versions.items():
            # Skip if target is not in the data
            if target not in df.columns:
                print(f"Warning: {target} not found in dataset")
                continue
            
            # Get unique participant values for the target
            participant_targets = df.groupby('Participant')[target].first()
            
            # Get target values only for participants in metrics
            y = participant_targets[metrics_df['Participant']].values
            
            # Convert string boolean values if needed
            if y.dtype == object:
                y = pd.Series(y).map({'True': True, 'False': False, 'PRE': None}).values
            
            # Skip if no valid target values
            if pd.isna(y).all():
                print(f"Warning: No valid {target} values found")
                continue
                
            # Prepare features
            X = metrics_df[features].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Drop rows with NaN target values
            mask = ~pd.isna(y)
            X_scaled = X_scaled[mask]
            y = y[mask]
            
            # Skip if not enough samples
            if len(y) < 5:
                print(f"Warning: Not enough valid samples for {target}")
                continue
            
            # Initialize classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            
            try:
                # Perform cross-validation
                cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc')
                
                print(f"\n{version_name} Velocity Results for {target}:")
                print(f"Cross-validation ROC-AUC scores: {cv_scores}")
                print(f"Mean ROC-AUC score: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
                
                # Train on full dataset for feature importance
                clf.fit(X_scaled, y)
                
                # Get feature importance
                importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': clf.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                print("\nFeature Importance:")
                print(importance)
                
                # Store results
                target_results.append({
                    'Target': target,
                    'Version': version_name,
                    'Mean_ROC_AUC': cv_scores.mean(),
                    'ROC_AUC_Std': cv_scores.std(),
                    'Feature_Importance': importance,
                    'N_Samples': len(y)
                })
                
                # Plot feature importance
                plt.figure(figsize=(8, 4))
                sns.barplot(data=importance, x='Importance', y='Feature')
                plt.title(f'Feature Importance for {target} Classification\n({version_name} Velocity)')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'stats', 'classifier', target.lower(), f'health_importance_{target}_{version_name.lower()}.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Warning: Could not perform classification for {target}: {str(e)}")
                continue
        
        # Compare versions side by side if we have results for both
        if len(target_results) >= 2:
            comparison_data = pd.DataFrame([
                {
                    'Version': r['Version'],
                    'ROC_AUC_Score': r['Mean_ROC_AUC'],
                    'Score_Std': r['ROC_AUC_Std']
                }
                for r in target_results
            ])
            
            plt.figure(figsize=(6, 4))
            bars = plt.bar(comparison_data['Version'], 
                          comparison_data['ROC_AUC_Score'],
                          yerr=comparison_data['Score_Std']*2,
                          capsize=5)
            
            plt.title(f'Health Classification Performance for {target}')
            plt.ylabel('Mean ROC-AUC Score')
            plt.ylim(0, 1)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'stats', 'classifier', target.lower(), f'health_classification_{target}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            results.extend(target_results)
    
    return results

def create_output_directories(cap_flow_path):
    """Create all necessary output subdirectories following coding standards.
    
    Args:
        cap_flow_path: Base path to the capillary-flow directory
    """
    results_dir = os.path.join(cap_flow_path, 'results')
    
    # Define directory structure based on coding standards
    directories = {
        'analytical': [
            'velocity_profiles',
            'stiffness_metrics',
        ],
        'stats': {
            'classifier': [
                'diabetes',
                'healthy',
                'heart_disease',
                'hypertension'
            ],
            'pca': [],
            'umap': []
        }
    }
    
    # Create directories
    for main_dir, subdirs in directories.items():
        base_dir = os.path.join(results_dir, main_dir)
        
        if isinstance(subdirs, list):
            # Create simple subdirectories
            for subdir in subdirs:
                full_path = os.path.join(base_dir, subdir)
                os.makedirs(full_path, exist_ok=True)
                # print(f"Created directory: {full_path}")
        elif isinstance(subdirs, dict):
            # Create nested subdirectories
            for subdir, nested_dirs in subdirs.items():
                subdir_path = os.path.join(base_dir, subdir)
                os.makedirs(subdir_path, exist_ok=True)
                # print(f"Created directory: {subdir_path}")
                
                for nested_dir in nested_dirs:
                    nested_path = os.path.join(subdir_path, nested_dir)
                    os.makedirs(nested_path, exist_ok=True)
                    # print(f"Created directory: {nested_path}")

    return results_dir

def main():
    start_total = time.time()
    
    print("\nStarting data processing...")
    start = time.time()
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    results_dir = create_output_directories(cap_flow_path)
    
    df = pd.read_csv(data_filepath)
    
    # Store metrics DataFrames for both versions
    metrics_dfs = {}
    
    # Process data with both regular and log-transformed velocities
    for use_log in [False, True]:
        transform_suffix = '_log' if use_log else ''
        
        # Create analytical subdirectories
        velocity_profiles_dir = os.path.join(results_dir, 'analytical', 'velocity_profiles')
        os.makedirs(velocity_profiles_dir, exist_ok=True)
        
        # Create stats subdirectories
        stats_subdirs = {
            'comparisons': '',
            'umap': '',
            'pca': '',
            'classifier': ['diabetes', 'healthy', 'heart_disease', 'hypertension']
        }
        
        for subdir, nested_dirs in stats_subdirs.items():
            base_dir = os.path.join(results_dir, 'stats', subdir)
            os.makedirs(base_dir, exist_ok=True)
            
            # Create nested directories for classifier
            if nested_dirs:
                for nested_dir in nested_dirs:
                    nested_path = os.path.join(base_dir, nested_dir)
                    os.makedirs(nested_path, exist_ok=True)
        
        # Calculate velocity profiles
        velocity_profiles_dict = calculate_velocity_profiles(df)
        
        # Calculate stiffness parameters for each participant
        stiffness_metrics = []
        plotted_participants = []
        skipped_plots = []
        
        for participant_id, velocity_profiles in velocity_profiles_dict.items():
            participant_df = df[df['Participant'] == participant_id]
            
            try:
                # Calculate stiffness parameters
                metrics = calculate_participant_stiffness(participant_df, participant_id, 
                                                        velocity_profiles, use_log=use_log)
                stiffness_metrics.append(metrics)
                
                # Create velocity curve plots
                try:
                    plot_velocity_curves(df, participant_id, results_dir, use_log=use_log)
                    plotted_participants.append(participant_id)
                except Exception as e:
                    skipped_plots.append((participant_id, str(e)))
                    print(f"\nWarning: Could not plot velocity curves for {participant_id}: {str(e)}")
                    
            except Exception as e:
                print(f"\nError processing participant {participant_id}: {str(e)}")
                continue
        
        # Convert stiffness metrics to DataFrame
        metrics_df = pd.DataFrame(stiffness_metrics)
        
        # Save metrics to CSV with appropriate suffix
        metrics_output_dir = os.path.join(results_dir, 'analytical', 'stiffness_metrics')
        metrics_df.to_csv(os.path.join(metrics_output_dir, 
                         f'stiffness_metrics{transform_suffix}.csv'), index=False)
        
        # Print diagnostic summary
        print(f"\nResults for {'log-transformed' if use_log else 'regular'} velocity:")
        print_diagnostic_summary(df, velocity_profiles_dict, plotted_participants, skipped_plots)
        
        # Create comparison plots
        plot_participant_comparisons(metrics_df, 
                                   os.path.join(results_dir, 'stats', 'comparisons'))
        
        # Run UMAP analysis
        run_umap_analysis(metrics_df, df, results_dir)
        
        # Run diagnostic analyses
        print("\nRunning diagnostic analyses...")
        bp_stats = analyze_bp_extraction(df)
        profile_stats = analyze_velocity_profiles(velocity_profiles_dict, 
                                                os.path.join(results_dir, 'analytical', f'velocity_profiles{transform_suffix}'))
        health_correlations = analyze_health_scores(metrics_df, 
                                                  os.path.join(results_dir, 'stats'))
        analyze_clustering_results(metrics_df, 
                                 os.path.join(results_dir, 'stats'))
        
        # Store metrics DataFrame
        metrics_dfs['log' if use_log else 'normal'] = metrics_df
        
        # Run health classification
        compare_health_classifications(df, metrics_dfs['normal'], metrics_dfs['log'], results_dir)
    
    end = time.time()
    print(f"\nTotal processing time: {end - start_total:.2f} seconds")

if __name__ == "__main__":
    main()