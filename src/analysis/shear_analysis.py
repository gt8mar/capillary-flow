"""
Filename: debug_make_diameter.py
------------------------------------------------------
This script helps debug missing area calculations in the capillary diameter data.
It identifies which capillaries have missing area calculations, checks if the 
segmentation files exist, and compares with entries in merged_csv4.csv.

By: Marcus Forst
"""

# Standard library imports
import os
import platform
from typing import Dict, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# Local imports
from src.tools.parse_filename import parse_filename
from src.tools.plotting_utils import plot_CI

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
    "Clark-":{
        'cap_flow': "C:\\Users\\ejerison\\capillary-flow",
        'downloads': 'C:\\Users\\ejerison\\Downloads'
    }
}
default_paths = {
    'cap_flow': "/hpc/projects/capillary-flow",
    'downloads': "/home/downloads"
}

source_sans = FontProperties(fname='C:\\Users\\ejerison\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')

def main():
    """Main function to debug missing area calculations."""
    # Set paths based on platform
    cap_flow_path = computer_paths[hostname]['cap_flow']
    data_file_path = os.path.join(cap_flow_path, 'results','diameter_analysis_df.csv')


    # Set up font
    try:
        source_sans = FontProperties(fname='C:\\Users\\ejerison\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
    except Exception as e:
        print(f"Warning: Could not set up font: {e}")
        source_sans = None
    
    # Load the data
    diameter_analysis_df = pd.read_csv(data_file_path)

    # calculate the shear rate
    diameter_analysis_df['Shear_Rate'] = 4 * diameter_analysis_df['Corrected_Velocity'] / diameter_analysis_df['Mean_Diameter']
    diameter_analysis_df['set_number'] = diameter_analysis_df['SET'].str.split('set').str[1].replace('_', '').astype(int)
    diameter_analysis_df['Set_affected'] = np.where(diameter_analysis_df['SET'] == 'set01', 'set01', 'set04')

    # calculate median shear rate for each participant video. 
    diameter_analysis_df = calculate_representative_diameters(diameter_analysis_df)


    # Create diameter plots directory if it doesn't exist
    diameter_plots_dir = os.path.join(cap_flow_path, 'results', 'diameter_plots')
    os.makedirs(diameter_plots_dir, exist_ok=True)
    shear_rate_plots_dir = os.path.join(cap_flow_path, 'results', 'shear')
    os.makedirs(shear_rate_plots_dir, exist_ok=True)

    # threshold_analysis(diameter_analysis_df, diameter_plots_dir, cap_flow_path)
    # plot_velocity_vs_diameter_by_age(diameter_analysis_df, diameter_plots_dir)
    # plot_velocity_vs_diameter_by_participant(diameter_analysis_df, diameter_plots_dir)
    # plot_velocity_vs_diameter_by_set(diameter_analysis_df)
    
    # make_mixed_effect_model(diameter_analysis_df)
    # threshold_analysis(diameter_analysis_df)  
    # plot_velocity_vs_diameter_theory()

    # plot_pressure_drop_per_length(diameter_analysis_df)
    
    # Call the new 3D shear rate visualization function
    # plot_shear_rate_3d(diameter_analysis_df)
    plot_shear_rate_3d_pressure_age(diameter_analysis_df)
    plot_pressure_drop_per_length_shear_rate(diameter_analysis_df, pressure = 0.2)
    plot_pressure_drop_per_length_shear_rate(diameter_analysis_df, pressure = 0.4)
    plot_pressure_drop_per_length_shear_rate(diameter_analysis_df, pressure = 1.2)

    plot_CI(diameter_analysis_df, variable = 'Age', method = 'bootstrap', n_iterations = 1000, ci_percentile = 99.5, write = True, dimensionless = False, video_median = False, log_scale = False, old = False, velocity_variable = 'Shear_Rate')
    plot_CI(diameter_analysis_df, variable = 'Diabetes', method = 'bootstrap', n_iterations = 1000, ci_percentile = 99.5, write = True, dimensionless = False, video_median = False, log_scale = False, old = False, velocity_variable = 'Shear_Rate')
    return 0


def threshold_analysis(diameter_analysis_df, diameter_plots_dir, cap_flow_path):
    # Create CDF plots for Mean Diameter split by age groups
    print("\nCreating CDF plots for Mean Diameter by age groups...")
    
    # Ensure we have age data
    if 'Age' not in diameter_analysis_df.columns or diameter_analysis_df['Age'].isna().all():
        print("Error: Age data is missing or all null. Cannot create age-based CDF plots.")
    else:
        # Create a copy of the dataframe for age analysis
        age_df = diameter_analysis_df.dropna(subset=['Mean_Diameter', 'Age']).copy()
        
        # Print age statistics
        print(f"Age range in data: {age_df['Age'].min()} to {age_df['Age'].max()} years")
        print(f"Mean age: {age_df['Age'].mean():.2f} years")
        print(f"Median age: {age_df['Age'].median():.2f} years")
        
        # Test different age thresholds
        age_min = int(np.floor(age_df['Age'].min()))
        age_max = int(np.ceil(age_df['Age'].max()))
        
        # Create a range of thresholds to test
        # If we have a wide age range, test every 5 years
        if age_max - age_min > 20:
            thresholds = list(range(age_min + 5, age_max - 5, 5))
        # Otherwise test every 2 years
        else:
            thresholds = list(range(age_min + 2, age_max - 2, 2))
        
        # Ensure we have at least some thresholds
        if len(thresholds) == 0:
            # If age range is very narrow, just use the median
            thresholds = [int(age_df['Age'].median())]
        
        print(f"Testing age thresholds: {thresholds}")
        
        # Create individual plots for each threshold
        ks_results = {}
        for threshold in thresholds:
            plt.close()
            # Set up style and font
            sns.set_style("whitegrid")
            source_sans = FontProperties(fname='C:\\Users\\gt8mar\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
            
            plt.rcParams.update({
                'pdf.fonttype': 42, 'ps.fonttype': 42,
                'font.size': 7, 'axes.labelsize': 7,
                'xtick.labelsize': 6, 'ytick.labelsize': 6,
                'legend.fontsize': 5, 'lines.linewidth': 0.5
            })
            
            fig, ax = plt.subplots(figsize=(2.4, 2.0))
            ks_stat = create_age_cdf_plot(age_df, threshold, cap_flow_path, ax)
            if ks_stat is not None:
                ks_results[threshold] = ks_stat
            
            plt.tight_layout()
            plt.savefig(os.path.join(diameter_plots_dir, f'age_cdf_threshold_{threshold}.png'), dpi=600, bbox_inches='tight')
            plt.close()
        
        # Find the threshold with the maximum KS statistic (most different distributions)
        if ks_results:
            best_threshold = max(ks_results, key=ks_results.get)
            print(f"\nThreshold with most distinct distributions: {best_threshold} years")
            print(f"KS statistic: {ks_results[best_threshold]:.3f}")
            
        # Create a plot showing KS statistic vs threshold
        if len(ks_results) > 1:
            plt.close()
            # Set up style and font
            sns.set_style("whitegrid")
            source_sans = FontProperties(fname='C:\\Users\\gt8mar\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
            
            plt.rcParams.update({
                'pdf.fonttype': 42, 'ps.fonttype': 42,
                'font.size': 7, 'axes.labelsize': 7,
                'xtick.labelsize': 6, 'ytick.labelsize': 6,
                'legend.fontsize': 5, 'lines.linewidth': 0.5
            })
            
            fig, ax = plt.subplots(figsize=(2.4, 2.0))
            thresholds_list = list(ks_results.keys())
            ks_stats = list(ks_results.values())
            
            ax.plot(thresholds_list, ks_stats, 'o-', linewidth=0.5)
            ax.axvline(x=best_threshold, color='red', linestyle='--', 
                      label=f'Best threshold: {best_threshold} years')
            
            ax.set_xlabel('Age Threshold (years)', fontproperties=source_sans)
            ax.set_ylabel('KS Statistic', fontproperties=source_sans)
            ax.set_title('KS Statistic vs Age Threshold', fontproperties=source_sans)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig('ks_statistic_vs_threshold.png', dpi=600, bbox_inches='tight')
            plt.close()
    return 0

def setup_plotting_style():
    """Set up consistent plotting style according to coding standards."""
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 5,
        'lines.linewidth': 0.5,
        'figure.figsize': (12, 10)
    })

def plot_velocity_vs_diameter_by_age(diameter_analysis_df, diameter_plots_dir):
    """
    For each pressure, plot the Corrected_Velocity vs Mean_Diameter, colored by Age.
    
    Args:
        diameter_analysis_df: DataFrame containing the data to plot
        diameter_plots_dir: Directory to save the plots
    """
    setup_plotting_style()
    for pressure in diameter_analysis_df['Pressure'].unique():
        plt.figure(figsize=(2.4, 2.0))
        scatter = plt.scatter(diameter_analysis_df[diameter_analysis_df['Pressure'] == pressure]['Mean_Diameter'], 
                            diameter_analysis_df[diameter_analysis_df['Pressure'] == pressure]['Corrected_Velocity'],
                            c=diameter_analysis_df[diameter_analysis_df['Pressure'] == pressure]['Age'],
                            alpha=0.5,
                            s=10)
        plt.colorbar(scatter, label='Age')
        plt.xlabel('Mean Diameter (μm)', fontproperties=source_sans)
        plt.ylabel('Corrected Velocity (μm/s)', fontproperties=source_sans)
        plt.title(f'P={pressure} PSI', fontproperties=source_sans)
        plt.savefig(os.path.join(diameter_plots_dir, f'pressure_{pressure}_velocity_vs_diameter_age.png'),
                    dpi=600, bbox_inches='tight')
        plt.close()
def plot_velocity_vs_diameter_by_participant(diameter_analysis_df, diameter_plots_dir):
    """
    Plot velocity vs diameter scatter plots for each participant.
    
    Args:
        diameter_analysis_df: DataFrame containing participant data with Mean_Diameter and Corrected_Velocity
        diameter_plots_dir: Directory path to save the plots
    """
    # for each participant, plot the 'Corrected_Velocity' vs 'Mean_Diameter'
    for participant in diameter_analysis_df['Participant'].unique():
        # Update plot styling
        plt.rcParams.update({
            'pdf.fonttype': 42,
            'ps.fonttype': 42, 
            'font.size': 7,
            'axes.labelsize': 7,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.fontsize': 5,
            'lines.linewidth': 0.5
        })
        plt.figure(figsize=(2.4, 2.0))
        plt.scatter(diameter_analysis_df[diameter_analysis_df['Participant'] == participant]['Mean_Diameter'], 
                   diameter_analysis_df[diameter_analysis_df['Participant'] == participant]['Corrected_Velocity'], 
                   alpha=0.5)
        plt.xlabel('Mean Diameter (μm)', fontproperties=source_sans)
        plt.ylabel('Corrected Velocity (μm/s)', fontproperties=source_sans)
        plt.title(f'{participant} - Velocity vs Diameter', fontproperties=source_sans)
        plt.tight_layout()
        plt.savefig(os.path.join(diameter_plots_dir, f'{participant}_velocity_vs_diameter.png'),
                    dpi=600, bbox_inches='tight')
        plt.close()
def plot_velocity_vs_diameter_by_set(diameter_analysis_df, output_dir):
    """
    Plot velocity vs diameter colored by SET number for each pressure level.
    
    Args:
        diameter_analysis_df: DataFrame containing Mean_Diameter, Corrected_Velocity, Pressure and SET columns
        diameter_plots_dir: Directory path to save the plots
    """
    setup_plotting_style()
    for pressure in diameter_analysis_df['Pressure'].unique():
        plt.figure(figsize=(2.4, 2.0))
        # Convert SET strings to numbers by extracting digits
        set_numbers = diameter_analysis_df[diameter_analysis_df['Pressure'] == pressure]['SET'].str.extract('(\d+)').astype(float)
        scatter = plt.scatter(diameter_analysis_df[diameter_analysis_df['Pressure'] == pressure]['Mean_Diameter'],
                            diameter_analysis_df[diameter_analysis_df['Pressure'] == pressure]['Corrected_Velocity'], 
                            c=set_numbers,
                            alpha=0.5)
        plt.colorbar(scatter, label='SET')
        plt.xlabel('Mean_Diameter', fontproperties=source_sans)
        plt.ylabel('Corrected_Velocity', fontproperties=source_sans)
        plt.title(f'{pressure} - Corrected_Velocity vs Mean_Diameter (colored by SET)', fontproperties=source_sans)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pressure_{pressure}_velocity_vs_diameter_set.png'),
                    dpi=600, bbox_inches='tight')
        plt.close()
        return 0
    
def make_mixed_effect_model(diameter_analysis_df):
    """
    Creates a mixed effects model to analyze the relationship between velocity, diameter, and pressure.

    This function fits several mixed effects models to analyze how capillary diameter and pressure 
    affect blood velocity, while accounting for participant-level random effects. It uses three 
    different modeling approaches:
    1. Basic mixed model with fixed effects for diameter and pressure
    2. Model with interaction between diameter and pressure 
    3. Model with random slope for diameter

    Args:
        diameter_analysis_df: DataFrame containing the following columns:
            - Corrected_Velocity: Blood velocity measurements (um/s)
            - Mean_Diameter: Capillary diameter measurements (um)
            - Participant: Participant IDs for random effects
            - Pressure: Applied pressure values

    Returns:
        None. Prints model summaries to console.

    Raises:
        ImportError: If required statsmodels packages are not available
        ValueError: If required columns are missing from input DataFrame
    """
    # Mixed Effects Model Analysis
    print("\nPerforming Mixed Effects Model Analysis...")
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.regression.mixed_linear_model import MixedLM
        
        # Prepare data for mixed effects model
        model_df = diameter_analysis_df.dropna(subset=['Corrected_Velocity', 'Mean_Diameter', 'Participant', 'Pressure'])
        
        # Ensure Pressure is treated as a numeric variable
        model_df['Pressure'] = pd.to_numeric(model_df['Pressure'], errors='coerce')
        
        # Print basic statistics
        print(f"Number of observations in model: {len(model_df)}")
        print(f"Number of unique participants: {model_df['Participant'].nunique()}")
        print(f"Pressure range: {model_df['Pressure'].min()} to {model_df['Pressure'].max()} (mean: {model_df['Pressure'].mean():.2f})")
        
        # Method 1: Using statsmodels formula API with Pressure as continuous
        print("\nMethod 1: Mixed Effects Model using formula API")
        formula = "Corrected_Velocity ~ Mean_Diameter + Pressure"
        mixed_model = smf.mixedlm(formula, model_df, groups=model_df["Participant"])
        mixed_result = mixed_model.fit()
        print(mixed_result.summary())
        
        # Method 2: Using MixedLM directly with interaction term
        print("\nMethod 2: Mixed Effects Model with interaction term")
        # Create interaction term between Mean_Diameter and Pressure
        model_df['Mean_Diameter_x_Pressure'] = model_df['Mean_Diameter'] * model_df['Pressure']
        
        # Fixed effects: Mean_Diameter, Pressure, and their interaction
        exog_cols = ['Mean_Diameter', 'Pressure', 'Mean_Diameter_x_Pressure']
        
        # Add constant
        exog = sm.add_constant(model_df[exog_cols])
        
        # Random effects: Random intercept for each participant
        groups = model_df['Participant']
        
        # Fit the model
        md = MixedLM(model_df['Corrected_Velocity'], exog, groups)
        mdf = md.fit()
        print(mdf.summary())
        
        # Method 3: Add random slope for Mean_Diameter
        print("\nMethod 3: Mixed Effects Model with random slope for Mean_Diameter")
        # Create design matrices for random effects
        exog_re = model_df[['Mean_Diameter']]
        
        # Fit the model with random intercept and random slope
        md_rs = MixedLM(model_df['Corrected_Velocity'], exog, groups, exog_re=exog_re)
        try:
            mdf_rs = md_rs.fit()
            print(mdf_rs.summary())
        except Exception as e:
            print(f"Could not fit model with random slope: {e}")
            print("This often happens with small datasets or when there's not enough variation within groups.")
        
        # Interpretation
        print("\nInterpretation of Mixed Effects Model Results:")
        print("1. Fixed Effects:")
        print("   - The coefficient for Mean_Diameter represents the effect of diameter on velocity when Pressure is zero")
        print("   - The coefficient for Pressure shows how velocity changes with each unit increase in pressure when diameter is zero")
        print("   - The interaction term (Mean_Diameter_x_Pressure) shows how the relationship between diameter and velocity changes with pressure")
        print("2. Random Effects:")
        print("   - The variance of the random intercept shows how much baseline velocity varies between participants")
        print("   - If included, the variance of the random slope shows how much the diameter-velocity relationship varies between participants")
        print("   - The residual variance shows how much velocity varies within participants after accounting for the model")
        
        # Visualize the model predictions
        setup_plotting_style()
        plt.figure(figsize=(2.4, 2.0))
        
        # Create a grid of pressure values for prediction
        unique_pressures = sorted(model_df['Pressure'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_pressures)))
        
        # For each pressure, plot the actual data and the model prediction
        for i, pressure in enumerate(unique_pressures):
            pressure_data = model_df[model_df['Pressure'] == pressure]
            
            # Plot actual data
            plt.scatter(pressure_data['Mean_Diameter'], pressure_data['Corrected_Velocity'], 
                        alpha=0.5, color=colors[i], label=f'Pressure={pressure} (actual)')
            
            # Sort by Mean_Diameter for smooth line
            pressure_data = pressure_data.sort_values('Mean_Diameter')
            
            # Get model predictions
            pred_y = (mdf.params['const'] + 
                     mdf.params['Mean_Diameter'] * pressure_data['Mean_Diameter'] + 
                     mdf.params['Pressure'] * pressure + 
                     mdf.params['Mean_Diameter_x_Pressure'] * pressure_data['Mean_Diameter'] * pressure)
            
            # Plot prediction line
            plt.plot(pressure_data['Mean_Diameter'], pred_y, '-', color=colors[i], 
                     linewidth=2, label=f'Pressure={pressure} (predicted)')
        
        plt.xlabel('Mean Diameter', fontproperties=source_sans)
        plt.ylabel('Corrected Velocity', fontproperties=source_sans)
        plt.title('Mixed Effects Model: Diameter vs Velocity by Pressure', fontproperties=source_sans)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        # plt.show()
        plt.close() 
        
        # Create a 3D visualization to better show the relationship
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the actual data points
            scatter = ax.scatter(model_df['Mean_Diameter'], 
                                model_df['Pressure'], 
                                model_df['Corrected_Velocity'],
                                c=model_df['Pressure'],
                                cmap='viridis',
                                alpha=0.6)
            
            # Create a mesh grid for the prediction surface
            x_range = np.linspace(model_df['Mean_Diameter'].min(), model_df['Mean_Diameter'].max(), 20)
            y_range = np.linspace(model_df['Pressure'].min(), model_df['Pressure'].max(), 20)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros(X.shape)
            
            # Calculate predicted values for the mesh grid
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = (mdf.params['const'] + 
                              mdf.params['Mean_Diameter'] * X[i, j] + 
                              mdf.params['Pressure'] * Y[i, j] + 
                              mdf.params['Mean_Diameter_x_Pressure'] * X[i, j] * Y[i, j])
            
            # Plot the prediction surface
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, linewidth=0)
            
            # Add labels and colorbar
            ax.set_xlabel('Mean Diameter (μm)', fontproperties=source_sans)
            ax.set_ylabel('Pressure (PSI)', fontproperties=source_sans)
            ax.set_zlabel('Corrected Velocity (μm/s)', fontproperties=source_sans)
            ax.set_title('3D Visualization of Mixed Effects Model', fontproperties=source_sans)
            fig.colorbar(scatter, ax=ax, label='Pressure')
            
            plt.tight_layout()
            plt.show()
            plt.close()
        except Exception as e:
            print(f"Could not create 3D visualization: {e}")
        
    except ImportError:
        print("Error: statsmodels package is required for mixed effects modeling.")
        print("Install it using: pip install statsmodels")
    except Exception as e:
        print(f"Error in mixed effects modeling: {e}")

# Function to create CDF plot for a specific age threshold
def create_age_cdf_plot(age_df, age_threshold, cap_flow_path, ax=None):
    """
    Create a CDF plot for Mean Diameter split by age groups based on the given threshold.
    
    Args:
        age_df: DataFrame containing Age and Mean_Diameter columns
        age_threshold: Age value to use as threshold for grouping
        ax: Matplotlib axis to plot on (optional)
        
    Returns:
        KS statistic if two groups are created, None otherwise
    """
    plt.close()
    setup_plotting_style()

    diameter_plots_dir = os.path.join(cap_flow_path, 'results', 'diameter_plots')

    # Create age groups
    age_df['Age_Group'] = np.where(age_df['Age'] <= age_threshold, 
                                    f'≤{age_threshold} years', 
                                    f'>{age_threshold} years')
    
    # Count samples in each group
    group_counts = age_df['Age_Group'].value_counts()
    
    # Calculate CDFs for each group
    groups = []
    for group_name, group_data in age_df.groupby('Age_Group'):
        # Sort the data
        sorted_data = np.sort(group_data['Mean_Diameter'])
        # Calculate the CDF values (0 to 1)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        groups.append((group_name, sorted_data, cdf, len(sorted_data)))
    
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.0))
    
    # Plot each group
    for group_name, sorted_data, cdf, count in groups:
        ax.plot(sorted_data, cdf, '-', linewidth=2, 
                label=f'{group_name} (n={count})')
    
    # Add reference line
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Calculate median for each group for annotation
    medians = age_df.groupby('Age_Group')['Mean_Diameter'].median()
    
    # Add median lines and annotations
    colors = ['C0', 'C1']  # Default matplotlib colors
    for i, (group, median) in enumerate(medians.items()):
        ax.axvline(x=median, color=colors[i], linestyle=':', alpha=0.7)
        ax.text(median, 0.52, f'Median: {median:.2f}', 
                color=colors[i], ha='center', va='bottom')
    
    # Calculate Kolmogorov-Smirnov statistic
    ks_stat = None
    if len(groups) == 2:
        from scipy import stats
        group1_data = age_df[age_df['Age_Group'] == groups[0][0]]['Mean_Diameter']
        group2_data = age_df[age_df['Age_Group'] == groups[1][0]]['Mean_Diameter']
        ks_stat, p_value = stats.ks_2samp(group1_data, group2_data)
        
        # Add KS test results to the plot
        ax.text(0.05, 0.05, 
                f'KS test: D={ks_stat:.3f}, p={p_value:.4f}',
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add labels and title
    ax.set_xlabel('Mean Diameter', fontproperties=source_sans)
    ax.set_ylabel('Cumulative Probability', fontproperties=source_sans)
    ax.set_title(f'CDF of Mean Diameter by Age Group (Threshold: {age_threshold} years)', fontproperties=source_sans)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(diameter_plots_dir, f'age_threshold_{age_threshold}_cdf_plot.png'),
                dpi=600, bbox_inches='tight')
    plt.close()
    
    return ks_stat

def secomb_viscocity_fn_vitro(diameter, H_discharge = 0.45, constant = 0.1):
    """
    Calculate the secomb viscosity based on the given parameters.
    
    Args:
        diameter: float, diameter of the capillary in um
        hematocrit: float, hematocrit of the capillary in percentage
        
    Returns:
        float, secomb viscosity in cp
    """
    visc_star = 6 * np.exp(-0.085*diameter) + 3.2 - 2.44 * np.exp(-0.06*(diameter)**0.645)
    denom = (1+((10**(-11))*diameter**(12)))
    average_viscosity = 2 # cp
    constant = (0.8 + np.exp(-0.075*diameter)) * (-1+((1)/denom)) + ((1)/denom)
    scaler = ((diameter)/(diameter-1.1))**2
    # viscosity = average_viscosity*(1 + (visc_star -1)*((((1-H_discharge)**constant)-1)/(((1-0.45)**constant)-1))*scaler)*scaler
    viscosity_45 = 220 * np.exp(-1.3*diameter) + 3.2 - 2.44 * np.exp(-0.06*(diameter)**0.645)
    viscosity = 1 + (viscosity_45 - 1)*(((1-H_discharge)**constant)-1)/(((1-0.45)**constant)-1)
    return viscosity # in cp

def secomb_viscocity_fn(diameter, H_discharge = 0.45, constant = 0.1):
    """
    Calculate the secomb viscosity based on the given parameters.
    
    Args:
        diameter: float, diameter of the capillary in um
        hematocrit: float, hematocrit of the capillary in percentage
        
    Returns:
        float, secomb viscosity in cp
    """
    visc_star = 6 * np.exp(-0.085*diameter) + 3.2 - 2.44 * np.exp(-0.06*(diameter)**0.645)
    denom = (1+((10**(-11))*diameter**(12)))
    average_viscosity = 1 # cp
    constant = (0.8 + np.exp(-0.075*diameter)) * (-1+((1)/denom)) + ((1)/denom)
    scaler = ((diameter)/(diameter-1.1))**2
    viscosity = average_viscosity*(1 + (visc_star -1)*((((1-H_discharge)**constant)-1)/(((1-0.45)**constant)-1))*scaler)*scaler
    return viscosity # in cp

def viscosity_to_velocity(viscosity, diameter, MAP = 93.0):
    """
    Calculate the velocity of the capillary based on the given parameters.

    Args:
        viscosity: float, viscosity of the capillary in Pascals*seconds
        diameter: float, diameter of the capillary in um
        pressure_drop: float, pressure drop of the capillary in mmHg
        
    Returns:
        float, velocity of the capillary in um/s
    """
    # Scale capillary pressure: baseline 20 mmHg at MAP of 93 mmHg
    # For every 10 mmHg change in MAP, capillary pressure changes by ~2 mmHg
    baseline_map = 93.0 # mmHg
    pressure_drop = 20.0 + 0.2 * (MAP - baseline_map) # mmHg
    pressure_drop_pascals = pressure_drop * 133.322 # Pa
    viscosity_pascals_s = viscosity * 1e-3 # Pa*s
    diameter_m = diameter * 1e-6 # m
    velocity_m = ((diameter_m/2)**2)/(8 * viscosity_pascals_s) * (pressure_drop_pascals/100)*(10**6) # m/s
    velocity_um_s = velocity_m * 1e6 # um/s
    return velocity_um_s

def pressure_drop_per_length(diameter, velocity, viscosity):
    """
    Calculate the pressure drop of the capillary based on the given parameters.

    Args:
        diameter: float, diameter of the capillary in um
        velocity: float, velocity of the capillary in um/s
        viscosity: float, viscosity of the capillary in cp
        
    Returns:
        float, pressure drop of the capillary in mmHg/um
    """
    viscosity_pascals_s = viscosity * 1e-3 # Pa*s
    diameter_m = diameter * 1e-6 # m
    velocity_m = velocity * 1e-6 # m/s
    pressure_drop_per_length = (8 * viscosity_pascals_s * velocity_m) / ((diameter_m/2)**2) # Pa/m
    pressure_drop_per_length_mmHg = pressure_drop_per_length * 760 / 101325 # mmHg/m
    pressure_drop_per_length_mmHg_um = pressure_drop_per_length_mmHg * 1e6 # mmHg/um
    return pressure_drop_per_length_mmHg_um

def plot_velocity_vs_diameter_theory():
    """
    Plot the velocity vs diameter based on the theory of secomb viscosity.
    """
    setup_plotting_style()

    diameters = np.linspace(1.5, 1000, 1000) # um
    viscosities = secomb_viscocity_fn(diameters) # cp
    plt.plot(diameters, viscosities)
    # plt.scatter(fig_csv[:, 0], fig_csv[:, 1], color='red')
    plt.xlabel('Diameter (um)', fontproperties=source_sans)
    # make the x axis on a log scale
    plt.xscale('log')
    plt.ylabel('Viscosity (cp)', fontproperties=source_sans)
    plt.ylim(1, 7)
    plt.title('Viscosity vs Diameter based on Secomb Viscosity', fontproperties=source_sans)
    plt.tight_layout()
    plt.savefig(os.path.join(computer_paths[hostname]['cap_flow'], 'results', 'diameter_plots', 'velocity_vs_diameter_theory.png'), dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()



    velocities = viscosity_to_velocity(viscosities, diameters) # um/s
    plt.plot(diameters, velocities)
    plt.xlabel('Diameter (um)', fontproperties=source_sans)
    plt.xlim(1, 60)
    plt.ylabel('Velocity (um/s)', fontproperties=source_sans)
    plt.ylim(0, 7000)
    plt.title('Velocity vs Diameter based on Secomb Viscosity', fontproperties=source_sans)
    plt.tight_layout()
    plt.savefig(os.path.join(computer_paths[hostname]['cap_flow'], 'results', 'diameter_plots', 'velocity_vs_diameter_theory.png'), dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_pressure_drop_per_length_shear_rate(diameter_analysis_df, color_by = 'set_number', pressure = 0.2):
    """
    Plot the pressure drop per length vs shear rate.

    Args:
        diameter_analysis_df: DataFrame containing the following columns:
            - Mean_Diameter: Capillary diameter measurements (um)
            - Corrected_Velocity: Blood velocity measurements (um/s)
            - Shear_Rate: Shear rate measurements (s⁻¹)
            - set_number: Set number measurements
            - Pressure: Applied pressure values

    Returns:
        None. Displays the plot.
    """
    setup_plotting_style()

    
    diameter_analysis_df = diameter_analysis_df[diameter_analysis_df['Pressure'] == pressure]
    diameters = diameter_analysis_df['Mean_Diameter']
    viscosities = secomb_viscocity_fn(diameters)
    velocities = diameter_analysis_df['Corrected_Velocity']
    shear_rates = diameter_analysis_df['Shear_Rate']
    ages = diameter_analysis_df['Age']
    pressure_drop_per_lengths = pressure_drop_per_length(diameters, velocities, viscosities)
    set_numbers = diameter_analysis_df['set_number']

    plt.scatter(shear_rates, pressure_drop_per_lengths, c=diameter_analysis_df['set_number'], cmap='viridis')
    plt.xlabel('Shear Rate (s⁻¹)', fontproperties=source_sans)
    plt.ylabel('Pressure Drop per Length (mmHg/um)', fontproperties=source_sans)
    plt.title(f'Pressure Drop per Length vs Shear Rate at {pressure} psi', fontproperties=source_sans)
    plt.colorbar(label='Set Number')
    plt.tight_layout()
    plt.savefig(os.path.join(computer_paths[hostname]['cap_flow'], 'results', 'shear', f'pressure_drop_per_length_shear_rate_{pressure}.png'), dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()
    return 0

def plot_pressure_drop_per_length(diameter_analysis_df):
    """
    Plot the pressure drop per length of the capillary based on the given parameters.
    """
    diameter_analysis_df = diameter_analysis_df[diameter_analysis_df['Pressure'] == 0.2]
    diameters = diameter_analysis_df['Mean_Diameter']
    velocities = diameter_analysis_df['Corrected_Velocity']
    viscosities = secomb_viscocity_fn(diameters)
    pressures = diameter_analysis_df['Pressure']
    pressure_drop_per_lengths = pressure_drop_per_length(diameters, velocities, viscosities)

    # now calculate what the pressure drop per length would be if the velocity was the average velocity, the 25th percentile velocity, and the 75th percentile velocity
    average_velocity = np.mean(velocities)
    diameter_range = np.linspace(1.5, np.max(diameters), 1000)
    viscosities_range = secomb_viscocity_fn(diameter_range)
    pressure_drop_per_length_average_velocity = pressure_drop_per_length(diameter_range, np.ones_like(diameter_range) * average_velocity, viscosities_range)
    pressure_drop_per_length_25th_percentile_velocity = pressure_drop_per_length(diameter_range, np.ones_like(diameter_range) * np.percentile(velocities, 25), viscosities_range)
    pressure_drop_per_length_75th_percentile_velocity = pressure_drop_per_length(diameter_range, np.ones_like(diameter_range) * np.percentile(velocities, 75), viscosities_range)

    plt.scatter(diameters, pressure_drop_per_lengths, c=velocities, cmap='magma')
    plt.scatter(diameter_range, pressure_drop_per_length_average_velocity, color='red')
    plt.scatter(diameter_range, pressure_drop_per_length_25th_percentile_velocity, color='blue')
    plt.scatter(diameter_range, pressure_drop_per_length_75th_percentile_velocity, color='green')
    plt.xlabel('Diameter (um)')
    plt.ylabel('Pressure Drop per Length (mmHg/um)')
    plt.ylim(0, np.max(pressure_drop_per_lengths))
    plt.title('Pressure Drop per Length vs Diameter')
    plt.legend()
    plt.show()

    # plot the pressure drop per length vs the velocity
    plt.scatter(velocities, pressure_drop_per_lengths)
    plt.xlabel('Velocity (um/s)')
    plt.ylabel('Pressure Drop per Length (mmHg/um)')
    plt.title('Pressure Drop per Length vs Velocity')
    plt.show()


def plot_shear_rate_3d(diameter_analysis_df: pd.DataFrame) -> int:
    """Creates a 3D visualization of shear rate vs diameter and pressure.
    
    Generates a 3D scatter plot showing the relationship between capillary diameter,
    pressure, and shear rate. Also creates a prediction surface based on a mixed
    effects model if statsmodels is available.
    
    Args:
        diameter_analysis_df: DataFrame containing Mean_Diameter, Pressure, and Shear_Rate columns
        
    Returns:
        0 if successful, 1 if error occurred
    """
    print("\nCreating 3D visualization of shear rate...")
    
    try:
        # Ensure we have the required columns
        required_cols = ['Mean_Diameter', 'Pressure', 'Shear_Rate', 'Participant']
        if not all(col in diameter_analysis_df.columns for col in required_cols):
            print("Error: Required columns missing from dataframe.")
            print(f"Required: {required_cols}")
            print(f"Available: {diameter_analysis_df.columns.tolist()}")
            return 1
        
        # Prepare data for visualization
        model_df = diameter_analysis_df.dropna(subset=required_cols)
        
        # Ensure Pressure is treated as a numeric variable
        model_df['Pressure'] = pd.to_numeric(model_df['Pressure'], errors='coerce')
        
        # Print basic statistics
        print(f"Number of observations: {len(model_df)}")
        print(f"Number of unique participants: {model_df['Participant'].nunique()}")
        print(f"Pressure range: {model_df['Pressure'].min()} to {model_df['Pressure'].max()} (mean: {model_df['Pressure'].mean():.2f})")
        print(f"Shear rate range: {model_df['Shear_Rate'].min():.2f} to {model_df['Shear_Rate'].max():.2f} (mean: {model_df['Shear_Rate'].mean():.2f})")
        
        # Set up plot styling according to coding standards
        plt.rcParams.update({
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'font.size': 7,
            'axes.labelsize': 7,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.fontsize': 5,
            'lines.linewidth': 0.5
        })
        
        
        
        # Create the 3D visualization
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create figure with appropriate size per coding standards
        fig = plt.figure(figsize=(2.4, 2.0))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the actual data points
        scatter = ax.scatter(model_df['Mean_Diameter'], 
                            model_df['Pressure'], 
                            model_df['Shear_Rate'],
                            c=model_df['Pressure'],
                            cmap='viridis',
                            alpha=0.6,
                            s=5)  # Smaller point size per coding standards
        
        
        # Add labels and colorbar
        ax.set_xlabel('Mean Diameter (μm)')
        ax.set_ylabel('Pressure (PSI)')
        ax.set_zlabel('Shear Rate (s⁻¹)')
        ax.set_title('3D Visualization of Shear Rate')
        fig.colorbar(scatter, ax=ax, label='Pressure')
        
        plt.tight_layout()
        plt.savefig(os.path.join(computer_paths[hostname]['cap_flow'], 'results', 'shear', 'shear_rate_3d.png'), dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
        
        return 0
        
    except Exception as e:
        print(f"Error creating 3D shear rate visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
def plot_shear_rate_3d_pressure_age(diameter_analysis_df: pd.DataFrame) -> int:
    """Creates a 3D visualization of shear rate vs diameter and pressure.
    
    Generates a 3D scatter plot showing the relationship between capillary diameter,
    pressure, and shear rate. Also creates a prediction surface based on a mixed
    effects model if statsmodels is available.
    
    Args:
        diameter_analysis_df: DataFrame containing Mean_Diameter, Pressure, and Shear_Rate columns
        
    Returns:
        0 if successful, 1 if error occurred
    """
    print("\nCreating 3D visualization of shear rate...")

    # select only the pressure of 0.2
    diameter_analysis_df = diameter_analysis_df[diameter_analysis_df['Pressure'] == 1.2]

    try:
        # Ensure we have the required columns
        required_cols = ['Mean_Diameter', 'Age', 'Shear_Rate', 'Participant']
        if not all(col in diameter_analysis_df.columns for col in required_cols):
            print("Error: Required columns missing from dataframe.")
            print(f"Required: {required_cols}")
            print(f"Available: {diameter_analysis_df.columns.tolist()}")
            return 1
        
        # Prepare data for visualization
        model_df = diameter_analysis_df.dropna(subset=required_cols)
        
        # Ensure Pressure is treated as a numeric variable
        model_df['Age'] = pd.to_numeric(model_df['Age'], errors='coerce')
        
        # Print basic statistics
        print(f"Number of observations: {len(model_df)}")
        print(f"Number of unique participants: {model_df['Participant'].nunique()}")
        print(f"Age range: {model_df['Age'].min()} to {model_df['Age'].max()} (mean: {model_df['Age'].mean():.2f})")
        print(f"Shear rate range: {model_df['Shear_Rate'].min():.2f} to {model_df['Shear_Rate'].max():.2f} (mean: {model_df['Shear_Rate'].mean():.2f})")
        
        # Set up plot styling according to coding standards
        plt.rcParams.update({
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'font.size': 7,
            'axes.labelsize': 7,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.fontsize': 5,
            'lines.linewidth': 0.5
        })
        
        
        
        # Create the 3D visualization
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create figure with appropriate size per coding standards
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the actual data points
        scatter = ax.scatter(model_df['Mean_Diameter'], 
                            model_df['Age'], 
                            model_df['Shear_Rate'],
                            c=model_df['set_number'],
                            cmap='viridis',
                            alpha=0.6,
                            s=5)  # Smaller point size per coding standards
        
        
        # Add labels and colorbar
        ax.set_xlabel('Mean Diameter (μm)')
        ax.set_ylabel('Age (years)')
        ax.set_zlabel('Shear Rate (s⁻¹)')
        ax.set_title('3D Visualization of Shear Rate')
        fig.colorbar(scatter, ax=ax, label='Set Number')
        
        plt.tight_layout()
        plt.savefig(os.path.join(computer_paths[hostname]['cap_flow'], 'results', 'shear', 'shear_rate_3d_pressure_age.png'), dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
        
        return 0
        
    except Exception as e:
        print(f"Error creating 3D shear rate visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1

def calculate_representative_diameters(df):
    """
    Calculate representative diameters and median shear rates for participant/location groups.
    
    For each capillary name in a location, finds capillaries with the highest area and 
    longest centerlines to determine a representative diameter. Then calculates new shear 
    rates based on these representative diameters and computes median shear rates per video.
    
    Args:
        df: DataFrame containing capillary data with columns for Participant, Location, 
            Capillary_Name, Area, Centerline_Length, Mean_Diameter, and Corrected_Velocity
    
    Returns:
        DataFrame with added columns for Representative_Diameter and Representative_Shear_Rate
    """
    print("\nCalculating representative diameters and median shear rates...")
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Ensure required columns exist
    required_cols = ['Participant', 'Location', 'Capillary_Name', 'Area', 'Centerline_Length', 'Mean_Diameter', 'Corrected_Velocity']
    missing_cols = [col for col in required_cols if col not in df_copy.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        # Extract location and capillary name from filename if needed
        if 'Filename' in df_copy.columns and ('Location' in missing_cols or 'Capillary_Name' in missing_cols):
            print("Attempting to extract Location and Capillary_Name from Filename...")
            try:
                # Parse filename to extract location and capillary name
                filename_info = df_copy['Filename'].apply(parse_filename)
                if 'Location' in missing_cols:
                    df_copy['Location'] = filename_info.apply(lambda x: x.get('location', 'unknown'))
                if 'Capillary_Name' in missing_cols:
                    df_copy['Capillary_Name'] = filename_info.apply(lambda x: x.get('capillary', 'unknown'))
            except Exception as e:
                print(f"Error extracting information from filenames: {e}")
                return df
    
    # Group by participant, location, and capillary name
    representative_diameters = {}
    comparison_results = []
    
    # Process each participant-location group
    for (participant, location), group in df_copy.groupby(['Participant', 'Location']):
        print(f"Processing {participant}, location {location}...")
        
        # Process each capillary name within this location
        for capillary_name, cap_group in group.groupby('Capillary'):
            if len(cap_group) == 0:
                continue
                
            # Find the maximum area for this capillary name
            max_area = cap_group['Area'].max()
            
            # Filter out capillaries with area less than half the maximum
            valid_caps = cap_group[cap_group['Area'] >= max_area * 0.5]
            
            if len(valid_caps) == 0:
                continue
                
            # Find the representative diameter (median of Mean_Diameter for valid capillaries)
            rep_diameter = valid_caps['Mean_Diameter'].median()
            
            # Store the representative diameter
            key = (participant, location, capillary_name)
            representative_diameters[key] = rep_diameter
            
            # Compare with the median of all capillaries with thi s name
            all_median = cap_group['Mean_Diameter'].median()
            comparison_results.append({
                'Participant': participant,
                'Location': location,
                'Capillary': capillary_name,
                'Representative_Diameter': rep_diameter,
                'All_Median_Diameter': all_median,
                'Difference': rep_diameter - all_median,
                'Percent_Difference': (rep_diameter - all_median) / all_median * 100 if all_median != 0 else float('inf')
            })
    
    # Print comparison results
    comparison_df = pd.DataFrame(comparison_results)
    print("\nComparison of Representative Diameter vs Median Diameter:")
    print(comparison_df.describe())
    
    # Add representative diameter to the original dataframe
    df_copy['Representative_Diameter'] = df_copy.apply(
        lambda row: representative_diameters.get(
            (row['Participant'], row['Location'], row['Capillary']), 
            row['Mean_Diameter']  # Use original if no representative diameter
        ), 
        axis=1
    )
    
    # Calculate new shear rate using representative diameter
    df_copy['Representative_Shear_Rate'] = 4 * df_copy['Corrected_Velocity'] / df_copy['Representative_Diameter']
    
    # Calculate median shear rates for each video
    video_medians = []
    for (participant, location, video), video_group in df_copy.groupby(['Participant', 'Location', 'Video']):
        normal_median = video_group['Shear_Rate'].median()
        rep_median = video_group['Representative_Shear_Rate'].median()
        
        video_medians.append({
            'Participant': participant,
            'Location': location,
            'Video': video,
            'Median_Shear_Rate': normal_median,
            'Median_Representative_Shear_Rate': rep_median,
            'Difference': rep_median - normal_median,
            'Percent_Difference': (rep_median - normal_median) / normal_median * 100 if normal_median != 0 else float('inf')
        })
    
    # Create a dataframe with video median results
    video_medians_df = pd.DataFrame(video_medians)
    print("\nVideo Median Shear Rates:")
    print(video_medians_df.describe())
    
    # Add the video median values back to the original dataframe
    for idx, row in video_medians_df.iterrows():
        mask = ((df_copy['Participant'] == row['Participant']) & 
                (df_copy['Location'] == row['Location']) & 
                (df_copy['Video'] == row['Video']))
        
        df_copy.loc[mask, 'Video_Median_Shear_Rate'] = row['Median_Shear_Rate']
        df_copy.loc[mask, 'Video_Median_Representative_Shear_Rate'] = row['Median_Representative_Shear_Rate']
    
    # Save the results to a CSV file
    results_path = os.path.join(computer_paths.get(hostname, default_paths)['cap_flow'], 
                               'results', 'shear_rate_analysis.csv')
    df_copy.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Return the modified dataframe
    return df_copy

if __name__ == '__main__':
    main()
