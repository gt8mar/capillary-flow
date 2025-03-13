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

def main():
    """Main function to debug missing area calculations."""
    # Set paths based on platform
    if platform.system() == 'Windows':
        cap_flow_path = 'C:\\Users\\gt8mar\\capillary-flow'
        diameters_file = os.path.join(cap_flow_path, 'results', 'cap_diameters_areas.csv')
        diameters_file_renamed = os.path.join(cap_flow_path, 'results', 'cap_diameters_areas_renamed.csv')
        velocity_file_path = os.path.join(cap_flow_path, 'merged_csv4.csv')
        segmentation_folder = os.path.join(cap_flow_path, 'results', 'segmented', 'individual_caps_original')
        segmentation_folder_renamed = os.path.join(cap_flow_path, 'results', 'segmented', 'renamed_individual_caps_original')
    else:
        # Default paths for other platforms
        cap_flow_path = "/hpc/projects/capillary-flow"
        diameters_file = os.path.join(cap_flow_path, 'results', 'cap_diameters_areas.csv')
        diameters_file_renamed = os.path.join(cap_flow_path, 'results', 'cap_diameters_areas_renamed.csv')
        velocity_file_path = os.path.join(cap_flow_path, 'merged_csv4.csv')
        segmentation_folder = os.path.join(cap_flow_path, 'results', 'segmented', 'individual_caps_original')
        segmentation_folder_renamed = os.path.join(cap_flow_path, 'results', 'segmented', 'renamed_individual_caps_original')
    print(f"Loading diameter data from: {diameters_file}")

    # plotting parameters
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
    })

    # Set up font
    try:
        source_sans = FontProperties(fname='C:\\Users\\gt8mar\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
    except Exception as e:
        print(f"Warning: Could not set up font: {e}")
        source_sans = None
    
    # Load the diameter data
    try:
        diameter_df = pd.read_csv(diameters_file)
        # strip 'bp' from the 'Video' column
        diameter_df['Video'] = diameter_df['Video'].str.replace('bp', '')
        print(f"Successfully loaded diameter data with {len(diameter_df)} entries")
    except Exception as e:
        print(f"Error loading diameter data: {e}")
        return
    
    # Load the renamed diameter data
    try:
        diameter_df_renamed = pd.read_csv(diameters_file_renamed)
        # strip 'bp' from the 'Video' column
        diameter_df_renamed['Video'] = diameter_df_renamed['Video'].str.replace('bp', '')
        print(f"Successfully loaded renamed diameter data with {len(diameter_df_renamed)} entries")
    except Exception as e:
        print(f"Error loading renamed diameter data: {e}")
        return
    
    # Merge the two dataframes based on area and bulk_diameter values
    print("\nMerging dataframes based on area and bulk_diameter values...")
    
    # Create a merged dataframe
    # merged_df = merge_diameter_dataframes(diameter_df, diameter_df_renamed)
    merged_df = pd.merge(diameter_df, diameter_df_renamed, on=['Participant', 'Date', 'Location', 'Video', 'Capillary'], how='outer')

    # if there is an area value in both Area_x and Area_y, keep the Area_y value and add a new column called 'Area_source' with the value 'renamed'
    merged_df['Area'] = merged_df.apply(lambda row: row['Area_y'] if not pd.isna(row['Area_y']) else row['Area_x'], axis=1)
    merged_df['Area_source'] = merged_df.apply(lambda row: 'renamed' if not pd.isna(row['Area_y']) else 'original', axis=1)
    # drop the Area_x and Area_y columns
    merged_df = merged_df.drop(columns=['Area_x', 'Area_y'])

    # if there is a Bulk_Diameter value in both Bulk_Diameter_x and Bulk_Diameter_y, keep the Bulk_Diameter_y value
    merged_df['Bulk_Diameter'] = merged_df.apply(lambda row: row['Bulk_Diameter_y'] if not pd.isna(row['Bulk_Diameter_y']) else row['Bulk_Diameter_x'], axis=1)
    # drop the Bulk_Diameter_x and Bulk_Diameter_y columns
    merged_df = merged_df.drop(columns=['Bulk_Diameter_x', 'Bulk_Diameter_y'])

    # for Centerline_Length_x	Mean_Radius_x	Std_Radius_x	Mean_Diameter_x	Std_Diameter_x, keep these columns, drop the respective y columns, and rename the x columns to remove the _x suffix
    merged_df = merged_df.drop(columns=['Centerline_Length_y', 'Mean_Radius_y', 'Std_Radius_y', 'Mean_Diameter_y', 'Std_Diameter_y'])
    merged_df = merged_df.rename(columns={'Centerline_Length_x': 'Centerline_Length', 'Mean_Radius_x': 'Mean_Radius', 'Std_Radius_x': 'Std_Radius', 'Mean_Diameter_x': 'Mean_Diameter', 'Std_Diameter_x': 'Std_Diameter'})


    merged_df = check_duplicate_areas(merged_df)
   

    
    # Save the merged dataframe
    merged_output_file = os.path.join(cap_flow_path, 'results', 'cap_diameters_areas_merged.csv')
    merged_df.to_csv(merged_output_file, index=False)
    print(f"Merged dataframe saved to: {merged_output_file}")
    print(f"Merged dataframe has {len(merged_df)} entries")
    
    # Continue with the original analysis using the merged dataframe
    # Identify entries with missing area calculations
    missing_area = merged_df[merged_df['Area'].isna() | (merged_df['Area'] == 0)]
    print(f"\nFound {len(missing_area)} entries with missing area calculations")
    
    if len(missing_area) == 0:
        print("No missing area calculations found. Exiting.")
        # return
    
    # Plot comparison of mean diameter vs bulk diameter
    plt.figure(figsize=(2.4, 2.0))
    plt.scatter(merged_df['Mean_Diameter'], merged_df['Bulk_Diameter'], alpha=0.5)
    plt.xlabel('Mean Diameter (μm)')
    plt.ylabel('Bulk Diameter (μm)') 
    plt.ylim(0, merged_df['Mean_Diameter'].max())
    plt.title('Mean vs Bulk Diameter')
    
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
    
    plt.savefig(os.path.join(cap_flow_path, 'results', 'mean_diameter_vs_bulk_diameter.png'), 
                dpi=600, bbox_inches='tight')
    plt.close()
    
    velocity_file = pd.read_csv(velocity_file_path)
    
    # merge merged_df with merge_csv4 dataframe
    big_merged_df = pd.merge(merged_df, velocity_file, on=['Participant', 'Date', 'Location', 'Video', 'Capillary'], how='outer')

    # Rename 'Corrected Velocity' to 'Corrected_Velocity'
    big_merged_df = big_merged_df.rename(columns={'Corrected Velocity': 'Corrected_Velocity'})

    # keep only the columns: 'Participant', 'Date', 'Location', 'Video', 'Capillary', 'Area', 'Area_source', 'Bulk_Diameter', 'Video Median Velocity', 'Log Video Median Velocity'
    # big_merged_df = big_merged_df[['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Area', 'Area_source', 'Bulk_Diameter', 'Centerline_Length', 'Mean_Radius', 'Std_Radius', 'Mean_Diameter', 'Std_Diameter', 'Corrected_Velocity']]
    # save big_merged_df to a csv file
    big_merged_df.to_csv(os.path.join(cap_flow_path, 'results', 'cap_diameters_areas_merged_with_velocity.csv'), index=False)

    # check if any videos have duplicate capillaries
    duplicate_capillaries = big_merged_df[big_merged_df.duplicated(subset=['Participant', 'Date', 'Location', 'Video', 'Capillary'])]
    print(f"Found {len(duplicate_capillaries)} duplicate capillaries")
    # print the duplicate capillaries
    print(duplicate_capillaries)

    # check if any rows in big_merged_df have an Area_x but not a 'Corrected_Velocity'
    rows_with_area_x_but_no_corrected_velocity = big_merged_df[big_merged_df['Area_x'].notna() & big_merged_df['Corrected_Velocity'].isna()]
    print(f"Found {len(rows_with_area_x_but_no_corrected_velocity)} rows with an Area_x but no 'Corrected_Velocity'")
    # print the rows with an Area_x but no 'Corrected_Velocity'
    print(rows_with_area_x_but_no_corrected_velocity)
    # save this subset to a csv file
    rows_with_area_x_but_no_corrected_velocity.to_csv(os.path.join(cap_flow_path, 'results', 'rows_with_area_x_but_no_corrected_velocity.csv'), index=False)
    
    # make a test dataframe where we only include rows where 'Corrected_Velocity' is not null and 'Area' is not null
    diameter_analysis_df = big_merged_df[big_merged_df['Corrected_Velocity'].notna() & big_merged_df['Area_x'].notna()]
    # rename the 'Area_x' column to 'Area'
    diameter_analysis_df = diameter_analysis_df.rename(columns={'Area_x': 'Area'})
    # save test_df to a csv file
    diameter_analysis_df.to_csv(os.path.join(cap_flow_path, 'results', 'diameter_analysis_df.csv'), index=False)
    
    # Create diameter plots directory if it doesn't exist
    diameter_plots_dir = os.path.join(cap_flow_path, 'results', 'diameter_plots')
    os.makedirs(diameter_plots_dir, exist_ok=True)

    # threshold_analysis(diameter_analysis_df, diameter_plots_dir, cap_flow_path)
    # plot_velocity_vs_diameter_by_age(diameter_analysis_df, diameter_plots_dir)
    # plot_velocity_vs_diameter_by_participant(diameter_analysis_df, diameter_plots_dir)
    # plot_velocity_vs_diameter_by_set(diameter_analysis_df)
    
    # make_mixed_effect_model(diameter_analysis_df)
    # threshold_analysis(diameter_analysis_df)  
    # plot_velocity_vs_diameter_theory()

    plot_pressure_drop_per_length(diameter_analysis_df)
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

def plot_velocity_vs_diameter_by_age(diameter_analysis_df, diameter_plots_dir):
    """
    For each pressure, plot the Corrected_Velocity vs Mean_Diameter, colored by Age.
    
    Args:
        diameter_analysis_df: DataFrame containing the data to plot
        diameter_plots_dir: Directory to save the plots
    """
    for pressure in diameter_analysis_df['Pressure'].unique():
        plt.figure(figsize=(2.4, 2.0))
        scatter = plt.scatter(diameter_analysis_df[diameter_analysis_df['Pressure'] == pressure]['Mean_Diameter'], 
                            diameter_analysis_df[diameter_analysis_df['Pressure'] == pressure]['Corrected_Velocity'],
                            c=diameter_analysis_df[diameter_analysis_df['Pressure'] == pressure]['Age'],
                            alpha=0.5,
                            s=10)
        plt.colorbar(scatter, label='Age')
        plt.xlabel('Mean Diameter (μm)')
        plt.ylabel('Corrected Velocity (μm/s)')
        plt.title(f'P={pressure} PSI')

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
        plt.figure(figsize=(2.4, 2.0))
        plt.scatter(diameter_analysis_df[diameter_analysis_df['Participant'] == participant]['Mean_Diameter'], 
                   diameter_analysis_df[diameter_analysis_df['Participant'] == participant]['Corrected_Velocity'], 
                   alpha=0.5)
        plt.xlabel('Mean Diameter (μm)')
        plt.ylabel('Corrected Velocity (μm/s)')
        plt.title(f'{participant} - Velocity vs Diameter')
        
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

        plt.savefig(os.path.join(diameter_plots_dir, f'{participant}_velocity_vs_diameter.png'),
                    dpi=600, bbox_inches='tight')
        plt.close()
def plot_velocity_vs_diameter_by_set(diameter_analysis_df):
    """
    Plot velocity vs diameter colored by SET number for each pressure level.
    
    Args:
        diameter_analysis_df: DataFrame containing Mean_Diameter, Corrected_Velocity, Pressure and SET columns
    """
    for pressure in diameter_analysis_df['Pressure'].unique():
        plt.figure(figsize=(10, 6))
        # Convert SET strings to numbers by extracting digits
        set_numbers = diameter_analysis_df[diameter_analysis_df['Pressure'] == pressure]['SET'].str.extract('(\d+)').astype(float)
        scatter = plt.scatter(diameter_analysis_df[diameter_analysis_df['Pressure'] == pressure]['Mean_Diameter'],
                            diameter_analysis_df[diameter_analysis_df['Pressure'] == pressure]['Corrected_Velocity'], 
                            c=set_numbers,
                            alpha=0.5)
        plt.colorbar(scatter, label='SET')
        plt.xlabel('Mean_Diameter')
        plt.ylabel('Corrected_Velocity')
        plt.title(f'{pressure} - Corrected_Velocity vs Mean_Diameter (colored by SET)')
        plt.show()
def make_mixed_effect_model(diameter_analysis_df):
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
        plt.figure(figsize=(12, 8))
        
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
        
        plt.xlabel('Mean Diameter')
        plt.ylabel('Corrected Velocity')
        plt.title('Mixed Effects Model: Diameter vs Velocity by Pressure')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        # plt.show()
        plt.close() 
        
        # Create a 3D visualization to better show the relationship
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10))
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
            ax.set_xlabel('Mean Diameter (μm)')
            ax.set_ylabel('Pressure (PSI)')
            ax.set_zlabel('Corrected Velocity (μm/s)')
            ax.set_title('3D Visualization of Mixed Effects Model')
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
        fig, ax = plt.subplots(figsize=(10, 6))
    
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
    ax.set_xlabel('Mean Diameter')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'CDF of Mean Diameter by Age Group (Threshold: {age_threshold} years)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(diameter_plots_dir, f'age_threshold_{age_threshold}_cdf_plot.png'),
                dpi=600, bbox_inches='tight')
    plt.close()
    
    return ks_stat

def check_duplicate_areas(merged_df: pd.DataFrame):
    """
    Check for duplicate areas in the merged dataframe and remove original capillaries
    when a renamed version exists with the same area in the same video.
    
    Args:
        merged_df: DataFrame containing capillary data with Area, Area_source, and metadata columns
        
    Returns:
        DataFrame with duplicate original capillaries removed
    """
    # Create a copy to avoid modifying the original dataframe
    df = merged_df.copy()
    
    # Group by Participant, Location, Video and Area to find duplicates
    duplicate_groups = df.groupby(['Participant', 'Location', 'Video', 'Area'])
    
    # Count how many duplicates we have
    duplicate_count = 0
    rows_to_drop = []
    
    # Iterate through each group
    for (participant, location, video, area), group in duplicate_groups:
        # If we have more than one capillary with the same area in the same video
        if len(group) > 1:
            # Check if we have both 'renamed' and 'original' in this group
            if 'renamed' in group['Area_source'].values and 'original' in group['Area_source'].values:
                # Get indices of 'original' rows to drop
                original_indices = group[group['Area_source'] == 'original'].index.tolist()
                rows_to_drop.extend(original_indices)
                duplicate_count += len(original_indices)
    
    # Drop the original capillaries that have renamed versions
    if rows_to_drop:
        print(f"Found {duplicate_count} duplicate original capillaries with renamed versions.")
        print("Dropping original capillaries that have renamed versions with the same area.")
        df = df.drop(rows_to_drop)
    else:
        print("No duplicate capillaries found with both original and renamed versions.")
    
    return df

def merge_diameter_dataframes(original_df: pd.DataFrame, renamed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the original and renamed diameter dataframes based on area and bulk_diameter values.
    
    For each participant-video combination, if the renamed dataframe has area values,
    use only the rows from the renamed dataframe for that participant-video.
    Otherwise, use the rows from the original dataframe.
    
    Args:
        original_df: The original diameter dataframe
        renamed_df: The renamed diameter dataframe
        
    Returns:
        A merged dataframe with prioritized data
    """
    # Create a copy of the dataframes to avoid modifying the originals
    original_df = original_df.copy()
    renamed_df = renamed_df.copy()
    
    # Add a source column to track where each row came from
    original_df['source'] = 'original'
    renamed_df['source'] = 'renamed'
    
    # Identify participant-video combinations in both dataframes
    original_groups = original_df.groupby(['Participant', 'Date', 'Location', 'Video'])
    renamed_groups = renamed_df.groupby(['Participant', 'Date', 'Location', 'Video'])
    
    # Create a list to store the rows for the final merged dataframe
    merged_rows = []
    
    # Track which participant-video combinations we've processed
    processed_combinations = set()
    
    # First, check renamed dataframe for participant-video combinations with area values
    for (participant, date, location, video), group in renamed_groups:
        key = (participant, date, location, video)
        
        # Check if this group has any non-null area values
        has_area = not group['Area'].isna().all() and not (group['Area'] == 0).all()
        
        if has_area:
            # If renamed dataframe has area values for this combination, use it
            merged_rows.append(group)
            processed_combinations.add(key)
            print(f"Using renamed data for {participant}, {date}, {location}, {video} (has area values)")
        
    # For any participant-video combinations not in the renamed dataframe or without area values,
    # use the original dataframe
    for (participant, date, location, video), group in original_groups:
        key = (participant, date, location, video)
        
        if key not in processed_combinations:
            merged_rows.append(group)
            print(f"Using original data for {participant}, {date}, {location}, {video}")
    
    # Concatenate all the selected rows
    result_df = pd.concat(merged_rows, ignore_index=True)
    
    # Print some statistics about the merge
    original_count = len(original_df)
    renamed_count = len(renamed_df)
    result_count = len(result_df)
    
    print(f"\nMerge statistics:")
    print(f"Original dataframe: {original_count} rows")
    print(f"Renamed dataframe: {renamed_count} rows")
    print(f"Merged dataframe: {result_count} rows")
    
    # Count how many participant-video combinations came from each source
    original_videos = len(result_df[result_df['source'] == 'original'].groupby(['Participant', 'Date', 'Location', 'Video']))
    renamed_videos = len(result_df[result_df['source'] == 'renamed'].groupby(['Participant', 'Date', 'Location', 'Video']))
    
    print(f"Participant-video combinations from original: {original_videos}")
    print(f"Participant-video combinations from renamed: {renamed_videos}")

    # Print rows that have no area values at the end of the merged dataframe
    print("\nRows with no area values:")
    no_area_rows = result_df[result_df['Area'].isna() | (result_df['Area'] == 0)]
    print(f"Number of rows with no area values: {len(no_area_rows)}")
    
    # Now drop the source column before returning
    result_df = result_df.drop(columns=['source'])
    
    return result_df

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


    diameters = np.linspace(1.5, 1000, 1000) # um
    viscosities = secomb_viscocity_fn(diameters) # cp
    plt.plot(diameters, viscosities)
    # plt.scatter(fig_csv[:, 0], fig_csv[:, 1], color='red')
    plt.xlabel('Diameter (um)')
    # make the x axis on a log scale
    plt.xscale('log')
    plt.ylabel('Viscosity (cp)')
    plt.ylim(1, 7)
    plt.title('Viscosity vs Diameter based on Secomb Viscosity')
    plt.show()



    velocities = viscosity_to_velocity(viscosities, diameters) # um/s
    plt.plot(diameters, velocities)
    plt.xlabel('Diameter (um)')
    plt.xlim(1, 60)
    plt.ylabel('Velocity (um/s)')
    plt.ylim(0, 7000)
    plt.title('Velocity vs Diameter based on Secomb Viscosity')
    plt.show()


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


if __name__ == '__main__':
    main()
