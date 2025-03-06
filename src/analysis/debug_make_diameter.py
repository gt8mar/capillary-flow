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
    
    # plot Mean_Diameter vs Bulk_Diameter
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_df['Mean_Diameter'], merged_df['Bulk_Diameter'], alpha=0.5)
    plt.xlabel('Mean_Diameter')
    plt.ylabel('Bulk_Diameter')
    # set the ylim to be the max of Mean_Diameter
    plt.ylim(0, merged_df['Mean_Diameter'].max())
    plt.title('Mean_Diameter vs Bulk_Diameter')
    plt.show()
    
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
    test_df = big_merged_df[big_merged_df['Corrected_Velocity'].notna() & big_merged_df['Area_x'].notna()]
    # rename the 'Area_x' column to 'Area'
    test_df = test_df.rename(columns={'Area_x': 'Area'})
    
    # # for each participant, plot the 'Corrected_Velocity' vs 'Mean_Diameter'
    # for participant in test_df['Participant'].unique():
    #     plt.figure(figsize=(10, 6))
    #     plt.scatter(test_df[test_df['Participant'] == participant]['Mean_Diameter'], test_df[test_df['Participant'] == participant]['Corrected_Velocity'], alpha=0.5)
    #     plt.xlabel('Mean_Diameter')
    #     plt.ylabel('Corrected_Velocity')
    #     plt.title(f'{participant} - Corrected_Velocity vs Mean_Diameter')
    #     plt.show()

    # For each pressure, plot the 'Corrected_Velocity' vs 'Mean_Diameter'. Color by 'Age' and then make another plot that colors by 'SET'
    # Plot colored by Age
    for pressure in test_df['Pressure'].unique():
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(test_df[test_df['Pressure'] == pressure]['Mean_Diameter'], 
                            test_df[test_df['Pressure'] == pressure]['Corrected_Velocity'],
                            c=test_df[test_df['Pressure'] == pressure]['Age'],
                            alpha=0.5)
        plt.colorbar(scatter, label='Age')
        plt.xlabel('Mean_Diameter')
        plt.ylabel('Corrected_Velocity') 
        plt.title(f'{pressure} - Corrected_Velocity vs Mean_Diameter (colored by Age)')
        plt.show()

    # Plot colored by SET
    for pressure in test_df['Pressure'].unique():
        plt.figure(figsize=(10, 6))
        # Convert SET strings to numbers by extracting digits
        set_numbers = test_df[test_df['Pressure'] == pressure]['SET'].str.extract('(\d+)').astype(float)
        scatter = plt.scatter(test_df[test_df['Pressure'] == pressure]['Mean_Diameter'],
                            test_df[test_df['Pressure'] == pressure]['Corrected_Velocity'], 
                            c=set_numbers,
                            alpha=0.5)
        plt.colorbar(scatter, label='SET')
        plt.xlabel('Mean_Diameter')
        plt.ylabel('Corrected_Velocity')
        plt.title(f'{pressure} - Corrected_Velocity vs Mean_Diameter (colored by SET)')
        plt.show()
    
    # Mixed Effects Model Analysis
    print("\nPerforming Mixed Effects Model Analysis...")
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.regression.mixed_linear_model import MixedLM
        
        # Prepare data for mixed effects model
        model_df = test_df.dropna(subset=['Corrected_Velocity', 'Mean_Diameter', 'Participant', 'Pressure'])
        
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
        plt.show()
        
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
            ax.set_xlabel('Mean Diameter')
            ax.set_ylabel('Pressure')
            ax.set_zlabel('Corrected Velocity')
            ax.set_title('3D Visualization of Mixed Effects Model')
            fig.colorbar(scatter, ax=ax, label='Pressure')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not create 3D visualization: {e}")
        
    except ImportError:
        print("Error: statsmodels package is required for mixed effects modeling.")
        print("Install it using: pip install statsmodels")
    except Exception as e:
        print(f"Error in mixed effects modeling: {e}")

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

if __name__ == '__main__':
    main()
