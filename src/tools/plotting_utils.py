"""
Filename: src/tools/plotting_utils.py

This file contains utility functions for plotting.

By: Marcus Forst
"""

import os
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from scipy import stats
from sklearn.utils import resample
from scipy.stats import ks_2samp
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from src.config import PATHS
import colorsys

# Get the hostname of the computer
hostname = platform.node()

# Function to safely load the SourceSans font
def get_source_sans_font():
    """
    Safely load the SourceSans font from the downloads path in config.
    Falls back to a default font if the Source Sans font is not available.
    
    Returns:
        FontProperties: Font for consistent text rendering
    """
    try:
        # First try using the PATHS from config.py
        if 'downloads' in PATHS:
            font_path = os.path.join(PATHS['downloads'], 'Source_Sans_3', 'static', 'SourceSans3-Regular.ttf')
            if os.path.exists(font_path):
                return FontProperties(fname=font_path)
        
        # Try platform-specific paths as fallback
        computer_paths = {
            "LAPTOP-I5KTBOR3": 'C:\\Users\\gt8ma\\Downloads',
            "Quake-Blood": 'C:\\Users\\gt8mar\\Downloads',
            "Clark-": 'C:\\Users\\ejerison\\Downloads'
        }
        
        # Try to find the hostname prefix
        for prefix in computer_paths:
            if hostname.startswith(prefix):
                downloads_path = computer_paths[prefix]
                font_path = os.path.join(downloads_path, 'Source_Sans_3', 'static', 'SourceSans3-Regular.ttf')
                if os.path.exists(font_path):
                    return FontProperties(fname=font_path)
        
        # If we can't find the font, use a default system font
        print("Warning: SourceSans3-Regular.ttf not found, using default font")
        return None
    except Exception as e:
        print(f"Warning: Error loading font: {e}")
        return None

# Set up source_sans for backward compatibility with other functions
source_sans = get_source_sans_font()

def plot_PCA(participant_medians):
    """
    Perform Principal Component Analysis on participant median data and create visualizations.
    
    This function takes participant median data (shear rate, velocity, pressure drop, 
    and diameter), performs PCA, and creates various plots to visualize the results,
    including scatterplots of principal components colored by participant metadata.
    
    Args:
        participant_medians: DataFrame containing median values per participant
                             with columns for Shear_Rate, Corrected_Velocity,
                             Pressure_Drop, and Mean_Diameter and metadata columns for Age, SET, and Sex
                             
    Returns:
        None. Saves PCA plots to results directory.
    """
    print("\nPerforming PCA on participant median data...")
    source_sans = get_source_sans_font()
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        from src.config import PATHS
        
        # Create output directory
        output_dir = os.path.join(PATHS.get('cap_flow', '.'), 'results', 'pca')
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if we have enough data
        if participant_medians.shape[0] < 3:
            print("Error: Not enough participants for PCA analysis.")
            return
            
        # Select features for PCA
        features = ['Shear_Rate', 'Corrected_Velocity', 'Pressure_Drop', 'Mean_Diameter']
        
        # Check which features are available
        missing_features = [f for f in features if f not in participant_medians.columns]
        if missing_features:
            print(f"Warning: Missing features for PCA: {missing_features}")
            features = [f for f in features if f in participant_medians.columns]
            
        if len(features) < 2:
            print("Error: Not enough features available for PCA.")
            return
            
        # Select data for PCA and remove any rows with NaN values
        data = participant_medians[features].dropna()
        
        # Check if we still have enough data after dropping NaNs
        if data.shape[0] < 3:
            print("Error: Not enough complete data rows for PCA after removing NaNs.")
            return
            
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Perform PCA
        pca = PCA(n_components=min(len(features), 3))  # At most 3 components
        principal_components = pca.fit_transform(scaled_data)
        
        # Create a DataFrame with the principal components
        columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
        pca_df = pd.DataFrame(data=principal_components, columns=columns)
        
        # Add participant information
        participant_indices = data.index.tolist()
        for col in participant_medians.columns:
            if col not in features and col in participant_medians.columns:
                pca_df[col] = participant_medians.loc[participant_indices, col].values
        
        # Print explained variance
        explained_variance = pca.explained_variance_ratio_
        print("Explained variance ratio:")
        for i, var in enumerate(explained_variance):
            print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")
        print(f"Total variance explained: {sum(explained_variance)*100:.2f}%")
        
        # Plot feature loadings
        plt.figure(figsize=(2.4, 2.0))
        loadings = pca.components_.T
        loading_df = pd.DataFrame(loadings, columns=columns, index=features)
        
        # Heatmap of loadings
        sns.heatmap(loading_df, annot=True, cmap='viridis', fmt=".2f", cbar=True)
        plt.title('PCA Feature Loadings', fontproperties=source_sans, fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_loadings.png'), dpi=600, bbox_inches='tight')
        plt.close()
        
        # Plot explained variance
        plt.figure(figsize=(2.4, 2.0))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
        plt.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), 'ro-')
        plt.xticks(range(1, len(explained_variance) + 1), [f'PC{i}' for i in range(1, len(explained_variance) + 1)])
        plt.xlabel('Principal Components', fontproperties=source_sans)
        plt.ylabel('Explained Variance Ratio', fontproperties=source_sans)
        plt.title('Explained Variance by Components', fontproperties=source_sans, fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'), dpi=600, bbox_inches='tight')
        plt.close()
        
        # Scatter plot of PC1 vs PC2
        if principal_components.shape[1] >= 2:
            # Check if we have metadata columns to color by
            metadata_columns = []
            for col in ['Age', 'SET', 'Sex']:
                if col in participant_medians.columns:
                    metadata_columns.append(col)
            
            # If no metadata columns, just create a single plot
            if not metadata_columns:
                plt.figure(figsize=(2.4, 2.0))
                plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
                plt.xlabel('Principal Component 1', fontproperties=source_sans)
                plt.ylabel('Principal Component 2', fontproperties=source_sans)
                plt.title('PCA: PC1 vs PC2', fontproperties=source_sans, fontsize=7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'pca_pc1_vs_pc2.png'), dpi=600, bbox_inches='tight')
                plt.close()
            else:
                # Create separate plots for each metadata column
                for col in metadata_columns:
                    plt.figure(figsize=(2.4, 2.0))
                    
                    # For numeric columns like Age, use a colormap
                    if pd.api.types.is_numeric_dtype(pca_df[col]):
                        scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                                             c=pca_df[col], cmap='viridis', 
                                             alpha=0.7)
                        plt.colorbar(scatter, label=col)
                    else:
                        # For categorical columns, use categorical coloring with a legend
                        categories = pca_df[col].unique()
                        for category in categories:
                            subset = pca_df[pca_df[col] == category]
                            plt.scatter(subset['PC1'], subset['PC2'], 
                                        alpha=0.7, label=str(category))
                        plt.legend(prop={'size': 5})
                    
                    plt.xlabel('Principal Component 1', fontproperties=source_sans)
                    plt.ylabel('Principal Component 2', fontproperties=source_sans)
                    plt.title(f'PCA: PC1 vs PC2 (by {col})', fontproperties=source_sans, fontsize=7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'pca_pc1_vs_pc2_by_{col}.png'), 
                               dpi=600, bbox_inches='tight')
                    plt.close()
        
        # Scatter plot of PC1 vs PC3 (if we have 3 components)
        if principal_components.shape[1] >= 3:
            # Similar structure as PC1 vs PC2 plots
            if not metadata_columns:
                plt.figure(figsize=(2.4, 2.0))
                plt.scatter(pca_df['PC1'], pca_df['PC3'], alpha=0.7)
                plt.xlabel('Principal Component 1', fontproperties=source_sans)
                plt.ylabel('Principal Component 3', fontproperties=source_sans)
                plt.title('PCA: PC1 vs PC3', fontproperties=source_sans, fontsize=7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'pca_pc1_vs_pc3.png'), dpi=600, bbox_inches='tight')
                plt.close()
            else:
                for col in metadata_columns:
                    plt.figure(figsize=(2.4, 2.0))
                    
                    if pd.api.types.is_numeric_dtype(pca_df[col]):
                        scatter = plt.scatter(pca_df['PC1'], pca_df['PC3'], 
                                             c=pca_df[col], cmap='viridis', 
                                             alpha=0.7)
                        plt.colorbar(scatter, label=col)
                    else:
                        categories = pca_df[col].unique()
                        for category in categories:
                            subset = pca_df[pca_df[col] == category]
                            plt.scatter(subset['PC1'], subset['PC3'], 
                                        alpha=0.7, label=str(category))
                        plt.legend(prop={'size': 5})
                    
                    plt.xlabel('Principal Component 1', fontproperties=source_sans)
                    plt.ylabel('Principal Component 3', fontproperties=source_sans)
                    plt.title(f'PCA: PC1 vs PC3 (by {col})', fontproperties=source_sans, fontsize=7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'pca_pc1_vs_pc3_by_{col}.png'), 
                               dpi=600, bbox_inches='tight')
                    plt.close()
                    
        # Create a 3D plot if we have 3 components
        if principal_components.shape[1] >= 3 and metadata_columns:
            try:
                from mpl_toolkits.mplot3d import Axes3D
                
                for col in metadata_columns:
                    fig = plt.figure(figsize=(3.2, 2.8))  # Slightly larger for 3D plot
                    ax = fig.add_subplot(111, projection='3d')
                    
                    if pd.api.types.is_numeric_dtype(pca_df[col]):
                        scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'],
                                           c=pca_df[col], cmap='viridis', 
                                           alpha=0.7)
                        fig.colorbar(scatter, ax=ax, label=col)
                    else:
                        categories = pca_df[col].unique()
                        for category in categories:
                            subset = pca_df[pca_df[col] == category]
                            ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'],
                                      alpha=0.7, label=str(category))
                        ax.legend(prop={'size': 5})
                    
                    ax.set_xlabel('PC1', fontproperties=source_sans)
                    ax.set_ylabel('PC2', fontproperties=source_sans)
                    ax.set_zlabel('PC3', fontproperties=source_sans)
                    ax.set_title(f'3D PCA by {col}', fontproperties=source_sans, fontsize=7)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'pca_3d_by_{col}.png'), 
                               dpi=600, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                print(f"Could not create 3D PCA plot: {e}")
        
        # Return the PCA results dataframe in case it's needed elsewhere
        return pca_df
        
    except ImportError as e:
        print(f"Error importing required libraries for PCA: {e}")
        print("Please install scikit-learn with: pip install scikit-learn")
        return None
    except Exception as e:
        print(f"Error performing PCA: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_CI(df, variable='Age', method='bootstrap', n_iterations=1000, 
            ci_percentile=99.5, write=True, dimensionless=False, video_median=False, 
            log_scale=False, old = False, velocity_variable = 'Corrected Velocity',
            participant_weighting=False):
    """
    Plots the mean/median and CI for the variable of interest, 
    with KS statistic. 
    
    Args:
        df (pd.DataFrame): The dataframe to plot.
        variable (str): The variable to plot. Example: 'Age', 'Sex', 'SYS_BP', 'Diabetes', 'Hypertension', 'Set_affected'
        method (str): The method to use for the CI. Example: 'bootstrap', 't-test'
        n_iterations (int): The number of iterations to use for the CI.
        ci_percentile (float): The percentile to use for the CI.
        write (bool): Whether to write the plot to a file.
        participant_weighting (bool): If True, each participant will be weighted equally in the distribution regardless of number of data points.

    Returns:
        None
    """
    # Set up style and font
    sns.set_style("whitegrid")
    source_sans = get_source_sans_font()
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    # control_df = df[df['SET']=='set01']
    # hypertensive_df = df[df['SET']=='set02']
    # diabetic_df = df[df['SET']=='set03']
    # if 'Set_affected' in df.columns:
    #     affected_df = df[df['Set_affected']=='set04']
    # else:
    #     print('Set_affected not in df. Using SET column.')
    #     affected_df = df[df['SET']=='set03']
    #     affected_df = affected_df.append(df[df['SET']=='set02'])

    # Set color palette based on variable
    if variable == 'Age':
        base_color = '#1f77b4'
        conditions = [df[variable] <= 59, df[variable] > 59]
        choices = ['≤59', '>59']
    elif variable == 'SYS_BP':
        base_color = '2ca02c'
        conditions = [df[variable] < 120, df[variable] >= 120]
        choices = ['<120', '≥120']
    elif variable == 'Sex':
        base_color = '674F92'
        conditions = [df[variable] == 'M', df[variable] == 'F']
        choices = ['Male', 'Female']
    elif variable == 'Diabetes':
        base_color = 'ff7f0e'
        # conditions = [
        #     df[variable].isin([False, None, 'Control', 'FALSE', 'PRE']),
        #     df[variable].isin([True, 'TRUE','TYPE 1', 'TYPE 2', 'Diabetes'])
        # ]
        conditions = [df['SET'] == 'set01', df['SET'] == 'set03']
        choices = ['Control', 'Diabetic']
    elif variable == 'Hypertension':
        base_color = 'd62728'
        # conditions = [
        #     df[variable].isin([False, None, 'Control', 'FALSE']),
        #     df[variable].isin([True, 1.0, 'Hypertension', 'TRUE'])
        # ]
        conditions = [df['SET'] == 'set01', df['SET'] == 'set02']
        choices = ['Control', 'Hypertensive']
    elif variable == 'Set_affected':
        base_color = '#00CED1' # sky blue
        conditions = [df['SET'] == 'set01', df['Set_affected'] == 'set04']
        choices = ['Control', 'Affected']
    else:
        raise ValueError(f"Unsupported variable: {variable}")

    palette = create_monochromatic_palette(base_color)
    palette = adjust_brightness_of_colors(palette, brightness_scale=.1)
    sns.set_palette(palette)

    if log_scale:
        df[velocity_variable] = df[velocity_variable] + 10


    if video_median:
        df = df.groupby(['Participant', 'Video', 'Capillary']).first().reset_index()
        df.rename(columns={'Dimensionless Velocity': 'Dimensionless Velocity OG'}, inplace=True) 
        df.rename(columns={'Video Median Dimensionless Velocity': 'Dimensionless Velocity'}, inplace=True)      

    # Group data with more explicit handling
    group_col = f'{variable} Group'
    df[group_col] = np.select(conditions, choices, default='Unknown')
    
    # # Filter out 'Unknown' values
    # df = df[df[group_col] != 'Unknown']
    
    # Print unique values for debugging
    print(f"Unique values in {group_col}: {df[group_col].unique()}")

    # # DEBUG: Print the velocity_variable and column names to verify
    # print(f"DEBUG - velocity_variable set to: '{velocity_variable}'")
    # print(f"DEBUG - Column names in dataframe: {df.columns.tolist()}")
    
    # Calculate stats
    stats_func = calculate_median_ci if method == 'bootstrap' else calculate_mean_ci
    
    # # DEBUG: Print parameters being passed to the stats function
    # print(f"DEBUG - Calling stats_func with parameters: method={method}, ci_percentile={ci_percentile}, dimensionless={dimensionless}, velocity_variable={velocity_variable}")
    
    stats_df = df.groupby([group_col, 'Pressure']).apply(
        lambda x: stats_func(x, ci_percentile=ci_percentile, dimensionless=dimensionless, 
                            velocity_variable=velocity_variable, participant_weighting=participant_weighting)
    ).reset_index()
    
    # # DEBUG: Print stats_df columns and a sample row
    # print(f"DEBUG - Columns in stats_df: {stats_df.columns.tolist()}")
    # if not stats_df.empty:
    #     print(f"DEBUG - Sample row from stats_df:")
    #     print(stats_df.iloc[0])
    
    # Calculate KS statistic
    grouped = df.groupby(group_col)
    ks_stats = []
    
    for pressure in df['Pressure'].unique():
        try:
            # Check if both groups exist
            if choices[0] not in grouped.groups or choices[1] not in grouped.groups:
                print(f"Warning: One or more groups missing for pressure {pressure}")
                continue
                
            group_1 = grouped.get_group(choices[0])
            group_2 = grouped.get_group(choices[1])
            
            if velocity_variable == 'Corrected_Velocity':
                if log_scale:
                    group_1['Log Corrected Velocity'] = np.log(group_1['Corrected Velocity'])
                    group_2['Log Corrected Velocity'] = np.log(group_2['Corrected Velocity'])
                    group_1_velocities = group_1[group_1['Pressure'] == pressure]['Log Corrected Velocity']
                    group_2_velocities = group_2[group_2['Pressure'] == pressure]['Log Corrected Velocity']
                else:
                    group_1_velocities = group_1[group_1['Pressure'] == pressure]['Corrected Velocity']
                    group_2_velocities = group_2[group_2['Pressure'] == pressure]['Corrected Velocity']
                
                # Check if either group has empty data for this pressure
                if len(group_1_velocities) == 0 or len(group_2_velocities) == 0:
                    print(f"Warning: Insufficient data for KS test at pressure {pressure}")
                    continue
                
                ks_stat, p_value = ks_2samp(group_1_velocities, group_2_velocities)
                if log_scale:
                    group_1_median = np.log(group_1[group_1['Pressure'] == pressure]['Log Corrected Velocity'].median())
                    group_2_median = np.log(group_2[group_2['Pressure'] == pressure]['Log Corrected Velocity'].median())
                else:
                    group_1_median = group_1[group_1['Pressure'] == pressure]['Corrected Velocity'].median()
                    group_2_median = group_2[group_2['Pressure'] == pressure]['Corrected Velocity'].median()
                ks_stats.append({'Pressure': pressure, 'KS Statistic': ks_stat, 'p-value': p_value, 'Group 1 Median': group_1_median, 'Group 2 Median': group_2_median})
            elif velocity_variable == 'Shear_Rate':
                if log_scale:
                    group_1['Log Shear Rate'] = np.log(group_1['Shear_Rate'])
                    group_2['Log Shear Rate'] = np.log(group_2['Shear_Rate'])
                    group_1_velocities = group_1[group_1['Pressure'] == pressure]['Log Shear Rate']
                    group_2_velocities = group_2[group_2['Pressure'] == pressure]['Log Shear Rate']
                else:
                    group_1_velocities = group_1[group_1['Pressure'] == pressure]['Shear_Rate']
                    group_2_velocities = group_2[group_2['Pressure'] == pressure]['Shear_Rate']
                
                # Check if either group has empty data for this pressure
                if len(group_1_velocities) == 0 or len(group_2_velocities) == 0:
                    print(f"Warning: Insufficient data for KS test at pressure {pressure}")
                    continue
                    
                ks_stat, p_value = ks_2samp(group_1_velocities, group_2_velocities)
                if log_scale:
                    group_1_median = np.log(group_1[group_1['Pressure'] == pressure]['Log Shear Rate'].median())
                    group_2_median = np.log(group_2[group_2['Pressure'] == pressure]['Log Shear Rate'].median())
                else:
                    group_1_median = group_1[group_1['Pressure'] == pressure]['Shear_Rate'].median()
                    group_2_median = group_2[group_2['Pressure'] == pressure]['Shear_Rate'].median()
                ks_stats.append({'Pressure': pressure, 'KS Statistic': ks_stat, 'p-value': p_value, 'Group 1 Median': group_1_median, 'Group 2 Median': group_2_median})

        except KeyError as e:
            print(f"Warning: Could not find group for pressure {pressure}: {e}")
            continue

    if ks_stats:  # Only create DataFrame if we have statistics
        ks_df = pd.DataFrame(ks_stats)
        print(variable)
        print(ks_df)

    # Plot
    plt.close()
    fig, ax = plt.subplots(figsize=(2.4, 2.0))

    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

    # Ensure consistent coloring by using only two colors
    control_color = palette[4]
    condition_color = palette[1]
    
    # Set y-axis column name based on available columns
    if dimensionless:
        y_col = 'Median Dimensionless Velocity' if method == 'bootstrap' else 'Mean Dimensionless Velocity'
    elif velocity_variable == 'Shear_Rate':
        y_col = 'Median Shear Rate' if method == 'bootstrap' else 'Mean Shear Rate'
    else:
        y_col = 'Median Velocity' if method == 'bootstrap' else 'Mean Velocity'
        
    lower_col = 'CI Lower Bound' if method == 'bootstrap' else 'Lower Bound'
    upper_col = 'CI Upper Bound' if method == 'bootstrap' else 'Upper Bound'

    # # DEBUG: Print the expected columns
    # print(f"DEBUG - Looking for y_col: '{y_col}', lower_col: '{lower_col}', upper_col: '{upper_col}'")
    
    # # DEBUG: Check if the expected columns exist
    # if y_col not in stats_df.columns:
    #     print(f"ERROR - Column '{y_col}' not found in stats_df")
    #     # Try to identify a suitable replacement column
    #     for potential_col in stats_df.columns:
    #         if 'median' in potential_col.lower() or 'mean' in potential_col.lower():
    #             print(f"DEBUG - Potential replacement column found: '{potential_col}'")
    
    # Plot control group
    control_data = stats_df[stats_df[group_col] == choices[0]]
    
    # # DEBUG: Print control_data info
    # print(f"DEBUG - Control data shape: {control_data.shape}")
    # print(f"DEBUG - Control data columns: {control_data.columns.tolist()}")
    
    # Check that we have control data before plotting
    if not control_data.empty and y_col in control_data.columns and not control_data[y_col].isna().all():
        ax.errorbar(control_data['Pressure'], control_data[y_col],
                    yerr=[control_data[y_col] - control_data[lower_col], 
                          control_data[upper_col] - control_data[y_col]],
                    label=choices[0], fmt='-o', markersize=2, color=control_color)
        ax.fill_between(control_data['Pressure'], control_data[lower_col], 
                        control_data[upper_col], alpha=0.4, color=control_color)
    else:
        print(f"Warning: No valid data for {choices[0]} group or '{y_col}' column not found")

    # Plot condition group
    condition_data = stats_df[stats_df[group_col] == choices[1]]
    
    # # DEBUG: Print condition_data info
    # print(f"DEBUG - Condition data shape: {condition_data.shape}")
    # print(f"DEBUG - Condition data columns: {condition_data.columns.tolist()}")
    
    # Check that we have condition data before plotting
    if not condition_data.empty and y_col in condition_data.columns and not condition_data[y_col].isna().all():
        ax.errorbar(condition_data['Pressure'], condition_data[y_col],
                    yerr=[condition_data[y_col] - condition_data[lower_col], 
                          condition_data[upper_col] - condition_data[y_col]],
                    label=choices[1], fmt='-o', markersize=2, color=condition_color)
        ax.fill_between(condition_data['Pressure'], condition_data[lower_col], 
                        condition_data[upper_col], alpha=0.4, color=condition_color)
    else:
        print(f"Warning: No valid data for {choices[1]} group or '{y_col}' column not found")

    # Add log scale if requested
    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        
    # Create legend handles with consistent colors
    legend_handles = [mpatches.Patch(color=control_color, label=f'{choices[0]} group', alpha=0.6),
                     mpatches.Patch(color=condition_color, label=f'{choices[1]} group', alpha=0.6)]

    ax.set_xlabel('Pressure (psi)', fontproperties=source_sans)
    
    # Set appropriate y-axis label and title
    if dimensionless:
        if source_sans:
            ax.set_ylabel('Dimensionless Velocity', fontproperties=source_sans)
            ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Dimensionless Velocity vs. Pressure with {ci_percentile}% CI', 
                        fontproperties=source_sans, fontsize=8)
        else:
            ax.set_ylabel('Dimensionless Velocity')
            ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Dimensionless Velocity vs. Pressure with {ci_percentile}% CI', 
                        fontsize=8)
    elif velocity_variable == 'Shear_Rate':
        if source_sans:
            ax.set_ylabel('Shear Rate (1/s)', fontproperties=source_sans)
            ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Shear Rate vs. Pressure with {ci_percentile}% CI', 
                        fontproperties=source_sans, fontsize=8)
        else:
            ax.set_ylabel('Shear Rate (1/s)')
            ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Shear Rate vs. Pressure with {ci_percentile}% CI', 
                        fontsize=8)
    else:
        if source_sans:
            ax.set_ylabel('Velocity (um/s)', fontproperties=source_sans)
            ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Velocity vs. Pressure with {ci_percentile}% CI', 
                        fontproperties=source_sans, fontsize=8)
        else:
            ax.set_ylabel('Velocity (um/s)')
            ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Velocity vs. Pressure with {ci_percentile}% CI', 
                        fontsize=8)
    
    # Handle font properties for legend
    if source_sans:
        ax.legend(handles=legend_handles, prop=source_sans)
    else:
        ax.legend(handles=legend_handles)
        
    ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    
    # Determine cap_flow_path from config if not imported
    try:
        cap_flow_path = PATHS.get('cap_flow', os.getcwd())
        output_cap_flow_path = cap_flow_path
    except NameError:
        output_cap_flow_path = PATHS.get('cap_flow', os.getcwd())
    
    # Save or display the plot
    if write:
        # Create results/shear directory if it doesn't exist (for shear rate plots)
        if velocity_variable == 'Shear_Rate':
            shear_dir = os.path.join(output_cap_flow_path, 'results', 'shear')
            os.makedirs(shear_dir, exist_ok=True)
            
        if video_median:
            if old:
                output_path = os.path.join(output_cap_flow_path, 'results', f'{variable}_videomedians_CI_old.png')
            else:
                if velocity_variable == 'Shear_Rate':
                    output_path = os.path.join(output_cap_flow_path, 'results', 'shear', f'{variable}_videomedians_CI_shear_rate.png')
                else:
                    output_path = os.path.join(output_cap_flow_path, 'results', f'{variable}_videomedians_CI_new.png')
        else:
            if velocity_variable == 'Shear_Rate':
                output_path = os.path.join(output_cap_flow_path, 'results', 'shear', f'{variable}_CI_shear_rate.png')
            else:
                output_path = os.path.join(output_cap_flow_path, 'results', f'{variable}_CI_new.png')
        if participant_weighting:
            output_path = os.path.join(output_cap_flow_path, 'results', f'{variable}_participant_weighted_CI_new.png')
        plt.savefig(output_path, dpi=600)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
        
    return 0

# Function to calculate median and bootstrap 95% CI
def calculate_median_ci(group, n_iterations=1000, ci_percentile=95, dimensionless=False, 
                        velocity_variable='Corrected_Velocity', participant_weighting=False):
    """
    Calculates the median and bootstrap 95% CI for the given group.
    If the group has a 'Dimensionless Velocity' column, it will use that column.
    Otherwise, it will use the 'Corrected_Velocity' or the 'Shear_Rate' column.

    Args:
        group (pd.DataFrame): The group to calculate the median and CI for.
        n_iterations (int): The number of bootstrap iterations to use.
        ci_percentile (int): The percentile to use for the CI.
        dimensionless (bool): Whether to return the median and CI in dimensionless units.
        velocity_variable (str): The variable to use for the velocity. Options are 'Corrected_Velocity' or 'Shear_Rate'.
        participant_weighting (bool): If True, each participant will be weighted equally in the bootstrapping process.

    Returns:
        pd.Series: A series containing the median and CI bounds.
    """
    # # DEBUG print statements
    # print(f"DEBUG - calculate_median_ci called with:")
    # print(f"  n_iterations={n_iterations}, ci_percentile={ci_percentile}")
    # print(f"  dimensionless={dimensionless}, velocity_variable='{velocity_variable}'")
    # print(f"  group shape: {group.shape}")
    # print(f"  group columns: {group.columns.tolist()}")
    
    # Initialize results with default values
    result = {
        'Median Dimensionless Velocity': np.nan,
        'Median Velocity': np.nan,
        'Median Shear Rate': np.nan,
        'CI Lower Bound': np.nan,
        'CI Upper Bound': np.nan
    }
    
    # Check if we have any data to work with
    if len(group) == 0:
        print("DEBUG - Group is empty, returning default values")
        return pd.Series(result)
    
    medians_dimless = []
    medians = []
    
    if dimensionless and 'Dimensionless Velocity' in group.columns:
        print("DEBUG - Using Dimensionless Velocity")
        # Drop any NaN values before bootstrapping
        valid_data = group['Dimensionless Velocity'].dropna()
        
        if len(valid_data) == 0:
            print("DEBUG - No valid Dimensionless Velocity data")
            return pd.Series(result)
        
        if participant_weighting:
            # Group data by participant and calculate participant-level medians
            participant_medians = group.groupby('Participant')['Dimensionless Velocity'].median().dropna()
            
            if len(participant_medians) == 0:
                print("DEBUG - No valid participant medians for Dimensionless Velocity")
                return pd.Series(result)
                
            # Bootstrap at the participant level
            for _ in range(n_iterations):
                # Resample participants with replacement
                sample = resample(participant_medians)
                medians_dimless.append(np.median(sample))
                
            # Calculate the overall median using participant-level medians
            median_dimless = np.median(participant_medians)
        else:
            # Original bootstrapping method at the measurement level
            for _ in range(n_iterations):
                sample = resample(valid_data)
                medians_dimless.append(np.median(sample))
                
            median_dimless = np.median(valid_data)
            
        lower = np.percentile(medians_dimless, (100 - ci_percentile) / 2)
        upper = np.percentile(medians_dimless, 100 - (100 - ci_percentile) / 2)
        
        result = {
            'Median Dimensionless Velocity': median_dimless, 
            'CI Lower Bound': lower, 
            'CI Upper Bound': upper
        }
    else:
        # Make sure the velocity variable exists
        if velocity_variable not in group.columns:
            print(f"DEBUG - WARNING: '{velocity_variable}' not found in group columns")
            # Try to check for similar columns
            for col in group.columns:
                if velocity_variable.replace('_', '') in col.replace('_', '').lower():
                    print(f"DEBUG - Found similar column: '{col}', using this instead")
                    velocity_variable = col
                    break
            else:
                print(f"DEBUG - No suitable replacement for '{velocity_variable}' found")
                return pd.Series(result)
            
        # Drop any NaN values before bootstrapping
        valid_data = group[velocity_variable].dropna()
        
        if len(valid_data) == 0:
            print(f"DEBUG - No valid {velocity_variable} data")
            return pd.Series(result)
        
        if participant_weighting:
            # Group data by participant and calculate participant-level medians
            participant_medians = group.groupby('Participant')[velocity_variable].median().dropna()
            
            if len(participant_medians) == 0:
                print(f"DEBUG - No valid participant medians for {velocity_variable}")
                return pd.Series(result)
                
            # Bootstrap at the participant level
            for _ in range(n_iterations):
                # Resample participants with replacement
                sample = resample(participant_medians)
                medians.append(np.median(sample))
                
            # Calculate the overall median using participant-level medians
            median = np.median(participant_medians)
        else:
            # Original bootstrapping method at the measurement level
            for _ in range(n_iterations):
                sample = resample(valid_data)
                medians.append(np.median(sample))
                
            median = np.median(valid_data)
            
        lower = np.percentile(medians, (100 - ci_percentile) / 2)
        upper = np.percentile(medians, 100 - (100 - ci_percentile) / 2)
        
        if velocity_variable == 'Shear_Rate':
            result = {
                'Median Shear Rate': median, 
                'CI Lower Bound': lower, 
                'CI Upper Bound': upper
            }
        else:
            result = {
                'Median Velocity': median, 
                'CI Lower Bound': lower, 
                'CI Upper Bound': upper
            }
    
    # print(f"DEBUG - Returning result keys: {list(result.keys())}")
    return pd.Series(result)

# Function to calculate mean, standard error, and 95% CI
def calculate_mean_ci(group, ci_percentile = 95, dimensionless = False, 
                     velocity_variable = 'Corrected_Velocity', participant_weighting=False):
    """
    Calculates the mean, standard error, and 95% CI for a given group.
    If the group has a 'Dimensionless Velocity' column, it will use that column.
    Otherwise, it will use the 'Corrected_Velocity' or the 'Shear_Rate' column.

    Args:
        group (pd.DataFrame): The group to calculate the stats for.
        ci_percentile (int): The percentile to use for the CI.
        dimensionless (bool): Whether to return the stats in dimensionless units.
        participant_weighting (bool): If True, each participant will be weighted equally.

    Returns:
        pd.Series: A series containing the mean, lower bound, and upper bound of the CI.
    """
    if dimensionless:
        if participant_weighting:
            # Group by participant and get mean per participant
            participant_means = group.groupby('Participant')['Dimensionless Velocity'].mean()
            mean = participant_means.mean()
            sem = stats.sem(participant_means)
        else:
            mean = group['Dimensionless Velocity'].mean()
            sem = stats.sem(group['Dimensionless Velocity'])
            
        ci = 1.96 * sem
        return pd.Series({'Mean Dimensionless Velocity': mean, 'Lower Bound': mean - ci, 'Upper Bound': mean + ci})
    else:
        if participant_weighting:
            # Group by participant and get mean per participant
            participant_means = group.groupby('Participant')[velocity_variable].mean()
            mean = participant_means.mean()
            sem = stats.sem(participant_means)
        else:
            mean = group[velocity_variable].mean()
            sem = stats.sem(group[velocity_variable])
            
        ci = 1.96 * sem
        return pd.Series({'Mean Velocity': mean, 'Lower Bound': mean - ci, 'Upper Bound': mean + ci})


def adjust_brightness_of_colors(color_list, brightness_scale=0.1):
    """Adjusts the brightness (lightness) of a list of RGB colors.
    
    Args:
        color_list (list): A list of RGB colors to adjust.
        brightness_scale (float): The amount to adjust the brightness by.
            Positive values (e.g., 0.1) make colors brighter by increasing lightness.
            Negative values (e.g., -0.1) make colors darker by decreasing lightness.
            Values closer to zero make smaller adjustments, while larger values (e.g., 0.3)
            create more dramatic brightness changes. The value is clamped between 0 and 1.

    Returns:
        list: A list of adjusted RGB colors.
    """
    adjusted_colors = []
    for color in color_list:
        h, l, s = colorsys.rgb_to_hls(*color)
        l_new = max(0, min(1, l + brightness_scale))
        rgb_new = colorsys.hls_to_rgb(h, l_new, s)
        adjusted_colors.append(rgb_new)
    return adjusted_colors


def to_rgb(hex_color):
    """Converts a hex color to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3))

def create_monochromatic_palette(base_color, n_colors=5):
    """Creates a monochromatic palette based on the given color."""
    rgb = to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    
    colors = []
    # Increasing the spread for more distinct colors
    lightness_increment = 0.4 / (n_colors - 1)  # Adjust the 0.4 value to increase or decrease contrast
    for i in range(n_colors):
        l_new = max(0, min(1, l + (i - n_colors / 2) * lightness_increment))
        rgb_new = colorsys.hls_to_rgb(h, l_new, s)
        colors.append(rgb_new)
    # plot all the colors in the set:
    # for color in colors:
    #     plt.axhspan(0, 1, color=color)
    #     plt.show()
    return colors

def adjust_saturation_of_colors(color_list, saturation_scale=10):
    """Adjusts the saturation of a list of RGB colors."""
    adjusted_colors = []
    for color in color_list:
        h, l, s = colorsys.rgb_to_hls(*color)
        s_new = max(0, min(1, s + saturation_scale))
        rgb_new = colorsys.hls_to_rgb(h, l, s_new)
        adjusted_colors.append(rgb_new)
    return adjusted_colors

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

def plot_histogram(diameter_analysis_df, variable, pressure, age_groups, cutoff_percentile=95):
    """
    Plot a histogram of the specified variable for a given pressure, showing multiple age groups 
    on the same plot with median lines and a legend.
    
    Args:
        diameter_analysis_df: DataFrame containing the data to plot
        variable: String specifying which variable to plot ('Mean_Diameter', 'Pressure_Drop', 
                 'Corrected_Velocity', or 'Shear_Rate')
        pressure: Pressure value to plot
        age_groups: List of age groups to plot
        cutoff_percentile: Percentile value to use as upper cutoff for outliers (except Mean_Diameter)
    """
    setup_plotting_style()  
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'histograms')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a copy of the dataframe
    df = diameter_analysis_df.copy()
    
    # Define variable-specific settings
    variable_settings = {
        'Mean_Diameter': {
            'color': '#1f77b4',  # Blue
            'label': 'Mean Diameter (μm)',
            'title_prefix': 'Mean Diameter',
            'unit': 'μm',
            'apply_cutoff': False
        },
        'Pressure_Drop': {
            'color': '#d62728',  # Red
            'label': 'Pressure Drop per Length (mmHg/μm)',
            'title_prefix': 'Pressure Drop',
            'unit': 'mmHg/μm',
            'apply_cutoff': True
        },
        'Corrected_Velocity': {
            'color': '#2ca02c',  # Green
            'label': 'Corrected Velocity (μm/s)',
            'title_prefix': 'Velocity',
            'unit': 'μm/s',
            'apply_cutoff': True
        },
        'Shear_Rate': {
            'color': '#9467bd',  # Purple
            'label': 'Shear Rate (s⁻¹)',
            'title_prefix': 'Shear Rate',
            'unit': 's⁻¹',
            'apply_cutoff': True
        }
    }
    
    # Check if the variable is valid
    if variable not in variable_settings:
        raise ValueError(f"Invalid variable: {variable}. Valid options are: {list(variable_settings.keys())}")
    
    # Get settings for the specified variable
    settings = variable_settings[variable]
    
    # Calculate pressure drop if plotting Pressure_Drop
    if variable == 'Pressure_Drop':
        diameters = df['Mean_Diameter']
        velocities = df['Corrected_Velocity']
        viscosities = secomb_viscocity_fn(diameters)
        df['Pressure_Drop'] = pressure_drop_per_length(diameters, velocities, viscosities)
    
    
    
    # Apply cutoff if needed
    if settings['apply_cutoff']:
        upper_cutoff = df[variable].quantile(cutoff_percentile/100)
        df[variable] = df[variable].where(df[variable] <= upper_cutoff, upper_cutoff)
    else:
        upper_cutoff = df[variable].max()

    # Filter by pressure
    df = df[df['Pressure'] == pressure]
    
    # Create color palette
    color_palette = create_monochromatic_palette(base_color=settings['color'], n_colors=5)
    color_palette = adjust_brightness_of_colors(color_palette, brightness_scale=0.1)
    color_palette = [color_palette[4], color_palette[1]]
    
    # Create a single figure for both age groups
    plt.figure(figsize=(2.4, 2.0))
    
    # Process each age group
    for i, age_group in enumerate(age_groups):
        if age_group == '≤50':
            age_group_df = df[df['Age'] <= 50]
            display_name = '≤50 years'
        elif age_group == '>50':
            age_group_df = df[df['Age'] > 50]
            display_name = '>50 years'
        
        # Calculate median for this group
        median_value = age_group_df[variable].median()
        
        # Plot histogram with transparency
        plt.hist(age_group_df[variable], bins=20, density=True, alpha=0.5, color=color_palette[i], 
                label=f'{display_name} (n={len(age_group_df)})')
        
        # Add vertical line for median
        plt.axvline(x=median_value, color=color_palette[i], linestyle='--', 
                   label=f'Median {display_name}: {median_value:.2f} {settings["unit"]}')
    
    # Add labels and title
    plt.xlabel(settings['label'], fontproperties=source_sans)
    plt.ylabel('Frequency', fontproperties=source_sans)
    plt.title(f'{pressure} PSI - {settings["title_prefix"]} Histogram', fontproperties=source_sans)
    plt.xlim(0, upper_cutoff)
    
    # Add legend with smaller font size
    plt.legend(prop={'size': 5})
    
    plt.tight_layout()
    
    # Save the plot with a name that includes the variable
    safe_variable = variable.lower().replace('_', '')
    plt.savefig(os.path.join(output_dir, f'pressure_{pressure}_{safe_variable}_histogram_by_age.png'),
                dpi=600, bbox_inches='tight')
    plt.close()
    return 0

def plot_violin(diameter_analysis_df, variable, pressure, age_groups, cutoff_percentile=95):
    """
    Plot a violin plot of the specified variable for a given pressure,
    showing multiple age groups side by side with median markers.
    
    Args:
        diameter_analysis_df: DataFrame containing the data to plot
        variable: String specifying which variable to plot ('Mean_Diameter', 'Pressure_Drop', 
                 'Corrected_Velocity', or 'Shear_Rate')
        pressure: Pressure value to plot
        age_groups: List of age groups to plot
        cutoff_percentile: Percentile value to use as upper cutoff for outliers (except Mean_Diameter)
    """
    setup_plotting_style()  
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'violin_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a copy of the dataframe
    df = diameter_analysis_df.copy()
    
    # Define variable-specific settings (same as in plot_histogram)
    variable_settings = {
        'Mean_Diameter': {
            'color': '#1f77b4',  # Blue
            'label': 'Mean Diameter (μm)',
            'title_prefix': 'Mean Diameter',
            'unit': 'μm',
            'apply_cutoff': False
        },
        'Pressure_Drop': {
            'color': '#d62728',  # Red
            'label': 'Pressure Drop per Length (mmHg/μm)',
            'title_prefix': 'Pressure Drop',
            'unit': 'mmHg/μm',
            'apply_cutoff': True
        },
        'Corrected_Velocity': {
            'color': '#2ca02c',  # Green
            'label': 'Corrected Velocity (μm/s)',
            'title_prefix': 'Velocity',
            'unit': 'μm/s',
            'apply_cutoff': True
        },
        'Shear_Rate': {
            'color': '#9467bd',  # Purple
            'label': 'Shear Rate (s⁻¹)',
            'title_prefix': 'Shear Rate',
            'unit': 's⁻¹',
            'apply_cutoff': True
        }
    }
    
    # Check if the variable is valid
    if variable not in variable_settings:
        raise ValueError(f"Invalid variable: {variable}. Valid options are: {list(variable_settings.keys())}")
    
    # Get settings for the specified variable
    settings = variable_settings[variable]
    
    # Calculate pressure drop if plotting Pressure_Drop
    if variable == 'Pressure_Drop':
        diameters = df['Mean_Diameter']
        velocities = df['Corrected_Velocity']
        viscosities = secomb_viscocity_fn(diameters)
        df['Pressure_Drop'] = pressure_drop_per_length(diameters, velocities, viscosities)
    
    # Apply cutoff if needed
    if settings['apply_cutoff']:
        upper_cutoff = df[variable].quantile(cutoff_percentile/100)
        df[variable] = df[variable].where(df[variable] <= upper_cutoff, upper_cutoff)
    else:
        upper_cutoff = df[variable].max()

    # Filter by pressure
    df = df[df['Pressure'] == pressure]
    
    # Create color palette (same as in plot_histogram)
    color_palette = create_monochromatic_palette(base_color=settings['color'], n_colors=5)
    color_palette = adjust_brightness_of_colors(color_palette, brightness_scale=0.1)
    color_palette = [color_palette[4], color_palette[1]]
    
    # Create a figure 
    plt.figure(figsize=(2.4, 2.0))
    
    # Prepare data for violin plot
    plot_data = []
    labels = []
    colors = []
    positions = []
    
    # Process each age group
    for i, age_group in enumerate(age_groups):
        if age_group == '≤50':
            age_group_df = df[df['Age'] <= 50]
            display_name = '≤50 years'
        elif age_group == '>50':
            age_group_df = df[df['Age'] > 50]
            display_name = '>50 years'
        
        # Add data to lists
        if not age_group_df.empty:
            plot_data.append(age_group_df[variable])
            labels.append(f'{display_name}\n(n={len(age_group_df)})')
            colors.append(color_palette[i])
            positions.append(i+1)
    
    # Create violin plot
    violins = plt.violinplot(plot_data, positions=positions, showmeans=False, 
                          showmedians=True, showextrema=True)
    
    # Customize violins
    for i, pc in enumerate(violins['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Customize median lines
    violins['cmedians'].set_color('black')
    
    # Add individual data points
    for i, age_group in enumerate(age_groups):
        if age_group == '≤50':
            age_group_df = df[df['Age'] <= 50]
        elif age_group == '>50':
            age_group_df = df[df['Age'] > 50]
        
        # Add jittered data points
        if not age_group_df.empty:
            x = np.random.normal(i+1, 0.05, size=len(age_group_df))
            plt.scatter(x, age_group_df[variable], alpha=0.2, s=1, color=colors[i])
    
    # Customize plot
    plt.xticks(positions, labels, fontproperties=source_sans)
    plt.ylabel(settings['label'], fontproperties=source_sans)
    plt.title(f'{pressure} PSI - {settings["title_prefix"]} Distribution', fontproperties=source_sans)
    plt.ylim(0, upper_cutoff)
    
    # Add statistics table
    stats_text = []
    for i, age_group in enumerate(age_groups):
        if age_group == '≤50':
            age_group_df = df[df['Age'] <= 50]
            display_name = '≤50'
        elif age_group == '>50':
            age_group_df = df[df['Age'] > 50]
            display_name = '>50'
        
        if not age_group_df.empty:
            median_val = age_group_df[variable].median()
            mean_val = age_group_df[variable].mean()
            stats_text.append(f"{display_name}: median={median_val:.1f}, mean={mean_val:.1f}")
    
    # Try to do statistical comparison if we have two groups
    if len(plot_data) == 2 and len(plot_data[0]) > 0 and len(plot_data[1]) > 0:
        try:
            from scipy import stats
            # Mann-Whitney U test (non-parametric)
            u_stat, p_value = stats.mannwhitneyu(plot_data[0], plot_data[1])
            stats_text.append(f"p-value: {p_value:.3f}")
        except:
            pass
    
    # # Add stats as annotation
    # plt.annotate('\n'.join(stats_text), xy=(0.5, 0.02), xycoords='axes fraction',
    #            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    #            ha='center', va='bottom', fontsize=5)
    
    plt.tight_layout()
    
    # Save the plot
    safe_variable = variable.lower().replace('_', '')
    plt.savefig(os.path.join(output_dir, f'pressure_{pressure}_{safe_variable}_violin_by_age.png'),
                dpi=600, bbox_inches='tight')
    plt.close()
    return 0

def plot_boxnwhisker(diameter_analysis_df, variable, pressure, age_groups, cutoff_percentile=95):
    """
    Plot a box and whisker plot of the specified variable for a given pressure,
    showing multiple age groups side by side with statistical annotations.
    
    Args:
        diameter_analysis_df: DataFrame containing the data to plot
        variable: String specifying which variable to plot ('Mean_Diameter', 'Pressure_Drop', 
                 'Corrected_Velocity', or 'Shear_Rate')
        pressure: Pressure value to plot
        age_groups: List of age groups to plot
        cutoff_percentile: Percentile value to use as upper cutoff for outliers (except Mean_Diameter)
    """
    setup_plotting_style()  
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'boxplot_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a copy of the dataframe
    df = diameter_analysis_df.copy()
    
    # Define variable-specific settings (same as in plot_histogram)
    variable_settings = {
        'Mean_Diameter': {
            'color': '#1f77b4',  # Blue
            'label': 'Mean Diameter (μm)',
            'title_prefix': 'Mean Diameter',
            'unit': 'μm',
            'apply_cutoff': False
        },
        'Pressure_Drop': {
            'color': '#d62728',  # Red
            'label': 'Pressure Drop per Length (mmHg/μm)',
            'title_prefix': 'Pressure Drop',
            'unit': 'mmHg/μm',
            'apply_cutoff': True
        },
        'Corrected_Velocity': {
            'color': '#2ca02c',  # Green
            'label': 'Corrected Velocity (μm/s)',
            'title_prefix': 'Velocity',
            'unit': 'μm/s',
            'apply_cutoff': True
        },
        'Shear_Rate': {
            'color': '#9467bd',  # Purple
            'label': 'Shear Rate (s⁻¹)',
            'title_prefix': 'Shear Rate',
            'unit': 's⁻¹',
            'apply_cutoff': True
        }
    }
    
    # Check if the variable is valid
    if variable not in variable_settings:
        raise ValueError(f"Invalid variable: {variable}. Valid options are: {list(variable_settings.keys())}")
    
    # Get settings for the specified variable
    settings = variable_settings[variable]
    
    # Calculate pressure drop if plotting Pressure_Drop
    if variable == 'Pressure_Drop':
        diameters = df['Mean_Diameter']
        velocities = df['Corrected_Velocity']
        viscosities = secomb_viscocity_fn(diameters)
        df['Pressure_Drop'] = pressure_drop_per_length(diameters, velocities, viscosities)
    
    # Apply cutoff if needed
    if settings['apply_cutoff']:
        upper_cutoff = df[variable].quantile(cutoff_percentile/100)
        df[variable] = df[variable].where(df[variable] <= upper_cutoff, upper_cutoff)
    else:
        upper_cutoff = df[variable].max()

    # Filter by pressure
    df = df[df['Pressure'] == pressure]
    
    # Create color palette (same as in other plotting functions)
    color_palette = create_monochromatic_palette(base_color=settings['color'], n_colors=5)
    color_palette = adjust_brightness_of_colors(color_palette, brightness_scale=0.1)
    color_palette = [color_palette[4], color_palette[1]]
    
    # Create a figure 
    plt.figure(figsize=(2.4, 2.0))
    
    # Prepare data for box plot
    plot_data = []
    labels = []
    colors = []
    group_stats = []
    
    # Process each age group
    for i, age_group in enumerate(age_groups):
        if age_group == '≤50':
            age_group_df = df[df['Age'] <= 50]
            display_name = '≤50 years'
        elif age_group == '>50':
            age_group_df = df[df['Age'] > 50]
            display_name = '>50 years'
        
        # Add data to lists
        if not age_group_df.empty:
            plot_data.append(age_group_df[variable])
            labels.append(f'{display_name}\n(n={len(age_group_df)})')
            colors.append(color_palette[i])
            
            # Calculate statistics for this group
            group_stats.append({
                'median': age_group_df[variable].median(),
                'mean': age_group_df[variable].mean(),
                'q1': age_group_df[variable].quantile(0.25),
                'q3': age_group_df[variable].quantile(0.75),
                'n': len(age_group_df),
                'display_name': display_name.split(' ')[0]  # Just the age part
            })
    
    # Create box plot
    box_props = dict(linestyle='-', linewidth=0.8, color='black')
    whisker_props = dict(linestyle='-', linewidth=0.8, color='black')
    median_props = dict(linestyle='-', linewidth=1.5, color='black')
    cap_props = dict(linestyle='-', linewidth=0.8, color='black')
    flier_props = dict(marker='o', markerfacecolor='none', markeredgecolor='black', 
                    markersize=2, alpha=0.5, linewidth=0.5)
                    
    # Create the boxplot with custom properties
    bp = plt.boxplot(plot_data, labels=labels, patch_artist=True,
                  boxprops=box_props, whiskerprops=whisker_props,
                  medianprops=median_props, capprops=cap_props,
                  flierprops=flier_props, showfliers=True)
                  
    # Customize box colors
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])
        box.set_alpha(0.7)
        
    # Add individual data points with jitter
    for i, data in enumerate(plot_data):
        # Add jittered points around position i+1
        x = np.random.normal(i+1, 0.05, size=len(data))
        plt.scatter(x, data, alpha=0.2, s=1, color=colors[i])
    
    # Add labels and title
    plt.ylabel(settings['label'], fontproperties=source_sans)
    plt.title(f'{pressure} PSI - {settings["title_prefix"]} Box Plot', fontproperties=source_sans)
    plt.ylim(0, upper_cutoff)
    
    # Add statistics table
    stats_text = []
    for stat in group_stats:
        stats_text.append(f"{stat['display_name']}: median={stat['median']:.1f}, IQR=[{stat['q1']:.1f}, {stat['q3']:.1f}]")
    
    # Try to do statistical comparison if we have two groups
    if len(plot_data) == 2 and len(plot_data[0]) > 0 and len(plot_data[1]) > 0:
        try:
            from scipy import stats
            # Mann-Whitney U test (non-parametric)
            u_stat, p_value = stats.mannwhitneyu(plot_data[0], plot_data[1])
            # Add significance markers on the plot
            significance_level = "ns"
            if p_value < 0.001:
                significance_level = "***"
            elif p_value < 0.01:
                significance_level = "**"
            elif p_value < 0.05:
                significance_level = "*"
                
            # Add the p-value to the stats text
            stats_text.append(f"p-value: {p_value:.3f} {significance_level}")
            
            # Draw a line with significance indicator if significant
            if p_value < 0.05:
                y_max = max([max(data) for data in plot_data])
                y_pos = upper_cutoff * 0.95
                plt.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=0.8)
                plt.text(1.5, y_pos * 1.02, significance_level, ha='center', va='bottom', fontsize=8)
        except Exception as e:
            print(f"Error performing statistical test: {e}")
            pass
    
    # # Add stats as annotation
    # plt.annotate('\n'.join(stats_text), xy=(0.5, 0.02), xycoords='axes fraction',
    #            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    #            ha='center', va='bottom', fontsize=5)
    
    plt.tight_layout()
    
    # Save the plot
    safe_variable = variable.lower().replace('_', '')
    plt.savefig(os.path.join(output_dir, f'pressure_{pressure}_{safe_variable}_boxplot_by_age.png'),
                dpi=600, bbox_inches='tight')
    plt.close()
    return 0

def plot_cdf(diameter_analysis_df, variable, pressure, age_groups, cutoff_percentile=95):
    """
    Plot a cumulative distribution function (CDF) of the specified variable for a given pressure,
    showing multiple age groups on the same plot with median lines and a legend.
    
    Args:
        diameter_analysis_df: DataFrame containing the data to plot
        variable: String specifying which variable to plot ('Mean_Diameter', 'Pressure_Drop', 
                 'Corrected_Velocity', or 'Shear_Rate')
        pressure: Pressure value to plot
        age_groups: List of age groups to plot
        cutoff_percentile: Percentile value to use as upper cutoff for outliers (except Mean_Diameter)
    """
    setup_plotting_style()  
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'cdf_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a copy of the dataframe
    df = diameter_analysis_df.copy()
    
    # Define variable-specific settings (same as in plot_histogram)
    variable_settings = {
        'Mean_Diameter': {
            'color': '#1f77b4',  # Blue
            'label': 'Mean Diameter (μm)',
            'title_prefix': 'Mean Diameter',
            'unit': 'μm',
            'apply_cutoff': False
        },
        'Pressure_Drop': {
            'color': '#d62728',  # Red
            'label': 'Pressure Drop per Length (mmHg/μm)',
            'title_prefix': 'Pressure Drop',
            'unit': 'mmHg/μm',
            'apply_cutoff': True
        },
        'Corrected_Velocity': {
            'color': '#2ca02c',  # Green
            'label': 'Corrected Velocity (μm/s)',
            'title_prefix': 'Velocity',
            'unit': 'μm/s',
            'apply_cutoff': True
        },
        'Shear_Rate': {
            'color': '#9467bd',  # Purple
            'label': 'Shear Rate (s⁻¹)',
            'title_prefix': 'Shear Rate',
            'unit': 's⁻¹',
            'apply_cutoff': True
        }
    }
    
    # Check if the variable is valid
    if variable not in variable_settings:
        raise ValueError(f"Invalid variable: {variable}. Valid options are: {list(variable_settings.keys())}")
    
    # Get settings for the specified variable
    settings = variable_settings[variable]
    
    # Calculate pressure drop if plotting Pressure_Drop
    if variable == 'Pressure_Drop':
        diameters = df['Mean_Diameter']
        velocities = df['Corrected_Velocity']
        viscosities = secomb_viscocity_fn(diameters)
        df['Pressure_Drop'] = pressure_drop_per_length(diameters, velocities, viscosities)
    
    # Apply cutoff if needed
    if settings['apply_cutoff']:
        upper_cutoff = df[variable].quantile(cutoff_percentile/100)
        df[variable] = df[variable].where(df[variable] <= upper_cutoff, upper_cutoff)
    else:
        upper_cutoff = df[variable].max()

    # Filter by pressure
    df = df[df['Pressure'] == pressure]
    
    # Create color palette (same as in plot_histogram)
    color_palette = create_monochromatic_palette(base_color=settings['color'], n_colors=5)
    color_palette = adjust_brightness_of_colors(color_palette, brightness_scale=0.1)
    color_palette = [color_palette[4], color_palette[1]]
    
    # Create a single figure for both age groups
    plt.figure(figsize=(2.4, 2.0))
    
    # Process each age group
    for i, age_group in enumerate(age_groups):
        if age_group == '≤50':
            age_group_df = df[df['Age'] <= 50]
            display_name = '≤50 years'
        elif age_group == '>50':
            age_group_df = df[df['Age'] > 50]
            display_name = '>50 years'
        
        # Calculate median for this group
        median_value = age_group_df[variable].median()
        
        # Calculate CDF
        sorted_data = np.sort(age_group_df[variable])
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Plot CDF
        plt.plot(sorted_data, cdf, '-', linewidth=1, color=color_palette[i], 
                label=f'{display_name} (n={len(age_group_df)})')
        
        # Add vertical line for median
        plt.axvline(x=median_value, color=color_palette[i], linestyle='--', 
                  label=f'Median {display_name}: {median_value:.2f} {settings["unit"]}')
    
    # Add reference line at 0.5 probability
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel(settings['label'], fontproperties=source_sans)
    plt.ylabel('Cumulative Probability', fontproperties=source_sans)
    plt.title(f'{pressure} PSI - {settings["title_prefix"]} CDF', fontproperties=source_sans)
    plt.xlim(0, upper_cutoff)
    plt.ylim(0, 1)
    
    # Add legend with smaller font size
    plt.legend(prop={'size': 5})
    
    plt.tight_layout()
    
    # Save the plot with a name that includes the variable
    safe_variable = variable.lower().replace('_', '')
    plt.savefig(os.path.join(output_dir, f'pressure_{pressure}_{safe_variable}_cdf_by_age.png'),
                dpi=600, bbox_inches='tight')
    plt.close()
    return 0

def plot_cdf_old(data, subsets, labels=['Entire Dataset', 'Subset'], title='CDF Comparison', 
             write=False, normalize=False, variable = 'Age', log = True):
    """
    Plots the CDF of the entire dataset and the inputted subsets.

    Args:
        data (array-like): The entire dataset
        subsets (list of array-like): The subsets to be compared
        labels (list of str): The labels for the entire dataset and the subsets
        title (str): The title of the plot
        write (bool): Whether to write the plot to a file
        normalize (bool): Whether to normalize the CDF
    
    Returns:
        0 if successful, 1 if no subsets provided
    """
    plt.close()
    # Set up style and font
    sns.set_style("whitegrid")
    source_sans = get_source_sans_font()

    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

    # Set color palette based on variable
    if variable == 'Age':
        base_color = '#1f77b4'##3FCA54''BDE4A7 #A1E5AB
    elif variable == 'SYS_BP':
        base_color = '2ca02c'#80C6C3 #ff7f0e
    elif variable == 'Sex':
        base_color = '674F92'#947EB0#2ca02c#CAC0D89467bd
    elif variable == 'Individual':
        base_color = '#1f77b4'
        individual_color = '#6B0F1A' #'#ff7f0e'
    elif variable == 'Diabetes_plot':
        base_color = '#ff7f0e' 
    elif variable == 'Hypertension_plot':
        base_color = '#d62728'
    else:
        raise ValueError(f"Unsupported variable: {variable}")

    palette = create_monochromatic_palette(base_color)
    # palette = adjust_saturation_of_colors(palette, saturation_scale=1.3)
    palette = adjust_brightness_of_colors(palette, brightness_scale=.2)
    sns.set_palette(palette)

    if not subsets:
        return 1

    fig, ax = plt.subplots(figsize=(2.4, 2.0))

    if log:
        data = data+1
        for i in range(len(subsets)):
            subsets[i]= subsets[i]+1
   
    # Plot main dataset
    x, y = calculate_cdf(data, normalize)
    ax.plot(x, y, label=labels[0])
    

    # Plot subsets
    for i in  range(len(subsets)):
        if i == 0:
            i_color = 0
            dot_color = 0
        elif i == 1:
            i_color = 3
            dot_color = 2
        elif i == 2:
            i_color = 4
            dot_color=3
        x, y = calculate_cdf(subsets[i], normalize)
        if variable == 'Individual':
            ax.plot(x, y, label=labels[i+1], linestyle='--', color=individual_color)
        else:
            ax.plot(x, y, label=labels[i+1], linestyle='--', color=palette[i_color])

    ax.set_ylabel('CDF', fontproperties=source_sans)
    if log:
        ax.set_xlabel('Velocity + 1 (um/s)' if 'Pressure' not in title else 'Pressure (psi)', fontproperties=source_sans)
    else:
        ax.set_xlabel('Velocity (um/s)' if 'Pressure' not in title else 'Pressure (psi)', fontproperties=source_sans)
    ax.set_title(title, fontsize=8, fontproperties=source_sans)

    if log:
        ax.set_xscale('log')
        # ax.set_xticklabels([1, 10, 100, 1000, 5000])
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    # Adjust legend
    ax.legend(loc='lower right', bbox_to_anchor=(1, 0.01), prop=source_sans, fontsize=6)
    
    ax.grid(True, linewidth=0.3)
    fig.set_dpi(300)
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    if write:
        save_plot(fig, title, dpi=300)
    else:
        plt.show()
    if write:
        plt.close()

    return 0


def plot_CI_multiple_bands(df, thresholds=[29, 49], variable='Age', method='bootstrap', 
                 n_iterations=1000, ci_percentile=99.5, write=True, dimensionless=False, 
                 video_median=False, log_scale=False, 
                 velocity_variable='Corrected Velocity'):
    """Creates a confidence interval plot with multiple age bands based on specified thresholds.
    
    Creates a line plot with confidence intervals showing velocity distributions for multiple
    age groups defined by the provided thresholds. By default creates three groups:
    under 30, 30-49, and 50+.
    
    Args:
        df: DataFrame containing the data to plot
        thresholds: List of age thresholds to define groups (e.g., [29, 49] creates groups <30, 30-49, 50+)
        variable: Column name to group by (typically 'Age')
        method: Method for calculating confidence intervals ('bootstrap' or 'mean')
        n_iterations: Number of bootstrap iterations if using bootstrap method
        ci_percentile: Confidence interval percentile (e.g., 95, 99, 99.5)
        write: Whether to save the plot to disk
        dimensionless: Whether to use dimensionless values
        video_median: Whether to use video median velocity instead of corrected velocity
        log_scale: Whether to use log scale for y-axis
        velocity_variable: Name of the velocity column to use
        
    Returns:
        0 if successful, 1 if error occurred
    
    Example:
        >>> plot_CI_multiple_bands(data, thresholds=[29, 49], write=True)
        0
    """
    # Get Source Sans font
    source_sans = get_source_sans_font()
    
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
    
    # Sort thresholds to ensure correct ordering
    thresholds = sorted(thresholds)
    
    # Create a copy of the dataframe
    plot_df = df.copy()
    
    # Determine which velocity variable to use
    if video_median:
        velocity_variable = 'Video_Median_Velocity'
        
    # Create age group labels based on thresholds
    group_labels = []
    
    # First group is below the first threshold
    group_labels.append(f'<{thresholds[0]+1}')
    
    # Middle groups are between thresholds
    for i in range(len(thresholds) - 1):
        group_labels.append(f'{thresholds[i]+1}-{thresholds[i+1]}')
    
    # Last group is above the last threshold
    group_labels.append(f'≥{thresholds[-1]+1}')
    
    # Create age groups based on thresholds
    plot_df['Age_Group'] = pd.cut(
        plot_df['Age'],
        bins=[0] + thresholds + [200],  # Add 0 at start and large value at end
        labels=group_labels,
        include_lowest=True
    )
    
    # Create a monochromatic color palette regardless of number of groups
    # Use a blue base color with increasing intensity
    base_color = '#1f77b4'  # Blue base
    num_groups = 10
    base_colors = create_monochromatic_palette(base_color, 10)
    
    # Get unique pressures and sort them
    pressures = sorted(plot_df['Pressure'].unique())
    
    # Create a dictionary to store results for each age group and pressure
    results = {}
    
    # Calculate confidence intervals for each age group and pressure
    for group_label in group_labels:
        results[group_label] = {'pressures': [], 'medians': [], 'lower': [], 'upper': []}
        
        for pressure in pressures:
            # Get data for this group and pressure
            group_data = plot_df[(plot_df['Age_Group'] == group_label) & 
                                (plot_df['Pressure'] == pressure)]
            
            if len(group_data) < 3:
                continue  # Skip if not enough data
                
            # Calculate confidence intervals
            if method == 'bootstrap':
                ci_data = calculate_median_ci(
                    group_data, 
                    n_iterations=n_iterations, 
                    ci_percentile=ci_percentile,
                    dimensionless=dimensionless,
                    velocity_variable=velocity_variable
                )
            else:
                ci_data = calculate_mean_ci(
                    group_data,
                    ci_percentile=ci_percentile,
                    dimensionless=dimensionless,
                    velocity_variable=velocity_variable
                )
            
            # Skip empty results
            if ci_data is None:
                continue
                
            median, lower, upper = ci_data
            
            # Store results
            results[group_label]['pressures'].append(pressure)
            results[group_label]['medians'].append(median)
            results[group_label]['lower'].append(lower)
            results[group_label]['upper'].append(upper)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    
    # Dictionary to store handles for legend
    legend_handles = {}
    
    # Plot each age group
    for group_idx, group_label in enumerate(group_labels):
        group_results = results[group_label]
        
        # Skip if no data for this group
        if not group_results['pressures']:
            continue
        
        # Get color for this group
        color = base_colors[group_idx*(3)]
        
        # Plot median line
        line, = ax.plot(
            group_results['pressures'], 
            group_results['medians'], 
            'o-', 
            color=color, 
            markersize=3,
            linewidth=0.75,
            label=group_label
        )
        
        # Plot confidence interval
        ax.fill_between(
            group_results['pressures'],
            group_results['lower'],
            group_results['upper'],
            alpha=0.2,
            color=color
        )
        
        # Store handle for legend
        legend_handles[group_label] = line
    
    # Set labels and title
    if source_sans:
        ax.set_xlabel('Pressure (PSI)', fontproperties=source_sans)
        if dimensionless:
            ax.set_ylabel('Dimensionless Velocity', fontproperties=source_sans)
        else:
            ax.set_ylabel('Velocity (µm/s)', fontproperties=source_sans)
        ax.set_title('Velocity by Pressure and Age Group', fontproperties=source_sans)
    else:
        ax.set_xlabel('Pressure (PSI)')
        if dimensionless:
            ax.set_ylabel('Dimensionless Velocity')
        else:
            ax.set_ylabel('Velocity (µm/s)')
        ax.set_title('Velocity by Pressure and Age Group')
    
    # Set log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Create legend with group labels
    if legend_handles:
        legend_items = [(handle, label) for label, handle in legend_handles.items()]
        # Sort by label to ensure consistent order
        legend_items.sort(key=lambda x: x[1])
        handles, labels = zip(*legend_items)
        
        if source_sans:
            ax.legend(handles, labels, prop=source_sans)
        else:
            ax.legend(handles, labels)
    
    plt.tight_layout()
    
    # Save figure if requested
    if write:
        # Create output directory
        from src.config import PATHS
        cap_flow_path = PATHS['cap_flow']
        output_dir = os.path.join(cap_flow_path, 'results', 'CI_plots')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename based on parameters
        filename = f'CI_multi_age_bands_{"-".join(str(t) for t in thresholds)}'
        if video_median:
            filename += '_video_median'
        if dimensionless:
            filename += '_dimensionless'
        if log_scale:
            filename += '_log'
        
        # Save as PNG and PDF
        plt.savefig(os.path.join(output_dir, f'{filename}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{filename}.pdf'), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()
    return 0

def plot_participant_velocity_profiles(df, method='bootstrap', n_iterations=1000, 
                                      ci_percentile=95, write=True, dimensionless=False, 
                                      log_scale=False, velocity_variable='Corrected Velocity',
                                      save_dir=None, filename_prefix='participant_profiles'):
    """
    Creates velocity profiles for each participant by averaging their velocities at each pressure,
    then plots confidence intervals for all participants.
    
    Args:
        df (pd.DataFrame): The dataframe containing the velocity data.
        method (str): The method to use for the CI. Example: 'bootstrap', 'mean'
        n_iterations (int): The number of iterations to use for bootstrap CI.
        ci_percentile (float): The percentile to use for the CI.
        write (bool): Whether to write the plot to a file.
        dimensionless (bool): Whether to use dimensionless values.
        log_scale (bool): Whether to use a log scale for the y-axis.
        velocity_variable (str): The velocity variable to use.
        save_dir (str): Directory to save plots. If None, uses default.
        filename_prefix (str): Prefix for saved filenames.
        
    Returns:
        pd.DataFrame: The participant profiles dataframe, containing averaged velocities
                     for each participant at each pressure.
    """
    # Set up style and font
    sns.set_style("whitegrid")
    source_sans = get_source_sans_font()
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()
    
    # Get unique pressures and sort them
    pressures = sorted(plot_df['Pressure'].unique())
    
    # Create participant velocity profiles by averaging velocities at each pressure
    # This avoids giving more weight to participants with more measurements
    participant_profiles = []
    
    # Get unique participant IDs
    participants = plot_df['Participant'].unique()
    
    # For each participant, calculate the average velocity at each pressure
    for participant in participants:
        participant_data = plot_df[plot_df['Participant'] == participant]
        
        # Group by pressure and calculate mean velocity
        profile = participant_data.groupby('Pressure')[velocity_variable].mean().reset_index()
        profile['Participant'] = participant
        
        # Add any additional participant metadata (take first occurrence)
        metadata_cols = ['Age', 'Sex', 'SET', 'Diabetes', 'Hypertension', 'SYS_BP']
        for col in metadata_cols:
            if col in participant_data.columns:
                profile[col] = participant_data[col].iloc[0]
        
        participant_profiles.append(profile)
    
    # Combine all participant profiles into one dataframe
    if participant_profiles:
        profiles_df = pd.concat(participant_profiles, ignore_index=True)
    else:
        print("Error: No participant profiles created. Check input data.")
        return None
    
    # Create figure for plotting
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    
    # Calculate statistics for each pressure
    stats_df = pd.DataFrame(columns=['Pressure', 'median', 'lower', 'upper'])
    
    for pressure in pressures:
        # Filter data for this pressure
        pressure_data = profiles_df[profiles_df['Pressure'] == pressure]
        
        # Skip if not enough data
        if len(pressure_data) < 3:
            print(f"Warning: Not enough data for pressure {pressure}, skipping.")
            continue
        
        # Calculate CI based on method
        if method == 'bootstrap':
            ci_data = calculate_median_ci(
                pressure_data, 
                n_iterations=n_iterations, 
                ci_percentile=ci_percentile,
                dimensionless=dimensionless,
                velocity_variable=velocity_variable
            )
        else:
            ci_data = calculate_mean_ci(
                pressure_data,
                ci_percentile=ci_percentile,
                dimensionless=dimensionless,
                velocity_variable=velocity_variable
            )
        
        # Skip if CI calculation failed
        if ci_data is None:
            print(f"Warning: CI calculation failed for pressure {pressure}, skipping.")
            continue
        
        median, lower, upper = ci_data
        
        # Add to stats dataframe
        stats_df = pd.concat([stats_df, pd.DataFrame({
            'Pressure': [pressure],
            'median': [median],
            'lower': [lower],
            'upper': [upper]
        })], ignore_index=True)
    
    # Sort stats by pressure
    stats_df = stats_df.sort_values('Pressure')
    
    # Plot the confidence intervals
    y_col = 'median'
    lower_col = 'lower'
    upper_col = 'upper'
    
    # Plot the main line
    ax.plot(stats_df['Pressure'], stats_df[y_col], '-o', color='blue', markersize=2)
    
    # Add the confidence interval as a shaded region
    ax.fill_between(stats_df['Pressure'], stats_df[lower_col], stats_df[upper_col], 
                   alpha=0.4, color='blue')
    
    # Add log scale if requested
    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
    # Set labels and title
    ax.set_xlabel('Pressure (psi)', fontproperties=source_sans)
    
    if dimensionless:
        ax.set_ylabel('Dimensionless Velocity', fontproperties=source_sans)
        ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Participant Velocity Profiles with {ci_percentile}% CI', 
                  fontproperties=source_sans, fontsize=8)
    else:
        ax.set_ylabel('Velocity (um/s)', fontproperties=source_sans)
        ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Participant Velocity Profiles with {ci_percentile}% CI', 
                  fontproperties=source_sans, fontsize=8)
    
    # Add annotation with sample size
    sample_size = len(participants)
    ax.annotate(f'n = {sample_size} participants', xy=(0.02, 0.95), xycoords='axes fraction',
               fontsize=5, fontproperties=source_sans)
    
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()
    
    # Save the figure if requested
    if write:
        if save_dir is None:
            # Use a default directory if none provided
            save_dir = os.path.join(PATHS.get('cap_flow', '.'), 'results', 'velocity_profiles')
            os.makedirs(save_dir, exist_ok=True)
        
        # Create filename with parameters
        filename = f"{filename_prefix}_{method}_ci{ci_percentile}"
        if dimensionless:
            filename += "_dimensionless"
        if log_scale:
            filename += "_log"
        
        # Save as PNG and PDF
        plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f"{filename}.pdf"), bbox_inches='tight')
    
    plt.close()
    
    return profiles_df

def plot_participant_velocity_profiles_by_group(df, variable='Age', method='bootstrap', n_iterations=1000,
                                              ci_percentile=95, write=True, dimensionless=False,
                                              log_scale=False, velocity_variable='Corrected Velocity',
                                              save_dir=None, filename_prefix=None):
    """
    Creates velocity profiles for each participant by averaging their velocities at each pressure,
    then plots confidence intervals comparing different groups (by age, diabetes, etc).
    
    Args:
        df (pd.DataFrame): The dataframe containing the velocity data.
        variable (str): The variable to group by. Example: 'Age', 'Sex', 'SYS_BP', 'Diabetes', 'Hypertension', 'Set_affected'
        method (str): The method to use for the CI. Example: 'bootstrap', 'mean'
        n_iterations (int): The number of iterations to use for bootstrap CI.
        ci_percentile (float): The percentile to use for the CI.
        write (bool): Whether to write the plot to a file.
        dimensionless (bool): Whether to use dimensionless values.
        log_scale (bool): Whether to use a log scale for the y-axis.
        velocity_variable (str): The velocity variable to use.
        save_dir (str): Directory to save plots. If None, uses default.
        filename_prefix (str): Prefix for saved filenames. If None, uses the variable name.
        
    Returns:
        pd.DataFrame: The participant profiles dataframe, containing averaged velocities
                     for each participant at each pressure, with group labels.
    """
    # Set up style and font
    sns.set_style("whitegrid")
    source_sans = get_source_sans_font()
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()
    
    # Set color palette and conditions based on variable (similar to plot_CI function)
    if variable == 'Age':
        base_color = '#1f77b4'
        conditions = [df[variable] <= 29, df[variable] > 29]
        choices = ['≤29', '>29']
    elif variable == 'SYS_BP':
        base_color = '2ca02c'
        conditions = [df[variable] < 120, df[variable] >= 120]
        choices = ['<120', '≥120']
    elif variable == 'Sex':
        base_color = '674F92'
        conditions = [df[variable] == 'M', df[variable] == 'F']
        choices = ['Male', 'Female']
    elif variable == 'Diabetes':
        base_color = 'ff7f0e'
        conditions = [df['SET'] == 'set01', df['SET'] == 'set03']
        choices = ['Control', 'Diabetic']
    elif variable == 'Hypertension':
        base_color = 'd62728'
        conditions = [df['SET'] == 'set01', df['SET'] == 'set02']
        choices = ['Control', 'Hypertensive']
    elif variable == 'Set_affected':
        base_color = '#00CED1'  # sky blue
        conditions = [df['SET'] == 'set01', df['Set_affected'] == 'set04']
        choices = ['Control', 'Affected']
    else:
        raise ValueError(f"Unsupported variable: {variable}")

    # Create color palette
    palette = create_monochromatic_palette(base_color)
    palette = adjust_brightness_of_colors(palette, brightness_scale=.1)
    
    # Ensure consistent coloring by using only two colors
    control_color = palette[4]
    condition_color = palette[1]
    
    # Add group column to the dataframe
    group_col = f'{variable} Group'
    plot_df[group_col] = np.select(conditions, choices, default='Unknown')
    
    # Print unique values for debugging
    print(f"Unique values in {group_col}: {plot_df[group_col].unique()}")
    
    # Create participant velocity profiles by averaging velocities at each pressure
    # This avoids giving more weight to participants with more measurements
    participant_profiles = []
    
    # Get unique participant IDs
    participants = plot_df['Participant'].unique()
    
    # For each participant, calculate the average velocity at each pressure
    for participant in participants:
        participant_data = plot_df[plot_df['Participant'] == participant]
        
        # Group by pressure and calculate mean velocity
        profile = participant_data.groupby('Pressure')[velocity_variable].mean().reset_index()
        profile['Participant'] = participant
        
        # Add any additional participant metadata (take first occurrence)
        metadata_cols = ['Age', 'Sex', 'SET', 'Diabetes', 'Hypertension', 'SYS_BP', group_col, 'Set_affected']
        for col in metadata_cols:
            if col in participant_data.columns:
                # Use the most common value for the group column (in case there are inconsistencies)
                if col == group_col:
                    profile[col] = participant_data[col].mode().iloc[0]
                else:
                    profile[col] = participant_data[col].iloc[0]
        
        participant_profiles.append(profile)
    
    # Combine all participant profiles into one dataframe
    if participant_profiles:
        profiles_df = pd.concat(participant_profiles, ignore_index=True)
    else:
        print("Error: No participant profiles created. Check input data.")
        return None
    
    # Create figure for plotting
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    
    # Get unique pressures and sort them
    pressures = sorted(profiles_df['Pressure'].unique())
    
    # Calculate stats for each group and pressure
    stats_df = pd.DataFrame()
    
    for group_name in choices:
        group_stats = []
        
        # Filter profiles for this group
        group_profiles = profiles_df[profiles_df[group_col] == group_name]
        
        # Skip if not enough participants in this group
        if len(group_profiles['Participant'].unique()) < 2:
            print(f"Warning: Not enough participants in group {group_name}, skipping.")
            continue
            
        for pressure in pressures:
            # Get data for this pressure
            pressure_data = group_profiles[group_profiles['Pressure'] == pressure]
            
            # Skip if not enough data
            if len(pressure_data) < 3:
                continue
                
            # Calculate CI based on method
            if method == 'bootstrap':
                ci_data = calculate_median_ci(
                    pressure_data, 
                    n_iterations=n_iterations, 
                    ci_percentile=ci_percentile,
                    dimensionless=dimensionless,
                    velocity_variable=velocity_variable
                )
            else:
                ci_data = calculate_mean_ci(
                    pressure_data,
                    ci_percentile=ci_percentile,
                    dimensionless=dimensionless,
                    velocity_variable=velocity_variable
                )
                
            # Skip if calculation failed
            if ci_data is None:
                continue
                
            median, lower, upper = ci_data
            
            # Add to group stats
            group_stats.append({
                'Group': group_name,
                'Pressure': pressure,
                'median': median,
                'lower': lower,
                'upper': upper
            })
            
        # Add group stats to main stats dataframe
        if group_stats:
            group_stats_df = pd.DataFrame(group_stats)
            stats_df = pd.concat([stats_df, group_stats_df], ignore_index=True)
    
    # Determine column names for plotting based on method and dimensionless flag
    y_col = 'median'
    lower_col = 'lower'
    upper_col = 'upper'
    
    # Plot for each group
    for i, group_name in enumerate(choices):
        group_data = stats_df[stats_df['Group'] == group_name]
        
        # Skip if no data for this group
        if group_data.empty:
            print(f"Warning: No valid data for group {group_name}")
            continue
            
        # Sort by pressure
        group_data = group_data.sort_values('Pressure')
        
        # Plot the main line with error bars
        color = control_color if i == 0 else condition_color
        
        ax.errorbar(
            group_data['Pressure'], 
            group_data[y_col],
            yerr=[group_data[y_col] - group_data[lower_col], group_data[upper_col] - group_data[y_col]],
            label=group_name, 
            fmt='-o', 
            markersize=2, 
            color=color
        )
        
        # Add the confidence interval as a shaded region
        ax.fill_between(
            group_data['Pressure'], 
            group_data[lower_col], 
            group_data[upper_col],
            alpha=0.4, 
            color=color
        )
    
    # Add log scale if requested
    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
    # Create legend handles with consistent colors
    legend_handles = [
        mpatches.Patch(color=control_color, label=f'{choices[0]} group', alpha=0.6),
        mpatches.Patch(color=condition_color, label=f'{choices[1]} group', alpha=0.6)
    ]
    
    # Set labels and title
    ax.set_xlabel('Pressure (psi)', fontproperties=source_sans)
    
    # Calculate participant counts for each group
    group_counts = profiles_df.groupby(group_col)['Participant'].nunique()
    count_str = ", ".join([f"{grp}: n={cnt}" for grp, cnt in group_counts.items() if grp in choices])
    
    # Set appropriate labels based on configuration
    if dimensionless:
        ax.set_ylabel('Dimensionless Velocity', fontproperties=source_sans)
        ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Participant Profiles by {variable} ({count_str})', 
                    fontproperties=source_sans, fontsize=8)
    else:
        ax.set_ylabel('Velocity (um/s)', fontproperties=source_sans)
        ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Participant Profiles by {variable} ({count_str})', 
                    fontproperties=source_sans, fontsize=8)
    
    # Add legend
    if source_sans:
        ax.legend(handles=legend_handles, prop=source_sans)
    else:
        ax.legend(handles=legend_handles)
    
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()
    
    # Save the figure if requested
    if write:
        if save_dir is None:
            # Use a default directory if none provided
            save_dir = os.path.join(PATHS.get('cap_flow', '.'), 'results', 'velocity_profiles')
            os.makedirs(save_dir, exist_ok=True)
        
        # Create filename with parameters
        if filename_prefix is None:
            filename_prefix = f"participant_profiles_{variable.lower()}"
            
        filename = f"{filename_prefix}_{method}_ci{ci_percentile}"
        if dimensionless:
            filename += "_dimensionless"
        if log_scale:
            filename += "_log"
        
        # Save as PNG and PDF
        plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f"{filename}.pdf"), bbox_inches='tight')
    
    plt.close()
    
    return profiles_df

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

# ----------------------------------------------
# Classic CDF utilities (ported from legacy scripts)
# ----------------------------------------------

def _calc_norm_cdfs(data):
    """Return the mean empirical CDF across an iterable of 1-D arrays.

    This helper is used when *normalize=True* in :func:`calculate_cdf`.
    Each element of *data* is treated as an individual sample (e.g. a
    participant).  The empirical CDF is computed for every sample and the
    resulting probability values are averaged.
    """
    cdfs = []
    for sample in data:
        sample_sorted = np.sort(sample)
        p_sample = np.arange(len(sample)) / (len(sample) - 1)
        cdfs.append(np.vstack([sample_sorted, p_sample]))
    cdfs = np.array(cdfs)
    return np.mean(cdfs, axis=0)


def calculate_cdf(data, normalize: bool = False):
    """Compute the empirical cumulative distribution function (CDF).

    Parameters
    ----------
    data : array-like or iterable of array-like
        Input data.  When *normalize* is *False* this should be a 1-D array of
        values.  When *normalize* is *True* this should be an iterable of
        arrays (for example a list where each element corresponds to one
        participant).  The average CDF across the arrays is returned.
    normalize : bool, default False
        Whether to average CDFs across the outer dimension of *data*.

    Returns
    -------
    tuple
        A tuple ``(x, y)`` where *x* is the sorted data and *y* is the CDF
        probabilities.
    """
    if normalize:
        return _calc_norm_cdfs(data)

    sorted_data = np.sort(data)
    p = np.linspace(0, 1, len(sorted_data))
    return sorted_data, p


def plot_cdf_basic(
    data,
    subsets,
    labels,
    title: str = "CDF Comparison",
    write: bool = False,
    normalize: bool = False,
    variable: str = "Age",
    log: bool = True,
):
    """Legacy CDF plotting routine preserved for backward compatibility.

    This function reproduces the visual style of the original ``plot_cdf``
    implementation that lived in *plot_big.py*.  It is intended for ad-hoc
    exploratory graphics and small figure generation scripts rather than the
    more generic utilities above.
    """

    plt.close()
    sns.set_style("whitegrid")
    source_sans = get_source_sans_font()

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5,
            "lines.linewidth": 0.5,
        }
    )

    # Base colour selection mirrors the original behaviour
    if variable == "Age":
        base_color = "#1f77b4"
    elif variable == "SYS_BP":
        base_color = "2ca02c"
    elif variable == "Sex":
        base_color = "674F92"
    elif variable == "Individual":
        base_color = "#1f77b4"
        individual_color = "#6B0F1A"
    elif variable == "Diabetes_plot":
        base_color = "#ff7f0e"
    elif variable == "Hypertension_plot":
        base_color = "#d62728"
    else:
        raise ValueError(f"Unsupported variable: {variable}")

    palette = create_monochromatic_palette(base_color)
    palette = adjust_brightness_of_colors(palette, brightness_scale=0.2)
    sns.set_palette(palette)

    if not subsets:
        print("⚠️  No subsets provided – nothing to plot.")
        return 1

    fig, ax = plt.subplots(figsize=(2.4, 2.0))

    # Shift data if log scale requested
    if log:
        data = data + 1
        subsets = [s + 1 for s in subsets]

    # Plot main dataset
    x, y = calculate_cdf(data, normalize)
    ax.plot(x, y, label=labels[0])

    # Plot subsets
    for i, subset in enumerate(subsets):
        if i == 0:
            i_color = 0
        elif i == 1:
            i_color = 3
        elif i == 2:
            i_color = 4
        else:
            i_color = i % len(palette)

        x, y = calculate_cdf(subset, normalize)
        if variable == "Individual":
            ax.plot(x, y, label=labels[i + 1], linestyle="--", color=individual_color)
        else:
            ax.plot(x, y, label=labels[i + 1], linestyle="--", color=palette[i_color])

    # Axis labels & title
    if source_sans:
        ax.set_ylabel("CDF", fontproperties=source_sans)
        if log:
            xlabel = "Velocity + 1 (µm/s)" if "Pressure" not in title else "Pressure (psi)"
        else:
            xlabel = "Velocity (µm/s)" if "Pressure" not in title else "Pressure (psi)"
        ax.set_xlabel(xlabel, fontproperties=source_sans)
        ax.set_title(title, fontsize=8, fontproperties=source_sans)
    else:
        ax.set_ylabel("CDF")
        ax.set_xlabel("Velocity + 1 (µm/s)" if log else "Velocity (µm/s)")
        ax.set_title(title)

    # Log scale formatting
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Legend
    if source_sans:
        ax.legend(loc="lower right", bbox_to_anchor=(1, 0.01), prop=source_sans, fontsize=6)
    else:
        ax.legend(loc="lower right", bbox_to_anchor=(1, 0.01), fontsize=6)

    ax.grid(True, linewidth=0.3)
    fig.set_dpi(300)
    plt.tight_layout()

    if write:
        cap_flow_path = PATHS.get("cap_flow", os.getcwd())
        out_dir = os.path.join(cap_flow_path, "results", "cdf_plots")
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{title.replace(' ', '_').lower()}.png"
        fig.savefig(os.path.join(out_dir, filename), dpi=600, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return 0
