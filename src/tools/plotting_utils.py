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
            log_scale=False, old = False, velocity_variable = 'Corrected Velocity'):
    """Plots the mean/median and CI for the variable of interest, with KS statistic."""
    # Set up style and font
    sns.set_style("whitegrid")
    source_sans = get_source_sans_font()
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    control_df = df[df['SET']=='set01']
    hypertensive_df = df[df['SET']=='set02']
    diabetic_df = df[df['SET']=='set03']
    affected_df = df[df['Set_affected']=='set04']

    # Set color palette based on variable
    if variable == 'Age':
        base_color = '#1f77b4'
        conditions = [df[variable] <= 50, df[variable] > 50]
        choices = ['≤50', '>50']
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
        lambda x: stats_func(x, ci_percentile=ci_percentile, dimensionless=dimensionless, velocity_variable=velocity_variable)
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
                
        plt.savefig(output_path, dpi=600)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
        
    return 0

# Function to calculate median and bootstrap 95% CI
def calculate_median_ci(group, n_iterations=1000, ci_percentile=95, dimensionless=False, velocity_variable='Corrected_Velocity'):
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

    Returns:
        pd.Series: A series containing the median and CI bounds.

    Example:
        stats_df = df.groupby([group_col, 'Pressure']).apply(stats_func, ci_percentile=ci_percentile, dimensionless=dimensionless).reset_index()
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
            
        for _ in range(n_iterations):
            sample = resample(valid_data)
            medians_dimless.append(np.median(sample))
            
        lower = np.percentile(medians_dimless, (100 - ci_percentile) / 2)
        upper = np.percentile(medians_dimless, 100 - (100 - ci_percentile) / 2)
        median_dimless = np.median(valid_data)
        
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
            
        # print(f"DEBUG - Using {velocity_variable}")    
        # Drop any NaN values before bootstrapping
        valid_data = group[velocity_variable].dropna()
        
        if len(valid_data) == 0:
            print(f"DEBUG - No valid {velocity_variable} data")
            return pd.Series(result)
            
        for _ in range(n_iterations):
            sample = resample(valid_data)
            medians.append(np.median(sample))
            
        lower = np.percentile(medians, (100 - ci_percentile) / 2)
        upper = np.percentile(medians, 100 - (100 - ci_percentile) / 2)
        median = np.median(valid_data)
        
        if velocity_variable == 'Shear_Rate':
            # print("DEBUG - Setting result for Shear_Rate")
            result = {
                'Median Shear Rate': median, 
                'CI Lower Bound': lower, 
                'CI Upper Bound': upper
            }
        else:
            # print("DEBUG - Setting result for Velocity")
            result = {
                'Median Velocity': median, 
                'CI Lower Bound': lower, 
                'CI Upper Bound': upper
            }
    
    # print(f"DEBUG - Returning result keys: {list(result.keys())}")
    return pd.Series(result)

# Function to calculate mean, standard error, and 95% CI
def calculate_mean_ci(group, ci_percentile = 95, dimensionless = False, velocity_variable = 'Corrected_Velocity'):
    """
    Calculates the mean, standard error, and 95% CI for a given group.
    If the group has a 'Dimensionless Velocity' column, it will use that column.
    Otherwise, it will use the 'Corrected_Velocity' or the 'Shear_Rate' column.

    Args:
        group (pd.DataFrame): The group to calculate the stats for.
        ci_percentile (int): The percentile to use for the CI.
        dimensionless (bool): Whether to return the stats in dimensionless units.

    Returns:
        pd.Series: A series containing the mean, lower bound, and upper bound of the CI.
    """
    if dimensionless:
        mean = group['Dimensionless Velocity'].mean()
        sem = stats.sem(group['Dimensionless Velocity'])
        ci = 1.96 * sem
        return pd.Series({'Mean Dimensionless Velocity': mean, 'Lower Bound': mean - ci, 'Upper Bound': mean + ci})
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