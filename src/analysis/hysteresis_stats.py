"""
Standalone module for hysteresis statistical analysis.

This module provides functions for calculating and visualizing statistical
significance in velocity hysteresis analysis without sklearn dependencies.

Functions:
    - calculate_velocity_hysteresis: Calculate hysteresis for each participant
    - calculate_hysteresis_pvalues: Calculate p-values for group differences
    - plot_up_down_diff_boxplots: Create annotated boxplots with significance markers

Author: Marcus Forst
Date: November 2024
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import scipy.stats as stats
from typing import Tuple

# Import paths from config
from src.config import PATHS, load_source_sans
cap_flow_path = PATHS['cap_flow']
source_sans = load_source_sans()


def calculate_velocity_hysteresis(df: pd.DataFrame, use_log_velocity: bool = False) -> pd.DataFrame:
    """Calculates velocity hysteresis (up_down_diff) for each participant and adds it to a new DataFrame.
    
    Computes the difference between upward and downward pressure measurements for 
    each participant, which represents the hysteresis in capillary flow velocities.
    This can be calculated using either raw or log-transformed velocities.
    
    Args:
        df: DataFrame containing participant data with velocity measurements and UpDown column
        use_log_velocity: Whether to use log-transformed velocity measurements (default: False)
    
    Returns:
        DataFrame with participant-level data including the up_down_diff column
    """
    print("\nCalculating velocity hysteresis for each participant...")
    
    # Create a list to store participant data
    participant_data = []
    
    # Determine which velocity column to use
    velocity_column = 'Log_Video_Median_Velocity' if use_log_velocity else 'Video_Median_Velocity'
    
    # Check if the specified velocity column exists
    if velocity_column not in df.columns:
        print(f"Warning: {velocity_column} not found in dataframe. Available columns: {df.columns.tolist()}")
        if use_log_velocity and 'Video_Median_Velocity' in df.columns:
            print("Calculating log velocity from Video_Median_Velocity...")
            df['Log_Video_Median_Velocity'] = np.log10(df['Video_Median_Velocity'])
            velocity_column = 'Log_Video_Median_Velocity'
        else:
            raise ValueError(f"Cannot calculate hysteresis: {velocity_column} not available and cannot be derived")
    
    # Process each participant
    for participant in df['Participant'].unique():
        # Get data for this participant
        participant_df = df[df['Participant'] == participant]
        
        # Calculate up/down velocity differences
        up_velocities = participant_df[participant_df['UpDown'] == 'U'][velocity_column]
        down_velocities = participant_df[participant_df['UpDown'] == 'D'][velocity_column]
        
        # Calculate hysteresis as the difference between mean up and down velocities
        up_down_diff = np.mean(up_velocities) - np.mean(down_velocities) if len(up_velocities) > 0 and len(down_velocities) > 0 else np.nan
        
        # Gather basic participant information
        basic_info = {
            'Participant': participant,
            'up_down_diff': up_down_diff
        }
        
        # Add health conditions if available in the dataset
        for condition in ['Diabetes', 'Hypertension']:  # Could add 'HeartDisease' if available
            if condition in participant_df.columns:
                # Convert to boolean based on value type
                value = participant_df[condition].iloc[0]
                if isinstance(value, str):
                    basic_info[condition] = str(value).upper() == 'TRUE'
                else:
                    basic_info[condition] = bool(value)
        
        # Add is_healthy flag if SET column is available
        if 'SET' in participant_df.columns:
            basic_info['is_healthy'] = participant_df['SET'].iloc[0].startswith('set01')
        
        # Add age if available
        if 'Age' in participant_df.columns:
            basic_info['Age'] = participant_df['Age'].iloc[0]
        
        # Add other demographic data as needed (can be expanded)
        for demo in ['Sex', 'SYS_BP', 'DIA_BP']:
            if demo in participant_df.columns:
                basic_info[demo] = participant_df[demo].iloc[0]
        
        participant_data.append(basic_info)
    
    # Create a new dataframe with participant-level data
    processed_df = pd.DataFrame(participant_data)
    
    # Report on hysteresis calculation
    valid_count = processed_df['up_down_diff'].notna().sum()
    print(f"Calculated hysteresis for {valid_count} of {len(processed_df)} participants")
    
    # Print summary statistics for up_down_diff
    print("\nHysteresis (up_down_diff) summary statistics:")
    print(f"Mean: {processed_df['up_down_diff'].mean():.3f}")
    print(f"Median: {processed_df['up_down_diff'].median():.3f}")
    print(f"Min: {processed_df['up_down_diff'].min():.3f}")
    print(f"Max: {processed_df['up_down_diff'].max():.3f}")
    
    return processed_df


def calculate_hysteresis_pvalues(processed_df: pd.DataFrame, use_absolute: bool = False) -> pd.DataFrame:
    """Calculate p-values for differences in hysteresis between groups.
    
    Uses Mann-Whitney U test for binary comparisons and Kruskal-Wallis for multiple groups.
    For multiple groups (e.g., age), also performs post-hoc pairwise Mann-Whitney U tests.
    
    Args:
        processed_df: DataFrame containing participant data with up_down_diff values
        use_absolute: Whether to use absolute values of hysteresis (default: False)
    
    Returns:
        DataFrame with columns: grouping_factor, group1, group2, test_type, 
                               statistic, p_value, n1, n2, significant
    """
    from scipy.stats import mannwhitneyu, kruskal
    
    # Create a copy to avoid modifying original
    plot_df = processed_df.copy()
    
    # Apply absolute value if requested
    if use_absolute:
        plot_df['up_down_diff'] = plot_df['up_down_diff'].abs()
    
    # Remove NaN values
    plot_df = plot_df.dropna(subset=['up_down_diff'])
    
    results = []
    
    # Define grouping factors
    grouping_factors = [
        {
            'name': 'Age Group',
            'column': 'Age',
            'is_categorical': False,
            'bins': [0, 30, 50, 60, 70, 100],
            'labels': ['<30', '30-49', '50-59', '60-69', '70+']
        },
        {
            'name': 'Health Status',
            'column': 'is_healthy',
            'is_categorical': True,
            'group_labels': {False: 'Affected', True: 'Healthy'}
        },
        {
            'name': 'Diabetes',
            'column': 'Diabetes',
            'is_categorical': True,
            'group_labels': {False: 'No Diabetes', True: 'Diabetes'}
        },
        {
            'name': 'Hypertension',
            'column': 'Hypertension',
            'is_categorical': True,
            'group_labels': {False: 'No Hypertension', True: 'Hypertension'}
        }
    ]
    
    for factor in grouping_factors:
        # Skip if column not present
        if factor['column'] not in plot_df.columns:
            continue
        
        # Create groups
        if not factor['is_categorical']:
            # Create age groups
            plot_df[f"{factor['column']}_Group"] = pd.cut(
                plot_df[factor['column']], 
                bins=factor['bins'], 
                labels=factor['labels'],
                include_lowest=True
            )
            group_col = f"{factor['column']}_Group"
        else:
            group_col = factor['column']
        
        # Get groups with data
        groups = plot_df.groupby(group_col)['up_down_diff'].apply(list).to_dict()
        group_names = list(groups.keys())
        
        # For categorical, map to readable names
        if factor['is_categorical'] and 'group_labels' in factor:
            group_names_display = [factor['group_labels'].get(k, str(k)) for k in group_names]
        else:
            group_names_display = [str(k) for k in group_names]
        
        # Remove groups with too few samples
        valid_groups = {k: v for k, v in groups.items() if len(v) >= 3}
        
        if len(valid_groups) < 2:
            print(f"Warning: Not enough groups with sufficient data for {factor['name']}")
            continue
        
        # If more than 2 groups, perform Kruskal-Wallis first
        if len(valid_groups) > 2:
            group_values = list(valid_groups.values())
            h_stat, p_value = kruskal(*group_values)
            
            results.append({
                'grouping_factor': factor['name'],
                'group1': 'All',
                'group2': 'All',
                'test_type': 'Kruskal-Wallis',
                'statistic': h_stat,
                'p_value': p_value,
                'n1': sum(len(g) for g in group_values),
                'n2': np.nan,
                'significant': 'Yes' if p_value < 0.05 else 'No'
            })
            
            # Perform post-hoc pairwise comparisons
            group_keys = list(valid_groups.keys())
            for i in range(len(group_keys)):
                for j in range(i+1, len(group_keys)):
                    key1, key2 = group_keys[i], group_keys[j]
                    
                    # Get display names
                    if factor['is_categorical'] and 'group_labels' in factor:
                        name1 = factor['group_labels'].get(key1, str(key1))
                        name2 = factor['group_labels'].get(key2, str(key2))
                    else:
                        name1 = str(key1)
                        name2 = str(key2)
                    
                    stat, p_value = mannwhitneyu(valid_groups[key1], valid_groups[key2], alternative='two-sided')
                    
                    results.append({
                        'grouping_factor': factor['name'],
                        'group1': name1,
                        'group2': name2,
                        'test_type': 'Mann-Whitney U',
                        'statistic': stat,
                        'p_value': p_value,
                        'n1': len(valid_groups[key1]),
                        'n2': len(valid_groups[key2]),
                        'significant': 'Yes' if p_value < 0.05 else 'No'
                    })
        
        # If exactly 2 groups, just do Mann-Whitney U
        elif len(valid_groups) == 2:
            group_keys = list(valid_groups.keys())
            key1, key2 = group_keys[0], group_keys[1]
            
            # Get display names
            if factor['is_categorical'] and 'group_labels' in factor:
                name1 = factor['group_labels'].get(key1, str(key1))
                name2 = factor['group_labels'].get(key2, str(key2))
            else:
                name1 = str(key1)
                name2 = str(key2)
            
            stat, p_value = mannwhitneyu(valid_groups[key1], valid_groups[key2], alternative='two-sided')
            
            results.append({
                'grouping_factor': factor['name'],
                'group1': name1,
                'group2': name2,
                'test_type': 'Mann-Whitney U',
                'statistic': stat,
                'p_value': p_value,
                'n1': len(valid_groups[key1]),
                'n2': len(valid_groups[key2]),
                'significant': 'Yes' if p_value < 0.05 else 'No'
            })
    
    return pd.DataFrame(results)


def plot_up_down_diff_boxplots(processed_df: pd.DataFrame, use_absolute: bool = False, 
                                output_dir: str = None, use_log_velocity: bool = False) -> int:
    """Creates boxplots of up_down_diff (velocity hysteresis) grouped by different factors.
    
    Generates separate boxplots showing the relationship between velocity hysteresis
    (difference between upward and downward pressure measurements) and various grouping
    factors including age, health status, diabetes, and hypertension.
    Includes statistical significance testing with asterisk annotations.
    
    Args:
        processed_df: DataFrame containing participant data with up_down_diff values
        use_absolute: Whether to plot absolute values of hysteresis (default: False)
        output_dir: Directory to save the plots (default: results/Hysteresis)
        use_log_velocity: Whether the data is based on log velocity (default: False)
    
    Returns:
        0 if successful, 1 if error occurred
    """
    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'Hysteresis')
        os.makedirs(output_dir, exist_ok=True)
    
    # Standard plot configuration with robust font loading
    sns.set_style("whitegrid")
    
    # Safely get the font
    def get_source_sans_font():
        """Safely load the SourceSans font with fallback to default font."""
        try:
            font_path = os.path.join(PATHS['downloads'], 'Source_Sans_3', 'static', 'SourceSans3-Regular.ttf')
            if os.path.exists(font_path):
                return FontProperties(fname=font_path)
            print("Warning: SourceSans3-Regular.ttf not found, using default font")
            return None
        except Exception as e:
            print(f"Warning: Error loading font: {e}")
            return None
    
    source_sans = get_source_sans_font()
    
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
    
    # Add log prefix for filenames
    log_prefix = "log_" if use_log_velocity else ""
    
    # Define grouping factors and their plot properties
    grouping_factors = [
        {
            'column': 'Age',
            'is_categorical': False,
            'bins': [0, 30, 50, 60, 70, 100],
            'labels': ['<30', '30-49', '50-59', '60-69', '70+'],
            'title': f'Absolute {"Log " if use_log_velocity else ""}Velocity Hysteresis by Age Group' if use_absolute else f'{"Log " if use_log_velocity else ""}Velocity Hysteresis by Age Group',
            'filename': f'{log_prefix}abs_hysteresis_by_age.png' if use_absolute else f'{log_prefix}hysteresis_by_age.png',
            'color': '#1f77b4'  # Default blue
        },
        {
            'column': 'is_healthy',
            'is_categorical': True,
            'title': f'Absolute {"Log " if use_log_velocity else ""}Velocity Hysteresis by Health Status' if use_absolute else f'{"Log " if use_log_velocity else ""}Velocity Hysteresis by Health Status',
            'filename': f'{log_prefix}abs_hysteresis_by_health_status.png' if use_absolute else f'{log_prefix}hysteresis_by_health_status.png',
            'color': '#2ca02c',  # Green
            'x_labels': ['Affected', 'Healthy']
        },
        {
            'column': 'Diabetes',
            'is_categorical': True,
            'title': f'Absolute {"Log " if use_log_velocity else ""}Velocity Hysteresis by Diabetes Status' if use_absolute else f'{"Log " if use_log_velocity else ""}Velocity Hysteresis by Diabetes Status',
            'filename': f'{log_prefix}abs_hysteresis_by_diabetes.png' if use_absolute else f'{log_prefix}hysteresis_by_diabetes.png',
            'color': '#ff7f0e',  # Orange
            'x_labels': ['No Diabetes', 'Diabetes']
        },
        {
            'column': 'Hypertension',
            'is_categorical': True,
            'title': f'Absolute {"Log " if use_log_velocity else ""}Velocity Hysteresis by Hypertension Status' if use_absolute else f'{"Log " if use_log_velocity else ""}Velocity Hysteresis by Hypertension Status',
            'filename': f'{log_prefix}abs_hysteresis_by_hypertension.png' if use_absolute else f'{log_prefix}hysteresis_by_hypertension.png',
            'color': '#d62728',  # Red
            'x_labels': ['No Hypertension', 'Hypertension']
        }
    ]
    
    # Check if up_down_diff exists in the dataframe
    if 'up_down_diff' not in processed_df.columns:
        print("Error: 'up_down_diff' column not found in the dataframe")
        return 1
    
    # Create a boxplot for each grouping factor
    for factor in grouping_factors:
        plt.figure(figsize=(2.4, 2.0))
        
        # Create a copy of the dataframe to avoid modifying the original
        plot_df = processed_df.copy()
        
        # Calculate absolute values if requested
        if use_absolute:
            plot_df['up_down_diff'] = plot_df['up_down_diff'].abs()
        
        # Handle age binning for age groups
        if not factor['is_categorical']:
            # Create age groups
            plot_df[f"{factor['column']}_Group"] = pd.cut(
                plot_df[factor['column']], 
                bins=factor['bins'], 
                labels=factor['labels'],
                include_lowest=True
            )
            group_col = f"{factor['column']}_Group"
        else:
            group_col = factor['column']
        
        # Create the boxplot
        ax = sns.boxplot(
            x=group_col,
            y='up_down_diff',
            data=plot_df,
            color=factor['color'],
            width=0.6,
            fliersize=3
        )
        
        # Add a horizontal line at y=0 for reference
        ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Set custom x-axis labels if provided
        if 'x_labels' in factor and factor['is_categorical']:
            ax.set_xticklabels(factor['x_labels'])
        
        # Set title and labels
        if source_sans:
            ax.set_title(factor['title'], fontproperties=source_sans, fontsize=8)
            ax.set_xlabel('Group', fontproperties=source_sans)
            ax.set_ylabel('|Velocity Hysteresis| (up-down)' if use_absolute else 'Velocity Hysteresis (up-down)', 
                        fontproperties=source_sans)
        else:
            ax.set_title(factor['title'], fontsize=8)
            ax.set_xlabel('Group')
            ax.set_ylabel('|Velocity Hysteresis| (up-down)' if use_absolute else 'Velocity Hysteresis (up-down)')
        
        # Add statistical annotation
        groups = plot_df.groupby(group_col)['up_down_diff'].apply(list).to_dict()
        
        if factor['is_categorical']:
            # For binary categorical variables
            if len(groups) == 2:
                group_values = list(groups.values())
                stat, p_value = stats.mannwhitneyu(group_values[0], group_values[1])
                
                # Determine significance level
                if p_value < 0.001:
                    sig_marker = "***"
                    p_text = f"p < 0.001 {sig_marker}"
                elif p_value < 0.01:
                    sig_marker = "**"
                    p_text = f"p = {p_value:.3f} {sig_marker}"
                elif p_value < 0.05:
                    sig_marker = "*"
                    p_text = f"p = {p_value:.3f} {sig_marker}"
                else:
                    sig_marker = "ns"
                    p_text = f"p = {p_value:.3f} (ns)"
                
                # Add p-value annotation at the top
                if source_sans:
                    ax.text(0.5, 0.95, p_text, transform=ax.transAxes, 
                           ha='center', fontproperties=source_sans, fontsize=6,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.5, 0.95, p_text, transform=ax.transAxes, 
                           ha='center', fontsize=6,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Draw significance bracket if significant
                if p_value < 0.05:
                    # Get y-axis range for positioning
                    y_min, y_max = ax.get_ylim()
                    y_range = y_max - y_min
                    bracket_height = y_max - (y_range * 0.05)
                    
                    # Draw horizontal line connecting the two boxes
                    ax.plot([1, 1, 2, 2], 
                           [bracket_height - (y_range * 0.02), bracket_height, bracket_height, bracket_height - (y_range * 0.02)], 
                           'k-', linewidth=0.8)
                    
                    # Add asterisks above the bracket
                    ax.text(1.5, bracket_height + (y_range * 0.01), sig_marker, 
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            # For multi-group variables (age groups), test all pairwise comparisons
            # and show only significant ones
            group_keys = sorted([k for k in groups.keys() if len(groups[k]) >= 3])
            significant_pairs = []
            
            for i in range(len(group_keys)):
                for j in range(i+1, len(group_keys)):
                    key1, key2 = group_keys[i], group_keys[j]
                    stat, p_value = stats.mannwhitneyu(groups[key1], groups[key2], alternative='two-sided')
                    
                    if p_value < 0.05:
                        significant_pairs.append({
                            'group1': key1,
                            'group2': key2,
                            'pos1': i,
                            'pos2': j,
                            'p_value': p_value,
                            'sig_marker': "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
                        })
            
            # Draw brackets for significant comparisons (limit to 3 most significant)
            if significant_pairs:
                # Sort by p-value and take top 3
                significant_pairs = sorted(significant_pairs, key=lambda x: x['p_value'])[:3]
                
                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                
                for idx, pair in enumerate(significant_pairs):
                    # Position brackets at different heights to avoid overlap
                    bracket_height = y_max - (y_range * (0.05 + idx * 0.10))
                    pos1, pos2 = pair['pos1'], pair['pos2']
                    
                    # Draw bracket
                    ax.plot([pos1+1, pos1+1, pos2+1, pos2+1], 
                           [bracket_height - (y_range * 0.02), bracket_height, bracket_height, bracket_height - (y_range * 0.02)], 
                           'k-', linewidth=0.8)
                    
                    # Add asterisks
                    ax.text((pos1 + pos2) / 2 + 1, bracket_height + (y_range * 0.01), 
                           pair['sig_marker'], 
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                # Add summary text
                summary_lines = [f"{pair['group1']}-{pair['group2']}: p={pair['p_value']:.3f}{pair['sig_marker']}" 
                               for pair in significant_pairs]
                summary_text = "Significant pairs:\n" + "\n".join(summary_lines)
                
                if source_sans:
                    ax.text(0.98, 0.97, summary_text, transform=ax.transAxes, 
                           ha='right', va='top', fontproperties=source_sans, fontsize=5,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.98, 0.97, summary_text, transform=ax.transAxes, 
                           ha='right', va='top', fontsize=5,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add group sample sizes
        if factor['is_categorical']:
            # Get counts for each category
            counts = plot_df[group_col].value_counts().sort_index()
            
            # Add as x-tick labels with counts
            xtick_labels = [f"{label}\n(n={counts[i]})" 
                          if i < len(counts) else label
                          for i, label in enumerate(ax.get_xticklabels())]
            ax.set_xticklabels(xtick_labels)
        else:
            # Get counts for each age group
            counts = plot_df[group_col].value_counts().sort_index()
            
            # Add counts to the x-tick labels
            xtick_labels = [f"{label.get_text()}\n(n={counts[label.get_text()]})" 
                          if label.get_text() in counts.index else label.get_text()
                          for label in ax.get_xticklabels()]
            ax.set_xticklabels(xtick_labels)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, factor['filename']), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Boxplots saved to {output_dir}")
    return 0


# Main function for standalone execution
def main():
    """Main function to run the hysteresis analysis."""
    print("\nStarting hysteresis analysis with statistical testing...")
    
    # Create main output directory
    output_dir = os.path.join(cap_flow_path, 'results', 'Hysteresis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Calculate velocity hysteresis for participants
    processed_df = calculate_velocity_hysteresis(df, use_log_velocity=False)
    
    # Calculate p-values for statistical significance
    print("\nCalculating p-values for group differences...")
    
    # Calculate p-values for regular hysteresis
    pvalues_regular = calculate_hysteresis_pvalues(processed_df, use_absolute=False)
    pvalues_regular['analysis_type'] = 'Regular Hysteresis'
    
    # Calculate p-values for absolute hysteresis
    pvalues_absolute = calculate_hysteresis_pvalues(processed_df, use_absolute=True)
    pvalues_absolute['analysis_type'] = 'Absolute Hysteresis'
    
    # Combine all p-values
    all_pvalues = pd.concat([pvalues_regular, pvalues_absolute], ignore_index=True)
    
    # Save p-values to CSV
    pvalue_filepath = os.path.join(output_dir, 'hysteresis_pvalues.csv')
    all_pvalues.to_csv(pvalue_filepath, index=False)
    print(f"\nP-values saved to: {pvalue_filepath}")
    
    # Print summary of significant findings
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE SUMMARY")
    print("="*70)
    significant_tests = all_pvalues[all_pvalues['p_value'] < 0.05]
    if len(significant_tests) > 0:
        print(f"\nFound {len(significant_tests)} significant differences (p < 0.05):\n")
        for _, row in significant_tests.iterrows():
            sig_level = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*"
            print(f"{row['analysis_type']} - {row['grouping_factor']}: "
                  f"{row['group1']} vs {row['group2']}")
            print(f"  p = {row['p_value']:.4f} {sig_level} ({row['test_type']})")
            print(f"  Sample sizes: n1={int(row['n1']) if not pd.isna(row['n1']) else 'N/A'}, "
                  f"n2={int(row['n2']) if not pd.isna(row['n2']) else 'N/A'}\n")
    else:
        print("\nNo significant differences found at p < 0.05 level.")
    print("="*70 + "\n")
    
    # Run boxplot analysis for regular and absolute values
    print("\nRunning boxplot analysis...")
    
    # Regular hysteresis plots
    plot_up_down_diff_boxplots(processed_df, use_absolute=False, 
                              output_dir=output_dir, use_log_velocity=False)
    
    # Absolute hysteresis plots
    plot_up_down_diff_boxplots(processed_df, use_absolute=True, 
                              output_dir=output_dir, use_log_velocity=False)
    
    print("\nAnalysis complete.")
    print(f"Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    main()

