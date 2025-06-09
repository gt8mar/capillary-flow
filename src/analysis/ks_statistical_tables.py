"""
Filename: src/analysis/ks_statistical_tables.py
---------------------------------------------------------

Generate and save KS statistical significance tables for velocity distributions 
at each external pressure for different demographic and health variables.

By: Marcus Forst
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from src.config import PATHS

def calculate_ks_statistics_for_variable(df, variable, velocity_variable='Corrected Velocity'):
    """
    Calculate KS statistics for a given variable across all pressures.
    
    Args:
        df: DataFrame containing the data
        variable: Variable to test (e.g., 'Age', 'Sex', 'SYS_BP', 'Diabetes', 'Hypertension')
        velocity_variable: Column name for velocity measurements
        
    Returns:
        DataFrame with KS statistics for each pressure
    """
    # Define grouping conditions for each variable
    if variable == 'Age':
        conditions = [df[variable] <= 59, df[variable] > 59]
        choices = ['≤59', '>59']
        group_labels = ['≤59 years', '>59 years']
    elif variable == 'SYS_BP':
        conditions = [df[variable] < 120, df[variable] >= 120]
        choices = ['<120', '≥120']
        group_labels = ['<120 mmHg', '≥120 mmHg']
    elif variable == 'Sex':
        conditions = [df[variable] == 'M', df[variable] == 'F']
        choices = ['Male', 'Female']
        group_labels = ['Male', 'Female']
    elif variable == 'Diabetes':
        conditions = [df['SET'] == 'set01', df['SET'] == 'set03']
        choices = ['Control', 'Diabetic']
        group_labels = ['Control', 'Diabetic']
    elif variable == 'Hypertension':
        conditions = [df['SET'] == 'set01', df['SET'] == 'set02']
        choices = ['Control', 'Hypertensive']
        group_labels = ['Control', 'Hypertensive']
    else:
        raise ValueError(f"Unsupported variable: {variable}")
    
    # Create group column
    group_col = f'{variable}_Group'
    df[group_col] = np.select(conditions, choices, default='Unknown')
    
    # Remove unknown values
    df_clean = df[df[group_col] != 'Unknown'].copy()
    
    # Group data
    grouped = df_clean.groupby(group_col)
    ks_stats = []
    
    # Calculate KS statistics for each pressure
    for pressure in sorted(df_clean['Pressure'].unique()):
        try:
            # Check if both groups exist
            if choices[0] not in grouped.groups or choices[1] not in grouped.groups:
                print(f"Warning: One or more groups missing for {variable} at pressure {pressure}")
                continue
                
            group_1 = grouped.get_group(choices[0])
            group_2 = grouped.get_group(choices[1])
            
            # Get velocity data for this pressure
            group_1_velocities = group_1[group_1['Pressure'] == pressure][velocity_variable]
            group_2_velocities = group_2[group_2['Pressure'] == pressure][velocity_variable]
            
            # Check if either group has empty data for this pressure
            if len(group_1_velocities) == 0 or len(group_2_velocities) == 0:
                print(f"Warning: Insufficient data for KS test for {variable} at pressure {pressure}")
                continue
            
            # Perform KS test
            ks_stat, p_value = ks_2samp(group_1_velocities, group_2_velocities)
            
            # Calculate medians
            group_1_median = group_1_velocities.median()
            group_2_median = group_2_velocities.median()
            
            # Calculate sample sizes
            n1 = len(group_1_velocities)
            n2 = len(group_2_velocities)
            
            ks_stats.append({
                'Variable': variable,
                'Pressure_psi': pressure,
                'KS_Statistic': ks_stat,
                'p_value': p_value,
                f'{group_labels[0]}_Median': group_1_median,
                f'{group_labels[1]}_Median': group_2_median,
                f'{group_labels[0]}_N': n1,
                f'{group_labels[1]}_N': n2,
                'Significant_p005': p_value < 0.05,
                'Significant_p001': p_value < 0.01
            })
            
        except KeyError as e:
            print(f"Warning: Could not find group for {variable} at pressure {pressure}: {e}")
            continue
    
    return pd.DataFrame(ks_stats)

def main():
    """
    Main function to generate and save KS statistical tables.
    """
    print("Loading data...")
    
    # Load the main dataset
    data_filepath = os.path.join(PATHS['cap_flow'], 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Clean data
    df = df.dropna(subset=['Age'])
    df = df.dropna(subset=['Corrected Velocity'])
    df = df.drop_duplicates(subset=['Participant', 'Video'])
    
    print(f"Data loaded: {len(df)} rows, {len(df['Participant'].unique())} unique participants")
    
    # Create output directory
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'ks_statistics_tables')
    os.makedirs(output_dir, exist_ok=True)
    
    # Variables to test
    demographic_variables = ['Age', 'SYS_BP', 'Sex']
    health_variables = ['Hypertension', 'Diabetes']
    all_variables = demographic_variables + health_variables
    
    print("\nCalculating KS statistics for all variables...")
    
    # Calculate KS statistics for each variable
    all_ks_results = []
    
    for variable in all_variables:
        print(f"Processing {variable}...")
        try:
            ks_df = calculate_ks_statistics_for_variable(df, variable)
            if not ks_df.empty:
                all_ks_results.append(ks_df)
                
                # Save individual table for this variable
                output_file = os.path.join(output_dir, f'{variable}_KS_statistics.csv')
                ks_df.to_csv(output_file, index=False)
                print(f"  Saved individual table: {output_file}")
                
                # Print summary
                print(f"  Found {len(ks_df)} pressure points")
                significant_05 = ks_df['Significant_p005'].sum()
                significant_01 = ks_df['Significant_p001'].sum()
                print(f"  Significant at p<0.05: {significant_05}/{len(ks_df)}")
                print(f"  Significant at p<0.01: {significant_01}/{len(ks_df)}")
            else:
                print(f"  No valid data for {variable}")
        except Exception as e:
            print(f"  Error processing {variable}: {e}")
    
    # Combine all results
    if all_ks_results:
        combined_df = pd.concat(all_ks_results, ignore_index=True)
        
        # Save combined table
        combined_output_file = os.path.join(output_dir, 'All_Variables_KS_Statistics.csv')
        combined_df.to_csv(combined_output_file, index=False)
        print(f"\nSaved combined table: {combined_output_file}")
        
        # Create summary table by variable
        summary_stats = []
        for variable in all_variables:
            var_data = combined_df[combined_df['Variable'] == variable]
            if not var_data.empty:
                summary_stats.append({
                    'Variable': variable,
                    'Total_Pressure_Points': len(var_data),
                    'Significant_p005': var_data['Significant_p005'].sum(),
                    'Significant_p001': var_data['Significant_p001'].sum(),
                    'Percent_Significant_p005': (var_data['Significant_p005'].sum() / len(var_data)) * 100,
                    'Mean_KS_Statistic': var_data['KS_Statistic'].mean(),
                    'Max_KS_Statistic': var_data['KS_Statistic'].max(),
                    'Min_p_value': var_data['p_value'].min()
                })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_output_file = os.path.join(output_dir, 'KS_Statistics_Summary.csv')
        summary_df.to_csv(summary_output_file, index=False)
        print(f"Saved summary table: {summary_output_file}")
        
        # Create comparison table as requested
        print("\nCreating comparison tables...")
        
        # Table 1: Demographics (Age, SYS_BP, Sex)
        demo_table = create_comparison_table(combined_df, demographic_variables)
        demo_output_file = os.path.join(output_dir, 'Demographics_KS_Comparison_Table.csv')
        demo_table.to_csv(demo_output_file, index=False)
        print(f"Saved demographics comparison table: {demo_output_file}")
        
        # Table 2: Health variables (Hypertension, Diabetes) with Age for comparison
        health_comparison_variables = ['Age'] + health_variables
        health_table = create_comparison_table(combined_df, health_comparison_variables)
        health_output_file = os.path.join(output_dir, 'Health_and_Age_KS_Comparison_Table.csv')
        health_table.to_csv(health_output_file, index=False)
        print(f"Saved health comparison table: {health_output_file}")
        
        print(f"\n✅ All KS statistical tables saved to: {output_dir}")
        print(f"Total files created: {len(os.listdir(output_dir))}")
        
    else:
        print("No valid KS statistics calculated")
    
    return 0

def create_comparison_table(combined_df, variables):
    """
    Create a comparison table showing KS statistics across pressures for multiple variables.
    
    Args:
        combined_df: DataFrame with all KS statistics
        variables: List of variables to include in the comparison
        
    Returns:
        DataFrame formatted for easy comparison
    """
    # Get all unique pressures
    pressures = sorted(combined_df['Pressure_psi'].unique())
    
    # Create comparison table
    comparison_data = []
    
    for pressure in pressures:
        row = {'Pressure_psi': pressure}
        
        for variable in variables:
            var_data = combined_df[
                (combined_df['Variable'] == variable) & 
                (combined_df['Pressure_psi'] == pressure)
            ]
            
            if not var_data.empty:
                ks_stat = var_data['KS_Statistic'].iloc[0]
                p_val = var_data['p_value'].iloc[0]
                significant = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                
                row[f'{variable}_KS_Stat'] = f"{ks_stat:.3f}"
                row[f'{variable}_p_value'] = f"{p_val:.4f}"
                row[f'{variable}_Sig'] = significant
            else:
                row[f'{variable}_KS_Stat'] = "NA"
                row[f'{variable}_p_value'] = "NA"
                row[f'{variable}_Sig'] = "NA"
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

if __name__ == '__main__':
    main() 