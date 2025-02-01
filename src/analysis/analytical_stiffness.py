import platform
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit



# Get the hostname of the computer
hostname = platform.node()

# Dictionary mapping hostnames to folder paths
cap_flow_folder_paths = {
    "LAPTOP-I5KTBOR3": 'C:\\Users\\gt8ma\\capillary-flow',
    "Quake-Blood": "C:\\Users\\gt8mar\\capillary-flow",
    "ComputerName3": "C:\\Users\\ejerison\\capillary-flow",
    # Add more computers as needed
}
default_folder_path = "/hpc/projects/capillary-flow"

cap_flow_path = cap_flow_folder_paths.get(hostname, default_folder_path)

def velocity_model(P, threshold, baseline, decay):
    """Piecewise model for capillary velocity response"""
    return np.where(P >= threshold, baseline * np.exp(-decay * (P - threshold)), baseline)

def fit_mixed_effects_model(df, response_var, group_name, output_dir):
    """
    Fit mixed effects model and create diagnostic plots
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data
    response_var : str
        Name of response variable (e.g., 'Log_Video_Median_Velocity' or 'Video_Median_Velocity')
    group_name : str
        Name of the group (e.g., 'Healthy', 'Hypertensive', 'Diabetic')
    output_dir : str
        Directory to save output plots
    
    Returns:
    --------
    statsmodels.regression.mixed_linear_model.MixedLMResults
        Fitted model results
    """
    # Fit mixed effects model
    mixed_model = smf.mixedlm(f'{response_var} ~ Pressure', 
                             df, 
                             groups=df['Participant'],
                             re_formula='~Pressure')
    mixed_results = mixed_model.fit()
    
    # Print results
    print(f'\nMixed Effects Model Results for {group_name} Group:')
    print(mixed_results.summary())

    # Plot fitted vs actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(mixed_results.fittedvalues, 
               df[response_var],
               alpha=0.5)
    plt.plot([df[response_var].min(), 
             df[response_var].max()],
            [df[response_var].min(), 
             df[response_var].max()],
            'r--')
    plt.xlabel('Fitted Values')
    plt.ylabel(f'Actual {response_var}')
    plt.title(f'Fitted vs Actual Values - {group_name} Group')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'mixed_effects_{group_name.lower()}_{response_var}.png'))
    plt.close()
    
    return mixed_results

def calculate_stiffness_parameters(mixed_model_results, df, group_name):
    """
    Calculate stiffness-related parameters with confidence intervals
    """
    # Extract fixed effects coefficients and their standard errors
    intercept = mixed_model_results.fe_params['Intercept']
    pressure_coef = mixed_model_results.fe_params['Pressure']
    
    # Get confidence intervals from the model
    conf_int = mixed_model_results.conf_int()
    pressure_ci_lower = conf_int.loc['Pressure', 0]
    pressure_ci_upper = conf_int.loc['Pressure', 1]
    intercept_ci_lower = conf_int.loc['Intercept', 0]
    intercept_ci_upper = conf_int.loc['Intercept', 1]
    
    # Calculate stiffness index and its confidence interval
    stiffness_index = -pressure_coef
    stiffness_ci_lower = -pressure_ci_upper  # Note: bounds flip due to negative
    stiffness_ci_upper = -pressure_ci_lower
    
    # Calculate compliance and its confidence interval
    compliance = 1 / stiffness_index if stiffness_index != 0 else float('inf')
    compliance_ci_lower = 1 / stiffness_ci_upper if stiffness_ci_upper != 0 else float('inf')
    compliance_ci_upper = 1 / stiffness_ci_lower if stiffness_ci_lower != 0 else float('inf')
    
    # Calculate pressure-velocity sensitivity with bootstrap confidence intervals
    n_bootstrap = 1000
    sensitivities = []
    for _ in range(n_bootstrap):
        boot_sample = df.sample(n=len(df), replace=True)
        pressure_range = boot_sample['Pressure'].max() - boot_sample['Pressure'].min()
        velocity_range = boot_sample['Video_Median_Velocity'].max() - boot_sample['Video_Median_Velocity'].min()
        sensitivities.append(velocity_range / pressure_range)
    
    sensitivity = np.mean(sensitivities)
    sensitivity_ci = np.percentile(sensitivities, [2.5, 97.5])
    
    # Calculate baseline velocity and its confidence interval
    if 'Log_' in df.columns[0]:
        baseline_velocity = np.exp(intercept)
        baseline_ci_lower = np.exp(intercept_ci_lower)
        baseline_ci_upper = np.exp(intercept_ci_upper)
    else:
        baseline_velocity = intercept
        baseline_ci_lower = intercept_ci_lower
        baseline_ci_upper = intercept_ci_upper
    
    return {
        'group': group_name,
        'stiffness_index': stiffness_index,
        'stiffness_ci': (stiffness_ci_lower, stiffness_ci_upper),
        'compliance': compliance,
        'compliance_ci': (compliance_ci_lower, compliance_ci_upper),
        'pressure_sensitivity': sensitivity,
        'sensitivity_ci': (sensitivity_ci[0], sensitivity_ci[1]),
        'baseline_velocity': baseline_velocity,
        'baseline_ci': (baseline_ci_lower, baseline_ci_upper)
    }

def plot_stiffness_comparison(stiffness_params, output_dir):
    """
    Create comparative plots of stiffness parameters across groups with error bars
    """
    # Convert list of dictionaries to more convenient format for plotting
    groups = [p['group'] for p in stiffness_params]
    
    params_to_plot = {
        'stiffness_index': ('Stiffness Index', 'stiffness_ci'),
        'compliance': ('Compliance', 'compliance_ci'),
        'pressure_sensitivity': ('Pressure Sensitivity', 'sensitivity_ci'),
        'baseline_velocity': ('Baseline Velocity', 'baseline_ci')
    }
    
    for param, (title, ci_key) in params_to_plot.items():
        values = [p[param] for p in stiffness_params]
        
        # Calculate errors ensuring they're positive
        errors_lower = [max(0, p[param] - p[ci_key][0]) for p in stiffness_params]
        errors_upper = [max(0, p[ci_key][1] - p[param]) for p in stiffness_params]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(groups, values)
        
        # Add error bars with separate lower and upper bounds
        plt.errorbar(groups, values, yerr=[errors_lower, errors_upper], 
                    fmt='none', color='black', capsize=5)
        
        # Calculate coefficient of variation for each group
        cv_values = [np.mean([el, eu]) / abs(value) * 100 if value != 0 else np.nan 
                    for value, el, eu in zip(values, errors_lower, errors_upper)]
        
        # Add CV values as text above bars
        for bar, cv in zip(bars, cv_values):
            height = bar.get_height()
            if not np.isnan(cv):
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'CV: {cv:.1f}%',
                        ha='center', va='bottom')
        
        plt.title(f'{title} Comparison\nwith 95% Confidence Intervals')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'stiffness_{param}_comparison_with_ci.png'))
        plt.close()

def calculate_participant_stiffness(df, participant_id):
    """
    Calculate stiffness parameters for a single participant
    """
    try:
        # Get participant's age
        participant_age = df['Age'].iloc[0]
        
        # Calculate velocity profiles and hysteresis
        # Group by pressure and UpDown direction, calculate mean velocities
        velocity_profiles = df.groupby(['Pressure', 'UpDown'])['Video_Median_Velocity'].mean().unstack()
        
        # Ensure we have all directions, if not, return NaN for hysteresis metrics
        if not all(direction in velocity_profiles.columns for direction in ['U', 'D', 'T']):
            hysteresis = np.nan
            total_area = np.nan
        else:
            # Sort by pressure to ensure correct area calculation
            velocity_profiles = velocity_profiles.sort_index()
            
            # Combine 'U' and 'T' for up curve, use 'D' for down curve
            up_velocities = velocity_profiles[['U', 'T']].mean(axis=1)
            down_velocities = velocity_profiles['D']
            
            # Calculate areas under curves using trapezoidal rule
            pressures = velocity_profiles.index.values
            up_area = np.trapz(up_velocities.values, pressures)
            down_area = np.trapz(down_velocities.values, pressures)
            
            # Calculate hysteresis (up - down area)
            hysteresis = up_area - down_area
            
            # Calculate total area (average of up and down)
            total_area = (up_area + down_area) / 2
        
        # Fit linear model with just Pressure
        model = smf.ols('Video_Median_Velocity ~ Pressure', data=df).fit()
        
        # Calculate other metrics as before
        stiffness_index = -model.params['Pressure']
        baseline_velocity = model.params['Intercept']
        compliance = 1 / stiffness_index if abs(stiffness_index) > 1e-10 else float('inf')
        pressure_range = df['Pressure'].max() - df['Pressure'].min()
        velocity_range = df['Video_Median_Velocity'].max() - df['Video_Median_Velocity'].min()
        sensitivity = velocity_range / pressure_range if pressure_range > 0 else np.nan
        r_squared = model.rsquared if not np.isnan(model.rsquared) else np.nan
        
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

def add_stiffness_metrics_to_df(df):
    """
    Add stiffness metrics for each participant to the dataframe
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe with velocity and pressure measurements
    
    Returns:
    --------
    pandas DataFrame
        Original dataframe with added stiffness metrics
    """
    # Calculate stiffness parameters for each participant
    participant_metrics = []
    for participant in df['Participant'].unique():
        participant_df = df[df['Participant'] == participant]
        print(f'age of participant {participant} is {participant_df["Age"].iloc[0]}')
        metrics = calculate_participant_stiffness(participant_df, participant)
        participant_metrics.append(metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(participant_metrics)
    
    # Merge with original dataframe
    result_df = df.merge(metrics_df, on='Participant', how='left')
    # Rename Age_x to Age
    result_df = result_df.rename(columns={'Age_x': 'Age'})
    # drop the Age_y column
    result_df = result_df.drop(columns=['Age_y'])
    # print any columns with _x or _y in the column names
    print(result_df.columns[result_df.columns.str.contains('_x') | result_df.columns.str.contains('_y')])
    
    return result_df

def plot_participant_comparisons(df, output_dir):
    """
    Create plots comparing stiffness parameters across participants and against age
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing metrics for each participant
    output_dir : str
        Directory to save output plots
    """
    # Parameters to plot
    params_to_plot = {
        'Stiffness_Index': 'Capillary Stiffness Index',
        'Compliance': 'Capillary Compliance',
        'Baseline_Velocity': 'Baseline Blood Velocity',
        'Pressure_Sensitivity': 'Pressure-Velocity Sensitivity',
        'Model_R_Squared': 'Model Fit (R²)'
    }
    
    for param, title in params_to_plot.items():
        try:
            # 1. Parameter vs Age scatter plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='Age', y=param)
            sns.regplot(data=df, x='Age', y=param, scatter=False, color='red')
            
            # Calculate correlation
            correlation = df[['Age', param]].corr().iloc[0, 1]
            plt.title(f'{title} vs Age\nCorrelation: {correlation:.3f}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param.lower()}_vs_age.png'))
            plt.close()
            
            # 2. Parameter distribution across participants
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, y=param)
            sns.swarmplot(data=df, y=param, color='black', size=5, alpha=0.7)
            plt.title(f'Distribution of {title} Across Participants')
            plt.xticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param.lower()}_distribution.png'))
            plt.close()
            
            # 3. Parameter values by participant (ordered by age)
            plt.figure(figsize=(15, 6))
            participant_order = df.sort_values('Age')['Participant'].values
            sns.barplot(data=df, x='Participant', y=param, order=participant_order)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'{title} by Participant (ordered by age)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param.lower()}_by_participant.png'))
            plt.close()
            
        except Exception as e:
            print(f"\nError plotting {param}: {str(e)}")
            continue

def main():
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_medians.csv')
    output_dir = os.path.join(cap_flow_path, 'results')
    df = pd.read_csv(data_filepath)
    # check if the dataframe has a hysterisis column
    if 'Hysterisis' in df.columns:
        print('it has a hysterisis column')
    else:
        print('it does not have a hysterisis column')

    # Split data into groups
    healthy_df = df[df['SET'] == 'set01']
    hypertensive_df = df[df['SET'] == 'set02']
    diabetic_df = df[df['SET'] == 'set03']

    # Analyze both log and non-log velocities
    response_variables = ['Log_Video_Median_Velocity', 'Video_Median_Velocity']
    
    for response_var in response_variables:
        models = {}
        stiffness_params = []
        for name, group_df in [('Healthy', healthy_df), 
                             ('Hypertensive', hypertensive_df), 
                             ('Diabetic', diabetic_df)]:
            models[name] = fit_mixed_effects_model(group_df, response_var, name, output_dir)
            # Calculate and store stiffness parameters
            params = calculate_stiffness_parameters(models[name], group_df, name)
            stiffness_params.append(params)
        
        # Plot stiffness comparisons
        plot_stiffness_comparison(stiffness_params, output_dir)

    # Add stiffness metrics to the dataframe
    df = add_stiffness_metrics_to_df(df)
    
    # Plot comparisons
    plot_participant_comparisons(df, output_dir)
    
    # Save the enhanced dataframe
    output_filepath = os.path.join(cap_flow_path, 'summary_df_with_stiffness.csv')
    df.to_csv(output_filepath, index=False)

if __name__ == "__main__":
    main()