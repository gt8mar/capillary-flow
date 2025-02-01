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
    Calculate stiffness parameters for a single participant, including age effects
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data for a single participant
    participant_id : str
        Participant identifier
    
    Returns:
    --------
    dict
        Dictionary containing stiffness parameters for the participant
    """
    # Fit linear model including Age as a covariate
    model = smf.ols('Video_Median_Velocity ~ Pressure + Age', data=df).fit()
    
    # Get confidence intervals
    conf_int = model.conf_int()
    
    # Calculate stiffness parameters
    stiffness_index = -model.params['Pressure']  # Negative because higher pressure -> lower velocity
    age_effect = model.params['Age']
    baseline_velocity = model.params['Intercept']
    
    # Calculate compliance
    compliance = 1 / stiffness_index if stiffness_index != 0 else float('inf')
    
    # Calculate pressure-velocity sensitivity
    pressure_range = df['Pressure'].max() - df['Pressure'].min()
    velocity_range = df['Video_Median_Velocity'].max() - df['Video_Median_Velocity'].min()
    sensitivity = velocity_range / pressure_range
    
    # Calculate R-squared
    r_squared = model.rsquared
    
    return {
        'Participant': participant_id,
        'Stiffness_Index': stiffness_index,
        'Age_Effect': age_effect,
        'Compliance': compliance,
        'Baseline_Velocity': baseline_velocity,
        'Pressure_Sensitivity': sensitivity,
        'Model_R_Squared': r_squared
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
        metrics = calculate_participant_stiffness(participant_df, participant)
        participant_metrics.append(metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(participant_metrics)
    
    # Merge with original dataframe
    result_df = df.merge(metrics_df, on='Participant', how='left')
    
    return result_df

def main():
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_medians.csv')
    output_dir = os.path.join(cap_flow_path, 'results')
    df = pd.read_csv(data_filepath)

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
    
    # Save the enhanced dataframe
    output_filepath = os.path.join(cap_flow_path, 'summary_df_with_stiffness.csv')
    df.to_csv(output_filepath, index=False)

if __name__ == "__main__":
    main()