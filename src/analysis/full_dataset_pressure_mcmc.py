"""
Filename: resistance.py
------------------------------

By: Marcus Forst
"""
import os, sys, platform
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from src.analysis.plot_big import create_monochromatic_palette, adjust_brightness_of_colors, calculate_median_ci, calculate_stats
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pymc as pm
import arviz as az

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


# Determine the folder path based on the hostname
cap_flow_path = cap_flow_folder_paths.get(hostname, default_folder_path)


def plot_CI(df, method='bootstrap', n_iterations=1000, 
            ci_percentile=99.5, write=True, dimensionless=False, video_median=False, log_scale=False):
    """Plots the mean/median and CI for a single dataset."""
    # Set up style and font
    sns.set_style("whitegrid")
    source_sans = FontProperties(fname='C:\\Users\\gt8mar\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

    # Set base color
    base_color = '#1f77b4'  # Default blue color
    palette = create_monochromatic_palette(base_color)
    palette = adjust_brightness_of_colors(palette, brightness_scale=.2)
    sns.set_palette(palette)

    if log_scale:
        df['Corrected Velocity'] = df['Corrected Velocity'] + 10

    if video_median:
        df = df.groupby(['Participant', 'Video', 'Capillary']).first().reset_index()
        df.rename(columns={'Dimensionless Velocity': 'Dimensionless Velocity OG'}, inplace=True) 
        df.rename(columns={'Video Median Dimensionless Velocity': 'Dimensionless Velocity'}, inplace=True)      

    # Calculate stats
    stats_func = calculate_median_ci if method == 'bootstrap' else calculate_stats
    stats_df = df.groupby(['Pressure']).apply(stats_func, ci_percentile=ci_percentile, dimensionless=dimensionless).reset_index()

    # Plot
    plt.close()
    fig, ax = plt.subplots(figsize=(2.4, 2.0))

    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

    if dimensionless:
        y_col = 'Median Dimensionless Velocity' if method == 'bootstrap' else 'Mean Dimensionless Velocity'
    else:
        y_col = 'Median Velocity' if method == 'bootstrap' else 'Mean Velocity'
    lower_col = 'CI Lower Bound' if method == 'bootstrap' else 'Lower Bound'
    upper_col = 'CI Upper Bound' if method == 'bootstrap' else 'Upper Bound'

    # Plot data
    ax.errorbar(stats_df['Pressure'], stats_df[y_col],
                yerr=[stats_df[y_col] - stats_df[lower_col], 
                      stats_df[upper_col] - stats_df[y_col]],
                fmt='-o', markersize=2, color=palette[0])
    ax.fill_between(stats_df['Pressure'], stats_df[lower_col], 
                    stats_df[upper_col], alpha=0.4, color=palette[0])

    # Add log scale if requested
    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    ax.set_xlabel('Pressure (psi)', fontproperties=source_sans)
    if dimensionless:
        ax.set_ylabel('Dimensionless Velocity', fontproperties=source_sans)
        ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Dimensionless Velocity vs. Pressure with {ci_percentile}% CI', 
                    fontproperties=source_sans, fontsize=8)
    else:
        ax.set_ylabel('Velocity (um/s)', fontproperties=source_sans)
        ax.set_title(f'{"Median" if method == "bootstrap" else "Mean"} Velocity vs. Pressure with {ci_percentile}% CI', 
                    fontproperties=source_sans, fontsize=8)
    
    ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    if write:
        if video_median:
            plt.savefig(os.path.join(cap_flow_path, 'results', f'single_dataset_videomedians_CI.png'), dpi=600)
        else:
            plt.savefig(os.path.join(cap_flow_path, 'results', f'single_dataset_CI.png'), dpi=600)
    else:
        plt.show()
    return 0

def flow_decay(P, Q0, k):
    return Q0 * (1 - P / k)**4

def flow_threshold(P, Q0, k, P_th):
    return np.where(P <= P_th, Q0, Q0 * (1 - (P - P_th) / k)**4)

def flow_sigmoid(P, Q_max, alpha, P0):
    return Q_max / (1 + np.exp(-alpha * (P - P0)))

def flow_piecewise(P, Q0, beta, P_static, P_plateau):
    return np.piecewise(P, 
        [P <= P_static, (P > P_static) & (P < P_plateau), P >= P_plateau],
        [lambda P: Q0, 
         lambda P: Q0 * np.exp(-beta * (P - P_static)), 
         lambda P: Q0 * 0.1])  # Plateau flow (0.1 as example)

def plot_random_effects_histogram(random_effects_df, effect_column, title="Random Effects Histogram"):
    """
    Plot a histogram of random effects.

    Parameters:
    random_effects_df (pd.DataFrame): DataFrame containing random effects data.
    effect_column (str): Name of the column with random effects (e.g., intercept or slope).
    title (str): Title of the histogram.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(random_effects_df[effect_column], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel(effect_column, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    plt.show()

def plot_random_intercepts_and_slopes(random_effects_df, participant_column, intercept_column, slope_column):
    """
    Plot random intercepts and slopes for participants.

    Parameters:
    random_effects_df (pd.DataFrame): DataFrame containing random effects data.
    participant_column (str): Name of the column with participant identifiers.
    intercept_column (str): Name of the column with random intercepts.
    slope_column (str): Name of the column with random slopes.
    """
    plt.figure(figsize=(10, 6))
    for _, row in random_effects_df.iterrows():
        plt.plot(
            [0, 1], 
            [row[intercept_column], row[intercept_column] + row[slope_column]], 
            label=f"{row[participant_column]}"
        )

    plt.xlabel("Pressure (Standardized)", fontsize=12)
    plt.ylabel("Log_Video_Median_Velocity", fontsize=12)
    plt.title("Random Intercepts and Slopes for Participants", fontsize=14)
    plt.grid(True)
    plt.show()

def mixed_effects_module(mixed_results):
    # Extract random effects
    random_effects = mixed_results.random_effects
    random_effects_list = [
        {"Participant": key, "Intercept": value[0], "Pressure": value[1]}
        for key, value in random_effects.items()
    ]
    random_effects_df = pd.DataFrame(random_effects_list)
    
    # Plot random intercepts and slopes
    plot_random_intercepts_and_slopes(
        random_effects_df,
        participant_column="Participant",
        intercept_column="Intercept",
        slope_column="Pressure"
    )

    # Plot histogram of random intercepts
    plot_random_effects_histogram(
        random_effects_df,
        effect_column="Intercept",
        title="Random Intercepts Histogram"
    )

    # Plot histogram of random slopes
    plot_random_effects_histogram(
        random_effects_df,
        effect_column="Pressure",
        title="Random Slopes Histogram"
    )
    return 0

def plot_group_specific_effects(trace):
    """
    Plot group-specific effects from the trace.

    Args:
        trace (pymc.backends.base.MultiTrace): Trace object from PyMC sampling.
    Returns:
        int: 0 if successful.
    """
    # Extract group-specific effects from the trace
    group_intercepts = trace.posterior['Group_Intercepts'].mean(dim=["chain", "draw"]).values
    group_slopes = trace.posterior['Group_Slopes'].mean(dim=["chain", "draw"]).values

    # Plot group intercepts
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(group_intercepts)), group_intercepts, color='skyblue', alpha=0.8)
    plt.axhline(trace.posterior['Intercept'].mean().values, color='red', linestyle='--', label="Overall Intercept")
    plt.xlabel("Group Index")
    plt.ylabel("Intercept Value")
    plt.title("Group-Specific Intercepts")
    plt.legend()
    plt.show()

    # Plot group slopes
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(group_slopes)), group_slopes, color='lightgreen', alpha=0.8)
    plt.axhline(trace.posterior['Slope'].mean().values, color='red', linestyle='--', label="Overall Slope")
    plt.xlabel("Group Index")
    plt.ylabel("Slope Value")
    plt.title("Group-Specific Slopes")
    plt.legend()
    plt.show()
    return 0

def plot_posterior_predictive_checks(idata, model):
    """
    Plot posterior predictive checks for the Bayesian model.
    
    Args:
        idata (arviz.InferenceData): InferenceData object containing the posterior samples
        model (pymc.Model): PyMC model object.
    
    Returns:
        int: 0 if successful.
    """
    with model:
        ppc = pm.sample_posterior_predictive(idata, var_names=["y_obs"], random_seed=42)
        
    # Add posterior predictive samples to the InferenceData object
    idata.extend(ppc)
    
    # Plot posterior predictive distribution
    az.plot_ppc(idata, figsize=(10, 5))
    plt.title("Posterior Predictive Checks")
    plt.show()
    
    # For slope
    slope_samples = idata.posterior['Slope'].values.flatten()
    p_value_slope = (slope_samples > 0).mean() if idata.posterior['Slope'].mean() < 0 else (slope_samples < 0).mean()
    print(f"Bayesian p-value for Slope: {p_value_slope:.4f}")
    
    return 0



def bayes_module(data):
    """
    Bayesian mixed-effects model for the capillary flow data.

    Args:
        data (pd.DataFrame): DataFrame containing the capillary flow data.

    Returns:
        int: 0 if successful.
    """ 
    # Prepare data
    n_groups = data["Participant"].nunique()
    group_idx = data["Participant"].astype("category").cat.codes.values

    # Define the Bayesian model
    with pm.Model() as model:
        # Priors
        intercept = pm.HalfNormal("Intercept", sigma=10)  # Always positive
        slope = pm.Normal("Slope", mu=0, sigma=5)
        
        # Random effects
        group_intercepts = pm.Normal("Group_Intercepts", mu=intercept, sigma=2, shape=n_groups)
        group_slopes = pm.Normal("Group_Slopes", mu=slope, sigma=2, shape=n_groups)
        
        # Likelihood
        mu = group_intercepts[group_idx] + group_slopes[group_idx] * data["Pressure"]
        sigma = pm.HalfNormal("Sigma", sigma=2)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=data["Log_Video_Median_Velocity"])
        
        # Sampling
        trace = pm.sample(2000, return_inferencedata=True)

        # idata = pm.sample(2000, tune=1000, return_inferencedata=True)  # return_inferencedata=True is default in newer versions

        az.plot_posterior(trace)
        plt.show()
        az.summary(trace)
        plt.show()
        print(az.summary(trace))

        # Plot group-specific effects
        plot_group_specific_effects(trace)

        # Plot posterior predictive checks
        plot_posterior_predictive_checks(trace, model)
        return 0
    
def main():
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_medians.csv')
    data = pd.read_csv(data_filepath)
    print(data.columns)
    plot_CI(data, method='bootstrap', n_iterations=1000, ci_percentile=99.5, write=True, dimensionless=False, video_median=True, log_scale=False)

    mixed_model = smf.mixedlm('Log_Video_Median_Velocity ~ Pressure', data, groups=data['Participant'], re_formula='~Pressure') #re_formula=1  #family=sm.families.Poisson()
    mixed_results = mixed_model.fit()  
    print('Mixed Model Results for Pressure:')
    print(mixed_results.summary())

    

    # bayes_module(data)
    

    

    # normal_group = data[data['SET'] == 'set01']
    # # print min and max age
    # print(f"min age: {normal_group['Age'].min()}")
    # print(f"max age: {normal_group['Age'].max()}")
    # normal_group_old = normal_group[normal_group['Age']>50]

    # pressure_data = normal_group_old['Pressure']
    # velocity_data = normal_group_old['Video Median Velocity']

    # # Model List
    # model_list = ['decay', 'threshold', 'sigmoid', 'piecewise']
    
    # # Initial guesses for parameters
    # Q0_guess_decay = 600
    # k_guess_decay = 1.5
    # k_guess_threshold = 1.0
    # input_params_decay = [Q0_guess_decay, k_guess_decay]
    # input_params_threshold = [Q0_guess_decay, k_guess_decay, 0.3]
    # input_params_sigmoid = [Q0_guess_decay, 1.0, 0.5]
    # input_params_piecewise = [Q0_guess_decay, -300, 0.4, 1.0]

    # # Example: Fit fourth-power decay model
    # for model in model_list:
    #     if model == 'decay':
    #         popt, pcov = curve_fit(flow_decay, pressure_data, velocity_data, p0=input_params_decay)
    #     elif model == 'threshold':
    #         popt, pcov = curve_fit(flow_threshold, pressure_data, velocity_data, p0=input_params_threshold)
    #     elif model == 'sigmoid':
    #         popt, pcov = curve_fit(flow_sigmoid, pressure_data, velocity_data, p0=input_params_sigmoid, maxfev=5000)
    #     elif model == 'piecewise':
    #         popt, pcov = curve_fit(flow_piecewise, pressure_data, velocity_data, p0=input_params_piecewise)
    #     else:
    #         print(f"Model {model} not recognized")
    #         continue

    #     # Extract parameters
    #     print(f"Model: {model}")
    #     print(f"Parameters: {popt}")
    #     print(f"the pcov is {pcov}")
    # # popt, pcov = curve_fit(flow_decay, pressure_data, velocity_data, p0=[Q0_guess, k_guess])
    # # print(popt) # Generate predictions
    #     P_vals = np.linspace(0, max(pressure_data), 100)

    #     # Extract parameters
    #     if model == 'decay':
    #         Q0, k = popt
    #         Q_vals = flow_decay(P_vals, Q0, k)
    #     elif model == 'threshold':
    #         Q0, k, P_th = popt
    #         Q_vals = flow_threshold(P_vals, Q0, k, P_th)
    #     elif model == 'sigmoid':
    #         Q0, alpha, P0 = popt
    #         Q_vals = flow_sigmoid(P_vals, Q0, alpha, P0)
    #     elif model == 'piecewise':
    #         Q0, beta, P_static, P_plateau = popt
    #         Q_vals = flow_piecewise(P_vals, Q0, beta, P_static, P_plateau)
    #     else:
    #         print(f"Model {model} not recognized")
    #         continue

       

    #     # Plot
    #     plt.figure(figsize=(10, 6))
    #     # plt.scatter(pressure_data, velocity_data, label='Observed')
    #     # sns.violinplot(x=pressure_data, y=velocity_data)
    #     sns.boxplot(x=pressure_data, y=velocity_data)
    #     plt.plot(P_vals, Q_vals, label=f'Model Fit {model}', color='red')
    #     plt.xlabel('Pressure (psi)')
    #     plt.ylabel('Velocity (um/s)')
    #     plt.legend()
    #     plt.show()
    return 0

if __name__ == '__main__':
    main()