"""
Filename: src/analysis/height_weight_mixed_model.py

This script performs mixed effects modeling to analyze how Age, Pressure, Sex, SYS_BP, Height, and Weight 
affect blood flow while accounting for participant-level random effects.

The script:
1. Loads and preprocesses the data
2. Fits mixed effects models with different specifications
3. Creates diagnostic plots
4. Reports results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.stats import norm

# Import paths from config
from src.config import PATHS, load_source_sans

def load_and_preprocess_data() -> pd.DataFrame:
    """
    Loads and preprocesses the data for mixed effects modeling.
    
    Returns:
        DataFrame containing the preprocessed data
    """
    print("\nLoading and preprocessing data...")
    
    # Load data
    data_filepath = os.path.join(PATHS['cap_flow'], 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)

    height_weight_supplement = pd.read_csv(os.path.join(PATHS['cap_flow'], 'height_weight.csv'))
    
    # Merge height and weight data with main dataframe
    df = pd.merge(df, height_weight_supplement, on='Participant', how='outer')

    # Make Height_x and Height_y into Height
    df['Height'] = df['Height_x'].fillna(df['Height_y'])
    df = df.drop(columns=['Height_x', 'Height_y'])

    # Make Weight_x and Weight_y into Weight
    df['Weight'] = df['Weight_x'].fillna(df['Weight_y'])
    df = df.drop(columns=['Weight_x', 'Weight_y'])

    # Drop participants with missing height or weight
    df = df.dropna(subset=['Height', 'Weight'])
    
    # Filter for control group (set01)
    controls_df = df[df['SET'] == 'set01'].copy()
    
    # Ensure numeric variables
    numeric_cols = ['Age', 'Pressure', 'SYS_BP', 'Video_Median_Velocity', 'Height', 'Weight']
    for col in numeric_cols:
        controls_df[col] = pd.to_numeric(controls_df[col], errors='coerce')
    
    # Create log-transformed velocity for potential use
    controls_df['Log_Video_Median_Velocity'] = np.log(controls_df['Video_Median_Velocity'] + 1)
    
    # Reset index - critical for avoiding index errors in statsmodels
    controls_df = controls_df.reset_index(drop=True)
    
    # Print basic statistics
    print("\nData Summary:")
    print(f"Number of observations: {len(controls_df)}")
    print(f"Number of unique participants: {controls_df['Participant'].nunique()}")
    print("\nVariable ranges:")
    for col in numeric_cols:
        print(f"{col}: {controls_df[col].min():.2f} to {controls_df[col].max():.2f} (mean: {controls_df[col].mean():.2f})")
    
    # Check for missing values in key columns
    missing_values = controls_df[numeric_cols + ['Participant', 'Sex']].isnull().sum()
    print("\nMissing values in key columns:")
    print(missing_values)
    
    # Drop rows with missing values in key variables
    model_vars = numeric_cols + ['Participant', 'Sex']
    controls_df = controls_df.dropna(subset=model_vars)
    print(f"\nObservations after dropping missing values: {len(controls_df)}")
    
    return controls_df

def fit_mixed_models(df: pd.DataFrame) -> Dict:
    """
    Fits several mixed effects models with different specifications.
    
    Args:
        df: DataFrame containing the preprocessed data
        
    Returns:
        Dictionary containing the fitted model results
    """
    print("\nFitting mixed effects models...")
    
    models = {}
    
    # Model 1: Basic model with fixed effects including height and weight
    print("\nModel 1: Basic fixed effects with height and weight")
    formula = "Video_Median_Velocity ~ Age + Pressure + Height + Weight" # + Sex + SYS_BP"
    model = smf.mixedlm(formula, df, groups=df["Participant"])
    models['basic_hw'] = model.fit()
    print(models['basic_hw'].summary())
    
    # Model 2: Add interaction between Age and Pressure
    print("\nModel 2: Age-Pressure interaction with height and weight")
    formula = "Video_Median_Velocity ~ Age * Pressure + Height + Weight" # + Sex + SYS_BP"
    model = smf.mixedlm(formula, df, groups=df["Participant"])
    models['age_pressure_hw'] = model.fit()
    print(models['age_pressure_hw'].summary())
    
    # Model 3: Random slopes for Pressure
    print("\nModel 3: Random slopes for Pressure with height and weight")
    formula = "Video_Median_Velocity ~ Age + Pressure + Height + Weight" #+ Sex + SYS_BP"
    model = smf.mixedlm(formula, df, groups=df["Participant"], re_formula="~Pressure")
    models['random_slopes_hw'] = model.fit()
    print(models['random_slopes_hw'].summary())
    
    # Model 4: Log-transformed velocity
    print("\nModel 4: Log-transformed velocity with height and weight")
    formula = "Log_Video_Median_Velocity ~ Age + Pressure + Height + Weight" #+ Sex + SYS_BP"
    model = smf.mixedlm(formula, df, groups=df["Participant"])
    models['log_velocity_hw'] = model.fit()
    print(models['log_velocity_hw'].summary())
    
    # Model 5: Log-transformed velocity with random slopes for Pressure
    print("\nModel 5: Log-transformed velocity with random slopes for Pressure, height, and weight")
    formula = "Log_Video_Median_Velocity ~ Age + Pressure + Height + Weight"  # + Sex + SYS_BP
    model = smf.mixedlm(formula, df, groups=df["Participant"], re_formula="~Pressure")
    models['log_random_slopes_hw'] = model.fit()
    print(models['log_random_slopes_hw'].summary())
    
    return models

def plot_diagnostics(df: pd.DataFrame, model_results: Dict) -> None:
    """
    Creates diagnostic plots for the mixed effects models.
    
    Args:
        df: DataFrame containing the data
        model_results: Dictionary of fitted model results
    """
    print("\nCreating diagnostic plots...")
    
    # Set up plotting style - using seaborn's set function instead of plt.style.use
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14
    })
    
    # Create output directory
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'mixed_model', 'height_weight')
    os.makedirs(output_dir, exist_ok=True)
    
    # For each model
    for model_name, result in model_results.items():
        print(f"\nCreating plots for {model_name} model...")
        
        # Get residuals and fitted values
        residuals = result.resid
        fitted = result.fittedvalues
        
        # 1. Basic diagnostic plots (2x2 grid)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Diagnostic Plots - {model_name} Model')
        
        # Residuals vs fitted
        sns.scatterplot(x=fitted, y=residuals, ax=axes[0])
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Fitted values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Fitted')
        
        # QQ plot
        sm.graphics.qqplot(residuals, line='45', fit=True, ax=axes[1])
        axes[1].set_title('Normal Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_diagnostics.png'))
        plt.close()
        
        # 2. Additional residual plots
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Residuals histogram with density
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_title(f'Residual Distribution - {model_name} Model')
        ax.set_xlabel('Residual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_residual_dist.png'))
        plt.close()
        
        # 3. Random effects plots (for models with random slopes)
        if model_name in ['random_slopes_hw', 'log_random_slopes_hw']:
            random_effects = result.random_effects
            re_df = pd.DataFrame.from_dict(random_effects, orient='index')
            re_df.columns = ['Intercept', 'Pressure_Slope']
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Random Effects Distribution')
            
            # Random intercepts
            sns.histplot(re_df['Intercept'], kde=True, ax=axes[0])
            axes[0].set_title('Random Intercepts')
            axes[0].set_xlabel('Value')
            
            # Random slopes
            sns.histplot(re_df['Pressure_Slope'], kde=True, ax=axes[1])
            axes[1].set_title('Random Pressure Slopes')
            axes[1].set_xlabel('Value')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_random_effects.png'))
            plt.close()
            
        # 4. Observed vs predicted plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if model_name in ['log_velocity_hw', 'log_random_slopes_hw']:
            observed = np.exp(df['Log_Video_Median_Velocity']) - 1
            predicted = np.exp(fitted) - 1
            ax.set_title(f'Observed vs Predicted (Back-transformed) - {model_name} Model')
        else:
            observed = df['Video_Median_Velocity']
            predicted = fitted
            ax.set_title(f'Observed vs Predicted - {model_name} Model')
            
        # Plot the points
        sns.scatterplot(x=observed, y=predicted, ax=ax)
        
        # Add a 45-degree reference line
        min_val = min(observed.min(), predicted.min())
        max_val = max(observed.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_predicted_vs_observed.png'))
        plt.close()

def plot_interaction_effects(df: pd.DataFrame, model_results: Dict) -> None:
    """
    Creates visualization plots for the age-pressure interaction and random slopes effects.
    
    Args:
        df: DataFrame containing the data
        model_results: Dictionary of fitted model results
    """
    print("\nCreating interaction effects plots...")
    
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'mixed_model', 'height_weight')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Age-Pressure Interaction Plot
    interaction_model = model_results['age_pressure_hw']
    
    # Create grid of age and pressure values
    age_range = np.linspace(df['Age'].min(), df['Age'].max(), 100)
    pressure_values = [0.2, 0.4, 0.8, 1.2]  # Common pressure values
    
    plt.figure(figsize=(10, 6))
    
    # Plot predicted values for each pressure
    for pressure in pressure_values:
        # Create prediction data
        pred_data = pd.DataFrame({
            'Age': age_range,
            'Pressure': pressure,
            'Height': df['Height'].mean(),
            'Weight': df['Weight'].mean(),
            # 'Sex': df['Sex'].mode()[0],  # Use most common sex
            # 'SYS_BP': df['SYS_BP'].mean()  # Use mean SYS_BP
        })
        
        # Get predicted values
        predicted = interaction_model.predict(pred_data)
        
        plt.plot(age_range, predicted, label=f'Pressure = {pressure} PSI')
    
    plt.xlabel('Age (years)')
    plt.ylabel('Predicted Blood Flow Velocity')
    plt.title('Age-Pressure Interaction Effect on Blood Flow (with avg Height/Weight)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'age_pressure_interaction.png'))
    plt.close()
    
    # 2. Random Slopes Visualization
    random_slopes_model = model_results['random_slopes_hw']
    random_effects = random_slopes_model.random_effects
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot individual participant slopes
    pressure_range = np.linspace(df['Pressure'].min(), df['Pressure'].max(), 100)
    
    # Plot for a subset of participants (e.g., first 10) to avoid overcrowding
    participant_subset = list(random_effects.keys())[:10]
    
    # Plot individual participant trajectories
    for participant in participant_subset:
        re = random_effects[participant]
        intercept = re.iloc[0]
        slope = re.iloc[1]
        
        # Calculate predicted values for this participant
        predicted = (random_slopes_model.fe_params['Intercept'] + intercept + 
                    (random_slopes_model.fe_params['Pressure'] + slope) * pressure_range)
        
        ax1.plot(pressure_range, predicted, alpha=0.5, label=f'Participant {participant}')
    
    ax1.set_xlabel('Pressure (PSI)')
    ax1.set_ylabel('Predicted Blood Flow Velocity')
    ax1.set_title('Individual Participant Pressure Responses\n(Random Slopes)')
    ax1.grid(True, alpha=0.3)
    
    # Plot the distribution of slopes
    slopes = [re.iloc[1] for re in random_effects.values()]
    sns.histplot(slopes, kde=True, ax=ax2)
    ax2.axvline(x=0, color='r', linestyle='--', label='Population Average')
    ax2.set_xlabel('Pressure Slope')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Individual Pressure Slopes')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'random_slopes_visualization.png'))
    plt.close()

def plot_pressure_vs_velocity_by_age(df: pd.DataFrame, model_results: Dict) -> None:
    """
    Creates a plot showing the relationship between pressure and flow velocity for different ages.
    
    Args:
        df: DataFrame containing the data
        model_results: Dictionary of fitted model results
    """
    print("\nCreating pressure vs flow velocity plot by age...")
    
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'mixed_model', 'height_weight')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the interaction model which includes Age*Pressure interaction
    interaction_model = model_results['age_pressure_hw']
    
    # Get age range for representative examples
    age_min = df['Age'].min()
    age_max = df['Age'].max()
    age_values = [
        np.ceil(age_min),  # Youngest age (rounded up)
        np.floor(age_min + (age_max - age_min) * 0.33),  # 1/3 through age range
        np.floor(age_min + (age_max - age_min) * 0.67),  # 2/3 through age range
        np.floor(age_max)  # Oldest age (rounded down)
    ]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create pressure range
    pressure_range = np.linspace(df['Pressure'].min(), df['Pressure'].max(), 100)
    
    # Get mean values for other predictors
    mean_height = df['Height'].mean()
    mean_weight = df['Weight'].mean()
    mean_sys_bp = df['SYS_BP'].mean()
    mode_sex = df['Sex'].mode()[0]
    
    # Plot predicted values for each age
    for age in age_values:
        # Create prediction data
        pred_data = pd.DataFrame({
            'Age': age,
            'Pressure': pressure_range,
            'Height': mean_height,
            'Weight': mean_weight,
            # 'Sex': mode_sex,
            # 'SYS_BP': mean_sys_bp
        })
        
        # Get predicted values
        predicted = interaction_model.predict(pred_data)
        
        plt.plot(pressure_range, predicted, linewidth=2, label=f'Age = {int(age)} years')
    
    # # Add scatter of actual data points with transparency
    # plt.scatter(df['Pressure'], df['Video_Median_Velocity'], alpha=0.3, 
    #             color='gray', label='Observed data')
    
    plt.xlabel('Pressure (PSI)')
    plt.ylabel('Blood Flow Velocity')
    plt.title('Effect of Pressure on Blood Flow at Different Ages (with avg Height/Weight)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pressure_vs_velocity_by_age.png'), dpi=300)
    plt.close()
    
    # Also create a similar plot but with confidence intervals for one middle age
    plt.figure(figsize=(10, 6))
    
    # Select a middle age
    middle_age = int(np.median(df['Age']))
    
    # Create prediction data for middle age
    pred_data = pd.DataFrame({
        'Age': middle_age,
        'Pressure': pressure_range,
        'Height': mean_height,
        'Weight': mean_weight,
        # 'Sex': mode_sex,
        # 'SYS_BP': mean_sys_bp
    })
    
    # Get predicted values
    predicted = interaction_model.predict(pred_data)
    
    # Try to get confidence intervals if possible (this might not work with all models)
    try:
        from statsmodels.sandbox.regression.predstd import wls_prediction_std
        _, lower, upper = wls_prediction_std(interaction_model, pred_data)
        
        plt.fill_between(pressure_range, lower, upper, alpha=0.3, color='blue', 
                        label=f'95% Confidence Interval')
        has_ci = True
    except:
        has_ci = False
    
    # Plot the main prediction line
    plt.plot(pressure_range, predicted, linewidth=2, color='blue', 
             label=f'Age = {middle_age} years')
    
    # Add scatter of actual data points with transparency
    plt.scatter(df['Pressure'], df['Video_Median_Velocity'], alpha=0.3, 
                color='gray', label='Observed data')
    
    plt.xlabel('Pressure (PSI)')
    plt.ylabel('Blood Flow Velocity')
    plt.title(f'Effect of Pressure on Blood Flow at Age {middle_age} (with avg Height/Weight)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pressure_vs_velocity_with_ci.png'), dpi=300)
    plt.close()

def write_summary_results(model_results: Dict) -> None:
    """
    Writes a summary of model results to a text file.
    
    Args:
        model_results: Dictionary of fitted model results
    """
    print("\nWriting summary results to text file...")
    
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'mixed_model', 'height_weight')
    os.makedirs(output_dir, exist_ok=True)
    
    summary_path = os.path.join(output_dir, 'model_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("MIXED EFFECTS MODEL ANALYSIS SUMMARY (WITH HEIGHT AND WEIGHT)\n")
        f.write("========================================================\n\n")
        
        # Write summary for each model
        for model_name, result in model_results.items():
            f.write(f"MODEL: {model_name}\n")
            f.write("=" * (len(model_name) + 7) + "\n\n")
            
            # Extract key statistics
            aic = result.aic
            bic = result.bic
            loglike = result.llf
            
            f.write(f"AIC: {aic:.2f}\n")
            f.write(f"BIC: {bic:.2f}\n")
            f.write(f"Log-Likelihood: {loglike:.2f}\n\n")
            
            # Extract fixed effect parameters
            f.write("Fixed Effects:\n")
            f.write("-" * 60 + "\n")
            f.write("{:<20} {:<12} {:<12} {:<12}\n".format("Parameter", "Estimate", "Std.Err", "P-value"))
            f.write("-" * 60 + "\n")
            
            # Access model parameters directly - MixedLMResults doesn't have params_names attribute
            fe_params = result.fe_params
            # Standard errors and p-values are directly available from the fitted result
            bse = result.bse  # Standard errors
            p_values = result.pvalues  # Two-sided p-values
            
            # Get parameter names from fe_params index
            for param_name in fe_params.index:
                estimate = fe_params[param_name]
                std_err = bse.get(param_name, np.nan)
                p_value = p_values.get(param_name, np.nan)

                if np.isnan(std_err) or np.isnan(p_value):
                    f.write("{:<20} {:<12.4f} {:<12} {:<12}\n".format(
                        param_name, estimate, "N/A", "N/A"))
                else:
                    f.write("{:<20} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                        param_name, estimate, std_err, p_value))
            
            f.write("-" * 60 + "\n\n")
            
            # Write random effects summary if available
            if hasattr(result, 'cov_re'):
                f.write("Random Effects Covariance Parameters:\n")
                f.write("-" * 60 + "\n")
                cov_re = np.asarray(result.cov_re)
                # If names are available and array is 1-D, pair names with values
                if hasattr(result, 'cov_re_names') and cov_re.ndim == 1:
                    for i, name in enumerate(result.cov_re_names):
                        f.write(f"{name}: {float(cov_re[i]):.4f}\n")
                else:
                    # Print the covariance matrix row-by-row
                    for row_idx in range(cov_re.shape[0]):
                        row_vals = ", ".join([f"{float(v):.4f}" for v in cov_re[row_idx].flatten()])
                        f.write(f"Row {row_idx + 1}: {row_vals}\n")
                f.write("-" * 60 + "\n\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"Summary results written to: {summary_path}")

def write_latex_table(model_results: Dict) -> None:
    """
    Creates a LaTeX table of model results.
    
    Args:
        model_results: Dictionary of fitted model results
    """
    print("\nCreating LaTeX table of model results...")
    
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'mixed_model', 'height_weight')
    os.makedirs(output_dir, exist_ok=True)
    
    latex_path = os.path.join(output_dir, 'model_results_table.tex')
    
    with open(latex_path, 'w') as f:
        # Write LaTeX preamble
        f.write("% LaTeX table for mixed effects model results with height and weight\n")
        f.write("% Generated by height_weight_mixed_model.py\n\n")
        
        # Begin table environment
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Mixed Effects Model Results for Blood Flow Analysis with Height and Weight}\n")
        f.write("\\label{tab:hw_mixed_model_results}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{l" + "r" * len(model_results) + "}\n")
        f.write("\\toprule\n")
        
        # Write header row with model names
        header = "Parameter"
        for model_name in model_results.keys():
            # Format model name for display (replace underscores with spaces, capitalize)
            display_name = model_name.replace('_', ' ').title()
            header += f" & {display_name}"
        f.write(f"{header} \\\\\n")
        f.write("\\midrule\n")
        
        # Helper to add significance stars
        def _star(p):
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            else:
                return ""

        # Get a unique set of all parameters across models
        all_params = set()
        for result in model_results.values():
            all_params.update(result.fe_params.index)

        # Fixed effects parameters
        f.write("\\multicolumn{" + str(len(model_results) + 1) + "}{l}{\\textbf{Fixed Effects}} \\\n")
        for param in sorted(all_params):
            row = param
            for _, result in model_results.items():
                if param in result.fe_params.index:
                    est = result.fe_params[param]
                    se = result.bse.get(param, np.nan)
                    pval = result.pvalues.get(param, np.nan)
                    star = _star(pval) if not np.isnan(pval) else ""
                    if np.isnan(se):
                        row += f" & {est:.4f}{star}"
                    else:
                        row += f" & {est:.4f}{star} ({se:.4f})"
                else:
                    row += " & --"
            f.write(f"{row} \\\n")

        # Random effects standard deviations
        f.write("\\midrule\n")
        f.write("\\multicolumn{" + str(len(model_results) + 1) + "}{l}{\\textbf{Random Effects (SD)}} \\\n")

        # Collect names of random effects across models
        re_names_set = set()
        for res in model_results.values():
            if hasattr(res, "cov_re"):
                cov = np.asarray(res.cov_re)
                if cov.ndim == 1:
                    re_names_set.add("Intercept")
                else:
                    names = getattr(res, "cov_re_names", None)
                    if names is None:
                        names = [f"RE{i+1}" for i in range(cov.shape[0])]
                    re_names_set.update(names)
        re_names = sorted(re_names_set)

        for re_name in re_names:
            row = re_name
            for _, res in model_results.items():
                if hasattr(res, "cov_re"):
                    cov = np.asarray(res.cov_re)
                    if cov.ndim == 1 and re_name == "Intercept":
                        sd = float(np.sqrt(cov[0]))
                        row += f" & {sd:.4f}"
                    else:
                        names = getattr(res, "cov_re_names", None)
                        if names is None:
                            names = [f"RE{i+1}" for i in range(cov.shape[0])]
                        if re_name in names:
                            idx = names.index(re_name)
                            sd = float(np.sqrt(cov[idx, idx]))
                            row += f" & {sd:.4f}"
                        else:
                            row += " & --"
                else:
                    row += " & --"
            f.write(f"{row} \\\n")

        # Residual SD
        row = "Residual SD"
        for res in model_results.values():
            row += f" & {np.sqrt(res.scale):.4f}"
        f.write(f"{row} \\\n")

        # Model fit statistics
        f.write("\\midrule\n")
        f.write("\\multicolumn{" + str(len(model_results) + 1) + "}{l}{\\textbf{Model Statistics}} \\\n")
        
        # AIC row
        row = "AIC"
        for result in model_results.values():
            row += f" & {result.aic:.2f}"
        f.write(f"{row} \\\n")
        
        # BIC row
        row = "BIC"
        for result in model_results.values():
            row += f" & {result.bic:.2f}"
        f.write(f"{row} \\\n")
        
        # Log-likelihood row
        row = "Log-Likelihood"
        for result in model_results.values():
            row += f" & {result.llf:.2f}"
        f.write(f"{row} \\\n")
        
        # Close table
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\begin{tablenotes}\n")
        f.write("\\small\n")
        f.write("\\item SD: standard deviation of random effect. Values in parentheses are standard errors. Significance codes: *** p<0.001, ** p<0.01, * p<0.05\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table written to: {latex_path}")

def plot_height_weight_effects(df: pd.DataFrame, model_results: Dict) -> None:
    """
    Creates plots to visualize the effects of height and weight on blood flow.
    
    Args:
        df: DataFrame containing the data
        model_results: Dictionary of fitted model results
    """
    print("\nCreating height and weight effects plots...")
    
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'mixed_model', 'height_weight')
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the basic model with height and weight
    model = model_results['basic_hw']
    
    # 1. Height effect plot
    plt.figure(figsize=(10, 6))
    
    # Create range of height values
    height_range = np.linspace(df['Height'].min(), df['Height'].max(), 100)
    
    # Get mean values for other predictors
    mean_age = df['Age'].mean()
    mean_weight = df['Weight'].mean()
    mean_pressure = df['Pressure'].mean()
    mean_height = df['Height'].mean()
    
    # Create prediction data
    pred_data = pd.DataFrame({
        'Age': mean_age,
        'Pressure': mean_pressure,
        'Height': height_range,
        'Weight': mean_weight
    })
    
    # Get predicted values
    predicted = model.predict(pred_data)
    
    # Plot the line
    plt.plot(height_range, predicted, linewidth=2)
    
    # Add scatter of actual data points with transparency
    plt.scatter(df['Height'], df['Video_Median_Velocity'], alpha=0.3, 
                color='gray', label='Observed data')
    
    plt.xlabel('Height (cm)')
    plt.ylabel('Blood Flow Velocity')
    plt.title('Effect of Height on Blood Flow (with average Age, Weight, and Pressure)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'height_effect.png'), dpi=300)
    plt.close()
    
    # 2. Weight effect plot
    plt.figure(figsize=(10, 6))
    
    # Create range of weight values
    weight_range = np.linspace(df['Weight'].min(), df['Weight'].max(), 100)
    
    # Create prediction data
    pred_data = pd.DataFrame({
        'Age': mean_age,
        'Pressure': mean_pressure,
        'Height': mean_height,
        'Weight': weight_range
    })
    
    # Get predicted values
    predicted = model.predict(pred_data)
    
    # Plot the line
    plt.plot(weight_range, predicted, linewidth=2)
    
    # Add scatter of actual data points with transparency
    plt.scatter(df['Weight'], df['Video_Median_Velocity'], alpha=0.3, 
                color='gray', label='Observed data')
    
    plt.xlabel('Weight (kg)')
    plt.ylabel('Blood Flow Velocity')
    plt.title('Effect of Weight on Blood Flow (with average Age, Height, and Pressure)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_effect.png'), dpi=300)
    plt.close()

def main():
    """Main function to run the mixed effects analysis."""
    print("\nRunning mixed effects analysis for blood flow data with height and weight...")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Fit mixed effects models
    model_results = fit_mixed_models(df)
    
    # Create diagnostic plots
    plot_diagnostics(df, model_results)
    
    # Create interaction effects plots
    plot_interaction_effects(df, model_results)
    
    # Create pressure vs velocity by age plot
    plot_pressure_vs_velocity_by_age(df, model_results)
    
    # Create height and weight effect plots
    plot_height_weight_effects(df, model_results)
    
    # Write summary results to text file
    write_summary_results(model_results)
    
    # Create LaTeX table
    write_latex_table(model_results)
    
    print("\nMixed effects analysis with height and weight complete.")
    return 0

if __name__ == "__main__":
    main() 