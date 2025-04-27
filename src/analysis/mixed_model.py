"""
Filename: src/analysis/mixed_model.py

This script performs mixed effects modeling to analyze how Age, Pressure, Sex, and SYS_BP 
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
    
    # Filter for control group (set01)
    controls_df = df[df['SET'] == 'set01'].copy()
    
    # Ensure numeric variables
    numeric_cols = ['Age', 'Pressure', 'SYS_BP', 'Video_Median_Velocity']
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
    
    # Model 1: Basic model with fixed effects only
    print("\nModel 1: Basic fixed effects")
    formula = "Video_Median_Velocity ~ Age + Pressure" # + Sex + SYS_BP"
    model = smf.mixedlm(formula, df, groups=df["Participant"])
    models['basic'] = model.fit()
    print(models['basic'].summary())
    
    # Model 2: Add interaction between Age and Pressure
    print("\nModel 2: Age-Pressure interaction")
    formula = "Video_Median_Velocity ~ Age * Pressure" # + Sex + SYS_BP"
    model = smf.mixedlm(formula, df, groups=df["Participant"])
    models['age_pressure_interaction'] = model.fit()
    print(models['age_pressure_interaction'].summary())
    
    # Model 3: Random slopes for Pressure
    print("\nModel 3: Random slopes for Pressure")
    formula = "Video_Median_Velocity ~ Age + Pressure" #+ Sex + SYS_BP"
    model = smf.mixedlm(formula, df, groups=df["Participant"], re_formula="~Pressure")
    models['random_slopes'] = model.fit()
    print(models['random_slopes'].summary())
    
    # Model 4: Log-transformed velocity
    print("\nModel 4: Log-transformed velocity")
    formula = "Log_Video_Median_Velocity ~ Age + Pressure" #+ Sex + SYS_BP"
    model = smf.mixedlm(formula, df, groups=df["Participant"])
    models['log_velocity'] = model.fit()
    print(models['log_velocity'].summary())
    
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
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'mixed_model')
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
        if model_name == 'random_slopes':
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
        
        if model_name == 'log_velocity':
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
    
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'mixed_model')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Age-Pressure Interaction Plot
    interaction_model = model_results['age_pressure_interaction']
    
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
            # 'Sex': df['Sex'].mode()[0],  # Use most common sex
            # 'SYS_BP': df['SYS_BP'].mean()  # Use mean SYS_BP
        })
        
        # Get predicted values
        predicted = interaction_model.predict(pred_data)
        
        plt.plot(age_range, predicted, label=f'Pressure = {pressure} PSI')
    
    plt.xlabel('Age (years)')
    plt.ylabel('Predicted Blood Flow Velocity')
    plt.title('Age-Pressure Interaction Effect on Blood Flow')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'age_pressure_interaction.png'))
    plt.close()
    
    # 2. Random Slopes Visualization
    random_slopes_model = model_results['random_slopes']
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
        intercept = re[0]
        slope = re[1]
        
        # Calculate predicted values for this participant
        predicted = (random_slopes_model.fe_params['Intercept'] + intercept + 
                    (random_slopes_model.fe_params['Pressure'] + slope) * pressure_range)
        
        ax1.plot(pressure_range, predicted, alpha=0.5, label=f'Participant {participant}')
    
    ax1.set_xlabel('Pressure (PSI)')
    ax1.set_ylabel('Predicted Blood Flow Velocity')
    ax1.set_title('Individual Participant Pressure Responses\n(Random Slopes)')
    ax1.grid(True, alpha=0.3)
    
    # Plot the distribution of slopes
    slopes = [re[1] for re in random_effects.values()]
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
    
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'mixed_model')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the interaction model which includes Age*Pressure interaction
    interaction_model = model_results['age_pressure_interaction']
    
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
    mean_sys_bp = df['SYS_BP'].mean()
    mode_sex = df['Sex'].mode()[0]
    
    # Plot predicted values for each age
    for age in age_values:
        # Create prediction data
        pred_data = pd.DataFrame({
            'Age': age,
            'Pressure': pressure_range,
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
    plt.title('Effect of Pressure on Blood Flow at Different Ages')
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
    plt.title(f'Effect of Pressure on Blood Flow at Age {middle_age}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pressure_vs_velocity_with_ci.png'), dpi=300)
    plt.close()

def main():
    """Main function to run the mixed effects analysis."""
    print("\nRunning mixed effects analysis for blood flow data...")
    
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
    
    print("\nMixed effects analysis complete.")
    return 0

if __name__ == "__main__":
    main() 