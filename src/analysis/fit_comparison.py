"""Module for comparing different fit models for velocity-pressure relationships.

This module loads capillary flow data and tests various fit models (exponential,
linear, polynomial) to determine which best describes the relationship between
blood velocity and external pressure.
"""

import platform
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from typing import Dict, Tuple, List, Callable

# Get the hostname of the computer
hostname = platform.node()

# Dictionary mapping hostnames to folder paths
computer_paths = {
    "LAPTOP-I5KTBOR3": {
        'cap_flow': 'C:\\Users\\gt8ma\\capillary-flow',
    },
    "Quake-Blood": {
        'cap_flow': "C:\\Users\\gt8mar\\capillary-flow",
    },
}

# Set default paths
default_paths = {
    'cap_flow': "/hpc/projects/capillary-flow",
}

# Get the paths for the current computer
paths = computer_paths.get(hostname, default_paths)
cap_flow_path = paths['cap_flow']

def load_data() -> pd.DataFrame:
    """Load and preprocess the capillary flow data.
    
    Returns:
        pd.DataFrame: Processed dataframe containing velocity and pressure data.
    """
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    return pd.read_csv(data_filepath)

def exponential_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential decay function."""
    return a * np.exp(-b * x) + c

def polynomial_fit(x: np.ndarray, *params: List[float]) -> np.ndarray:
    """Polynomial function of arbitrary degree.
    
    Args:
        x: Input pressure values.
        params: Coefficients of the polynomial.
    
    Returns:
        np.ndarray: Predicted velocity values.
    """
    return sum(p * x**i for i, p in enumerate(params))

def fit_models(pressure: np.ndarray, velocity: np.ndarray, velocity_type: str) -> Dict[str, Tuple]:
    """
    Fit different models to the pressure-velocity data.
    
    Args:
        pressure: Array of pressure values.
        velocity: Array of velocity values.
        velocity_type: Type of velocity to fit.
    
    Returns:
        Dict: Dictionary containing model fits and metrics.
    """
    models = {}
    
    # Only attempt fits if we have enough points
    if len(pressure) < 3:
        print(f"Warning: Not enough points for fitting")
        return models
    
    # Linear fit
    linear_params = np.polyfit(pressure, velocity, 1)
    linear_pred = np.polyval(linear_params, pressure)
    models['Linear'] = {
        'params': linear_params,
        'r2': r2_score(velocity, linear_pred),
        'mse': mean_squared_error(velocity, linear_pred),
        'predict': lambda x: np.polyval(linear_params, x),
        'velocity_type': velocity_type
    }
    
    # Log-linear fit (only if all velocities are positive)
    if np.all(velocity > 0):
        log_velocity = np.log(velocity)
        log_params = np.polyfit(pressure, log_velocity, 1)
        log_pred = np.exp(np.polyval(log_params, pressure))
        models['Log-Linear'] = {
            'params': log_params,
            'r2': r2_score(velocity, log_pred),
            'r2_log': r2_score(log_velocity, np.polyval(log_params, pressure)),
            'mse': mean_squared_error(velocity, log_pred),
            'predict': lambda x: np.exp(np.polyval(log_params, x)),
            'velocity_type': velocity_type
        }
    
    # Exponential fit
    try:
        # Better initial parameter guesses
        v_range = np.ptp(velocity)
        v_min = np.min(velocity)
        p0 = [v_range, 0.1, v_min]  # [amplitude, decay rate, offset]
        
        # Add bounds to keep parameters reasonable
        bounds = (
            [0, 0, -np.inf],  # lower bounds
            [np.inf, 10, np.inf]  # upper bounds
        )
        
        exp_params, _ = curve_fit(
            exponential_decay, 
            pressure, 
            velocity,
            p0=p0,
            bounds=bounds,
            maxfev=2000  # increase max iterations
        )
        
        exp_pred = exponential_decay(pressure, *exp_params)
        r2 = r2_score(velocity, exp_pred)
        
        # Only keep fit if it's reasonable
        if r2 > 0:  # or use some other quality threshold
            models['Exponential'] = {
                'params': exp_params,
                'r2': r2,
                'mse': mean_squared_error(velocity, exp_pred),
                'predict': lambda x: exponential_decay(x, *exp_params),
                'velocity_type': velocity_type
            }
    except Exception as e:
        print(f"Warning: Could not fit exponential for {velocity_type}: {str(e)}")
    
    # Polynomial fit (degree 2)
    poly_params = np.polyfit(pressure, velocity, 2)
    poly_pred = np.polyval(poly_params, pressure)
    models['Polynomial_2'] = {
        'params': poly_params,
        'r2': r2_score(velocity, poly_pred),
        'mse': mean_squared_error(velocity, poly_pred),
        'predict': lambda x, p=poly_params: np.polyval(p, x),
        'velocity_type': velocity_type
    }
    
    return models

def plot_fits(pressure: np.ndarray, velocity: np.ndarray, 
             models: Dict, participant_id: str, velocity_type: str, output_dir: str):
    """Plot the data and fitted models."""
    plt.figure(figsize=(12, 7))
    
    # Plot raw data
    plt.scatter(pressure, velocity, color='black', alpha=0.5, 
                label=f'Data ({velocity_type})', s=30)
    
    # Plot fitted curves
    x_fit = np.linspace(min(pressure), max(pressure), 100)
    colors = ['blue', 'cyan', 'red', 'green', 'purple', 'orange']
    for (name, model), color in zip(models.items(), colors):
        y_fit = model['predict'](x_fit)
        r2_label = f" (R² = {model['r2']:.3f})"
        if name == 'Log-Linear':
            r2_label += f" (R²_log = {model['r2_log']:.3f})"
        plt.plot(x_fit, y_fit, color=color, label=f'{name}{r2_label}')
    
    plt.xlabel('Pressure (psi)')
    plt.ylabel(f'Velocity ({velocity_type}) (μm/s)')
    plt.title(f'Model Fits for {participant_id} using {velocity_type} velocities')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f'fits_{participant_id}_{velocity_type}.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """Compare different models across all participants using different velocity aggregations."""
    results = []
    output_dir = os.path.join(cap_flow_path, 'results', 'Decay_fits')
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    participant_fits_dir = os.path.join(output_dir, 'participant_fits')
    os.makedirs(participant_fits_dir, exist_ok=True)
    
    for participant in df['Participant'].unique():
        participant_df = df[df['Participant'] == participant]
        
        # Test different velocity aggregations
        velocity_methods = {
            'max': lambda x: x.max(),
            'median': lambda x: x.median(),
            'mean': lambda x: x.mean()
        }
        
        for method_name, method_func in velocity_methods.items():
            # Get velocity for each pressure using current aggregation method
            velocities = participant_df.groupby('Pressure')['Video_Median_Velocity'].agg(method_func)
            pressure = velocities.index.values
            velocity = velocities.values
            
            # Fit models
            models = fit_models(pressure, velocity, method_name)
            
            # Plot fits
            plot_fits(pressure, velocity, models, participant, method_name, participant_fits_dir)
            
            # Store results
            for model_name, model_info in models.items():
                results.append({
                    'Participant': participant,
                    'Model': model_name,
                    'Velocity_Type': method_name,
                    'R2': model_info['r2'],
                    'MSE': model_info['mse'],
                    'R2_log': model_info.get('r2_log', None)  # Only for log-linear
                })
    
    return pd.DataFrame(results)

def plot_model_comparison(results_df: pd.DataFrame, output_dir: str):
    """Plot comparison of model performances."""
    # Box plot of R² scores by velocity type
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='Model', y='R2', hue='Velocity_Type')
    plt.title('Model R² Score Comparison by Velocity Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_r2_comparison.png'))
    plt.close()
    
    # Box plot of MSE by velocity type
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='Model', y='MSE', hue='Velocity_Type')
    plt.title('Model MSE Comparison by Velocity Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_mse_comparison.png'))
    plt.close()

def main():
    """Main function to run the analysis."""
    print("\nStarting model comparison analysis...")
    
    # Load data
    df = load_data()
    
    # Compare models
    results_df = compare_models(df)
    
    # Plot comparisons
    output_dir = os.path.join(cap_flow_path, 'results', 'Decay_fits')
    os.makedirs(output_dir, exist_ok=True)
    plot_model_comparison(results_df, output_dir)
    
    # Print summary statistics
    print("\nModel Performance Summary:")
    summary = results_df.groupby(['Model', 'Velocity_Type']).agg({
        'R2': ['mean', 'std'],
        'MSE': ['mean', 'std']
    }).round(3)
    print(summary)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)
    summary.to_csv(os.path.join(output_dir, 'model_comparison_summary.csv'))
    
    return results_df

if __name__ == "__main__":
    results = main()
