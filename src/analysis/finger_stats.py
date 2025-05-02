"""
Filename: src/analysis/finger_stats.py

File for analyzing finger-specific metrics in capillary velocity data.

This script:
1. Loads required data and fonts
2. Provides functions for analyzing finger size (bottom circumference)
3. Analyzes correlation between finger size and capillary velocity
4. Performs ANOVA to compare effects of different factors on velocity
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.font_manager import FontProperties
from typing import Dict, Tuple, List
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.formula.api import mixedlm

# Import paths from config instead of defining computer paths locally
from src.config import PATHS, load_source_sans

# Define constants
cap_flow_path = PATHS['cap_flow']
source_sans = load_source_sans()

def setup_plotting():
    """
    Sets up plotting style parameters for consistent figure appearance.
    """
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

def analyze_finger_size_correlation(merged_df: pd.DataFrame) -> Dict:
    """
    Analyze correlations between finger size (bottom circumference) and velocity.
    
    Args:
        merged_df: DataFrame containing both finger metrics and velocity data
        
    Returns:
        Dictionary of correlation results by pressure level
    """
    # Note: Pearson correlation does not account for the clustered nature 
    # (repeated measures) of the data. Results should be interpreted cautiously.
    # Group by pressure to analyze correlation at each pressure level
    pressure_levels = merged_df['Pressure'].unique()
    correlations = {}
    
    print("\nCalculating correlations between finger size (bottom) and velocity:")
    
    for pressure in pressure_levels:
        pressure_df = merged_df[merged_df['Pressure'] == pressure]
        
        # Ensure we have enough data for meaningful correlation
        if len(pressure_df) < 3 or pressure_df['FingerSizeBottom'].isna().all():
            print(f"- Insufficient data for pressure {pressure} psi")
            correlations[pressure] = {
                'r': np.nan,
                'p': np.nan,
                'n': len(pressure_df)
            }
            continue
        
        # Calculate correlation
        valid_data = pressure_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity'])
        corr_result = stats.pearsonr(
            valid_data['FingerSizeBottom'],
            valid_data['Video_Median_Velocity']
        )
        
        correlations[pressure] = {
            'r': corr_result[0],
            'p': corr_result[1],
            'n': len(valid_data)
        }
        
        print(f"- {pressure} psi: r = {corr_result[0]:.3f}, p = {corr_result[1]:.3f}, n = {len(valid_data)}")
    
    return correlations

def plot_finger_size_correlations(merged_df: pd.DataFrame, correlations: Dict, output_dir: str) -> None:
    """
    Create visualizations of finger size (bottom) correlation with velocity.
    
    Args:
        merged_df: DataFrame containing finger metrics and velocity data
        correlations: Dictionary of correlation results by pressure
        output_dir: Directory to save output plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot scatter for each pressure level with correlation
    for pressure in correlations.keys():
        pressure_df = merged_df[merged_df['Pressure'] == pressure]
        valid_data = pressure_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity'])
        
        if len(valid_data) < 3:
            continue
            
        plt.figure(figsize=(3.5, 2.5))
        
        # Create scatter plot
        sns.scatterplot(
            x='FingerSizeBottom', 
            y='Video_Median_Velocity',
            data=valid_data,
            hue='Finger',
            s=40
        )
        
        # Add regression line if we have enough data
        if len(valid_data) >= 3:
            sns.regplot(
                x='FingerSizeBottom', 
                y='Video_Median_Velocity',
                data=valid_data,
                scatter=False,
                ci=None,
                line_kws={'color': 'red', 'linestyle': '--'}
            )
        
        # Add correlation info
        r_value = correlations[pressure]['r']
        p_value = correlations[pressure]['p']
        n_value = correlations[pressure]['n']
        
        p_text = f"r = {r_value:.2f}, p = {p_value:.3f}, n = {n_value}" if p_value >= 0.001 else f"r = {r_value:.2f}, p < 0.001, n = {n_value}"
        
        if source_sans:
            plt.xlabel('Finger Bottom Circumference (mm)', fontproperties=source_sans)
            plt.ylabel('Velocity (mm/s)', fontproperties=source_sans)
            plt.title(f'Finger Size vs Velocity at {pressure} psi', fontproperties=source_sans)
            plt.text(0.05, 0.95, p_text, transform=plt.gca().transAxes, 
                   fontproperties=source_sans, fontsize=6)
        else:
            plt.xlabel('Finger Bottom Circumference (mm)')
            plt.ylabel('Velocity (mm/s)')
            plt.title(f'Finger Size vs Velocity at {pressure} psi')
            plt.text(0.05, 0.95, p_text, transform=plt.gca().transAxes, fontsize=6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'finger_size_vs_velocity_{pressure}psi.png'), dpi=300)
        plt.close()
    
    # Plot overall correlation across all pressures
    all_data = merged_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity'])
    
    if len(all_data) >= 3:
        plt.figure(figsize=(3.5, 2.5))
        
        # Create scatter plot
        sns.scatterplot(
            x='FingerSizeBottom', 
            y='Video_Median_Velocity',
            data=all_data,
            hue='Pressure',
            palette='viridis',
            s=40
        )
        
        # Add regression line
        sns.regplot(
            x='FingerSizeBottom', 
            y='Video_Median_Velocity',
            data=all_data,
            scatter=False,
            ci=None,
            line_kws={'color': 'red', 'linestyle': '--'}
        )
        
        # Calculate overall correlation
        corr_result = stats.pearsonr(all_data['FingerSizeBottom'], all_data['Video_Median_Velocity'])
        p_text = f"r = {corr_result[0]:.2f}, p = {corr_result[1]:.3f}, n = {len(all_data)}" if corr_result[1] >= 0.001 else f"r = {corr_result[0]:.2f}, p < 0.001, n = {len(all_data)}"
        
        if source_sans:
            plt.xlabel('Finger Bottom Circumference (mm)', fontproperties=source_sans)
            plt.ylabel('Velocity (mm/s)', fontproperties=source_sans)
            plt.title('Finger Size vs Velocity (All Pressures)', fontproperties=source_sans)
            plt.text(0.05, 0.95, p_text, transform=plt.gca().transAxes, 
                   fontproperties=source_sans, fontsize=6)
        else:
            plt.xlabel('Finger Bottom Circumference (mm)')
            plt.ylabel('Velocity (mm/s)')
            plt.title('Finger Size vs Velocity (All Pressures)')
            plt.text(0.05, 0.95, p_text, transform=plt.gca().transAxes, fontsize=6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'finger_size_vs_velocity_all_pressures.png'), dpi=300)
        plt.close()

def analyze_finger_size_log_correlation(merged_df: pd.DataFrame) -> Dict:
    """
    Analyze correlations between finger size (bottom circumference) and log-transformed velocity.
    
    Args:
        merged_df: DataFrame containing both finger metrics and velocity data
        
    Returns:
        Dictionary of correlation results by pressure level
    """
    # Group by pressure to analyze correlation at each pressure level
    pressure_levels = merged_df['Pressure'].unique()
    correlations = {}
    
    print("\nCalculating correlations between finger size (bottom) and log velocity:")
    
    for pressure in pressure_levels:
        pressure_df = merged_df[merged_df['Pressure'] == pressure]
        
        # Ensure we have enough data for meaningful correlation
        if len(pressure_df) < 3 or pressure_df['FingerSizeBottom'].isna().all():
            print(f"- Insufficient data for pressure {pressure} psi")
            correlations[pressure] = {
                'r': np.nan,
                'p': np.nan,
                'n': len(pressure_df)
            }
            continue
        
        # Calculate correlation
        valid_data = pressure_df.dropna(subset=['FingerSizeBottom', 'Log_Video_Median_Velocity'])
        corr_result = stats.pearsonr(
            valid_data['FingerSizeBottom'],
            valid_data['Log_Video_Median_Velocity']
        )
        
        correlations[pressure] = {
            'r': corr_result[0],
            'p': corr_result[1],
            'n': len(valid_data)
        }
        
        print(f"- {pressure} psi: r = {corr_result[0]:.3f}, p = {corr_result[1]:.3f}, n = {len(valid_data)}")
    
    return correlations

def plot_finger_size_log_correlations(merged_df: pd.DataFrame, correlations: Dict, output_dir: str) -> None:
    """
    Create visualizations of finger size (bottom) correlation with log-transformed velocity.
    
    Args:
        merged_df: DataFrame containing finger metrics and velocity data
        correlations: Dictionary of correlation results by pressure
        output_dir: Directory to save output plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot scatter for each pressure level with correlation
    for pressure in correlations.keys():
        pressure_df = merged_df[merged_df['Pressure'] == pressure]
        valid_data = pressure_df.dropna(subset=['FingerSizeBottom', 'Log_Video_Median_Velocity'])
        
        if len(valid_data) < 3:
            continue
            
        plt.figure(figsize=(3.5, 2.5))
        
        # Create scatter plot
        sns.scatterplot(
            x='FingerSizeBottom', 
            y='Log_Video_Median_Velocity',
            data=valid_data,
            hue='Finger',
            s=40
        )
        
        # Add regression line if we have enough data
        if len(valid_data) >= 3:
            sns.regplot(
                x='FingerSizeBottom', 
                y='Log_Video_Median_Velocity',
                data=valid_data,
                scatter=False,
                ci=None,
                line_kws={'color': 'red', 'linestyle': '--'}
            )
        
        # Add correlation info
        r_value = correlations[pressure]['r']
        p_value = correlations[pressure]['p']
        n_value = correlations[pressure]['n']
        
        p_text = f"r = {r_value:.2f}, p = {p_value:.3f}, n = {n_value}" if p_value >= 0.001 else f"r = {r_value:.2f}, p < 0.001, n = {n_value}"
        
        if source_sans:
            plt.xlabel('Finger Bottom Circumference (mm)', fontproperties=source_sans)
            plt.ylabel('Log Velocity (log mm/s)', fontproperties=source_sans)
            plt.title(f'Finger Size vs Log Velocity at {pressure} psi', fontproperties=source_sans)
            plt.text(0.05, 0.95, p_text, transform=plt.gca().transAxes, 
                   fontproperties=source_sans, fontsize=6)
        else:
            plt.xlabel('Finger Bottom Circumference (mm)')
            plt.ylabel('Log Velocity (log mm/s)')
            plt.title(f'Finger Size vs Log Velocity at {pressure} psi')
            plt.text(0.05, 0.95, p_text, transform=plt.gca().transAxes, fontsize=6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'finger_size_vs_log_velocity_{pressure}psi.png'), dpi=300)
        plt.close()
    
    # Plot overall correlation across all pressures
    all_data = merged_df.dropna(subset=['FingerSizeBottom', 'Log_Video_Median_Velocity'])
    
    if len(all_data) >= 3:
        plt.figure(figsize=(3.5, 2.5))
        
        # Create scatter plot
        sns.scatterplot(
            x='FingerSizeBottom', 
            y='Log_Video_Median_Velocity',
            data=all_data,
            hue='Pressure',
            palette='viridis',
            s=40
        )
        
        # Add regression line
        sns.regplot(
            x='FingerSizeBottom', 
            y='Log_Video_Median_Velocity',
            data=all_data,
            scatter=False,
            ci=None,
            line_kws={'color': 'red', 'linestyle': '--'}
        )
        
        # Calculate overall correlation
        corr_result = stats.pearsonr(all_data['FingerSizeBottom'], all_data['Log_Video_Median_Velocity'])
        p_text = f"r = {corr_result[0]:.2f}, p = {corr_result[1]:.3f}, n = {len(all_data)}" if corr_result[1] >= 0.001 else f"r = {corr_result[0]:.2f}, p < 0.001, n = {len(all_data)}"
        
        if source_sans:
            plt.xlabel('Finger Bottom Circumference (mm)', fontproperties=source_sans)
            plt.ylabel('Log Velocity (log mm/s)', fontproperties=source_sans)
            plt.title('Finger Size vs Log Velocity (All Pressures)', fontproperties=source_sans)
            plt.text(0.05, 0.95, p_text, transform=plt.gca().transAxes, 
                   fontproperties=source_sans, fontsize=6)
        else:
            plt.xlabel('Finger Bottom Circumference (mm)')
            plt.ylabel('Log Velocity (log mm/s)')
            plt.title('Finger Size vs Log Velocity (All Pressures)')
            plt.text(0.05, 0.95, p_text, transform=plt.gca().transAxes, fontsize=6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'finger_size_vs_log_velocity_all_pressures.png'), dpi=300)
        plt.close()

def perform_anova_analysis(merged_df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """
    Perform ANOVA/regression to compare the effects of finger size and age on velocity.
    Uses continuous variables rather than categorized groups.
    
    Args:
        merged_df: DataFrame containing finger metrics, velocity data, and demographic info
        
    Returns:
        Tuple containing dictionary of ANOVA results and summary DataFrame
    """
    # Note: This analysis uses OLS (Ordinary Least Squares) which does not explicitly 
    # account for the repeated measures structure (multiple observations per participant).
    # Results should be interpreted with caution, as standard errors might be underestimated.
    # Mixed-effects models (perform_anova_analysis_mixed) are preferred for this data structure.
    print("\nPerforming ANOVA/regression analysis for finger size vs age effects on velocity:")
    
    # Ensure we have complete data for analysis
    analysis_df = merged_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity', 'Age'])
    
    if len(analysis_df) < 10:
        print("Insufficient data for ANOVA analysis")
        return {}, pd.DataFrame()
    
    # Create results dictionary
    anova_results = {}
    
    # Model 1: Finger size only (as continuous variable)
    model_finger = ols('Video_Median_Velocity ~ FingerSizeBottom', data=analysis_df).fit()
    anova_finger = sm.stats.anova_lm(model_finger, typ=2)
    r2_finger = model_finger.rsquared
    anova_results['finger_size'] = {
        'F': anova_finger.loc['FingerSizeBottom', 'F'],
        'p': anova_finger.loc['FingerSizeBottom', 'PR(>F)'],
        'df': f"{anova_finger.loc['FingerSizeBottom', 'df']}, {anova_finger.loc['Residual', 'df']}",
        'r2': r2_finger,
        'coef': model_finger.params['FingerSizeBottom'],
        'p_value': model_finger.pvalues['FingerSizeBottom']
    }
    
    # Model 2: Age only (as continuous variable)
    model_age = ols('Video_Median_Velocity ~ Age', data=analysis_df).fit()
    anova_age = sm.stats.anova_lm(model_age, typ=2)
    r2_age = model_age.rsquared
    anova_results['age'] = {
        'F': anova_age.loc['Age', 'F'],
        'p': anova_age.loc['Age', 'PR(>F)'],
        'df': f"{anova_age.loc['Age', 'df']}, {anova_age.loc['Residual', 'df']}",
        'r2': r2_age,
        'coef': model_age.params['Age'],
        'p_value': model_age.pvalues['Age']
    }
    
    # Model 3: Both variables (additive model)
    model_additive = ols('Video_Median_Velocity ~ FingerSizeBottom + Age', data=analysis_df).fit()
    anova_additive = sm.stats.anova_lm(model_additive, typ=2)
    r2_additive = model_additive.rsquared
    
    anova_results['finger_size_in_model'] = {
        'F': anova_additive.loc['FingerSizeBottom', 'F'],
        'p': anova_additive.loc['FingerSizeBottom', 'PR(>F)'],
        'df': f"{anova_additive.loc['FingerSizeBottom', 'df']}, {anova_additive.loc['Residual', 'df']}",
        'coef': model_additive.params['FingerSizeBottom'],
        'p_value': model_additive.pvalues['FingerSizeBottom']
    }
    
    anova_results['age_in_model'] = {
        'F': anova_additive.loc['Age', 'F'],
        'p': anova_additive.loc['Age', 'PR(>F)'],
        'df': f"{anova_additive.loc['Age', 'df']}, {anova_additive.loc['Residual', 'df']}",
        'coef': model_additive.params['Age'],
        'p_value': model_additive.pvalues['Age']
    }
    
    # Model 4: Interaction model
    model_interaction = ols('Video_Median_Velocity ~ FingerSizeBottom * Age', data=analysis_df).fit()
    anova_interaction = sm.stats.anova_lm(model_interaction, typ=2)
    r2_interaction = model_interaction.rsquared
    
    anova_results['interaction'] = {
        'F': anova_interaction.loc['FingerSizeBottom:Age', 'F'] if 'FingerSizeBottom:Age' in anova_interaction.index else np.nan,
        'p': anova_interaction.loc['FingerSizeBottom:Age', 'PR(>F)'] if 'FingerSizeBottom:Age' in anova_interaction.index else np.nan,
        'df': f"{anova_interaction.loc['FingerSizeBottom:Age', 'df']}, {anova_interaction.loc['Residual', 'df']}" if 'FingerSizeBottom:Age' in anova_interaction.index else "N/A",
        'coef': model_interaction.params['FingerSizeBottom:Age'] if 'FingerSizeBottom:Age' in model_interaction.params.index else np.nan,
        'p_value': model_interaction.pvalues['FingerSizeBottom:Age'] if 'FingerSizeBottom:Age' in model_interaction.pvalues.index else np.nan
    }
    
    # Store model R² values
    anova_results['model_comparison'] = {
        'finger_size_only_r2': r2_finger,
        'age_only_r2': r2_age,
        'additive_r2': r2_additive,
        'interaction_r2': r2_interaction
    }
    
    # Print results
    print("\nANOVA Results:")
    print(f"Finger Size (single predictor): F({anova_results['finger_size']['df']}) = {anova_results['finger_size']['F']:.2f}, p = {anova_results['finger_size']['p']:.4f}, R² = {r2_finger:.3f}")
    print(f"Age (single predictor): F({anova_results['age']['df']}) = {anova_results['age']['F']:.2f}, p = {anova_results['age']['p']:.4f}, R² = {r2_age:.3f}")
    print(f"Finger Size (in additive model): F({anova_results['finger_size_in_model']['df']}) = {anova_results['finger_size_in_model']['F']:.2f}, p = {anova_results['finger_size_in_model']['p']:.4f}")
    print(f"Age (in additive model): F({anova_results['age_in_model']['df']}) = {anova_results['age_in_model']['F']:.2f}, p = {anova_results['age_in_model']['p']:.4f}")
    print(f"Additive model R²: {r2_additive:.3f}")
    
    if 'FingerSizeBottom:Age' in anova_interaction.index:
        print(f"Interaction: F({anova_results['interaction']['df']}) = {anova_results['interaction']['F']:.2f}, p = {anova_results['interaction']['p']:.4f}")
        print(f"Interaction model R²: {r2_interaction:.3f}")
    
    # Create summary DataFrame for easier reporting
    summary_data = []
    
    summary_data.append({
        'Factor': 'Finger Size (alone)',
        'F': anova_results['finger_size']['F'],
        'p': anova_results['finger_size']['p'],
        'df': anova_results['finger_size']['df'],
        'R²': r2_finger,
        'Coefficient': anova_results['finger_size']['coef']
    })
    
    summary_data.append({
        'Factor': 'Age (alone)',
        'F': anova_results['age']['F'],
        'p': anova_results['age']['p'],
        'df': anova_results['age']['df'],
        'R²': r2_age,
        'Coefficient': anova_results['age']['coef']
    })
    
    summary_data.append({
        'Factor': 'Finger Size (additive model)',
        'F': anova_results['finger_size_in_model']['F'],
        'p': anova_results['finger_size_in_model']['p'],
        'df': anova_results['finger_size_in_model']['df'],
        'R²': r2_additive,
        'Coefficient': anova_results['finger_size_in_model']['coef']
    })
    
    summary_data.append({
        'Factor': 'Age (additive model)',
        'F': anova_results['age_in_model']['F'],
        'p': anova_results['age_in_model']['p'],
        'df': anova_results['age_in_model']['df'],
        'R²': r2_additive,
        'Coefficient': anova_results['age_in_model']['coef']
    })
    
    if 'FingerSizeBottom:Age' in anova_interaction.index:
        summary_data.append({
            'Factor': 'Finger Size × Age Interaction',
            'F': anova_results['interaction']['F'],
            'p': anova_results['interaction']['p'],
            'df': anova_results['interaction']['df'],
            'R²': r2_interaction,
            'Coefficient': anova_results['interaction']['coef']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    return anova_results, summary_df

def plot_anova_results(merged_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create visualizations of regression results comparing finger size and age effects.
    Uses continuous variables rather than categorized groups.
    
    Args:
        merged_df: DataFrame containing finger metrics, velocity data, and demographic info
        output_dir: Directory to save output plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    analysis_df = merged_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity', 'Age'])
    
    if len(analysis_df) < 10:
        print("Insufficient data for regression visualization")
        return
    
    # Plot 1: Scatter plot with regression line for finger size vs velocity
    plt.figure(figsize=(3.5, 2.5))
    sns.regplot(x='FingerSizeBottom', y='Video_Median_Velocity', data=analysis_df,
               scatter_kws={'alpha': 0.6, 's': 30}, line_kws={'color': 'red'})
    
    # Calculate correlation for the annotation
    corr = analysis_df['FingerSizeBottom'].corr(analysis_df['Video_Median_Velocity'])
    p_value = stats.pearsonr(analysis_df['FingerSizeBottom'], analysis_df['Video_Median_Velocity'])[1]
    corr_text = f"r = {corr:.3f}, p = {p_value:.3f}" if p_value >= 0.001 else f"r = {corr:.3f}, p < 0.001"
    
    if source_sans:
        plt.xlabel('Finger Bottom Circumference (mm)', fontproperties=source_sans)
        plt.ylabel('Velocity (mm/s)', fontproperties=source_sans)
        plt.title('Finger Size Effect on Velocity', fontproperties=source_sans)
        plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, 
               fontproperties=source_sans, fontsize=6)
    else:
        plt.xlabel('Finger Bottom Circumference (mm)')
        plt.ylabel('Velocity (mm/s)')
        plt.title('Finger Size Effect on Velocity')
        plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'finger_size_regression.png'), dpi=300)
    plt.close()
    
    # Plot 2: Scatter plot with regression line for age vs velocity
    plt.figure(figsize=(3.5, 2.5))
    sns.regplot(x='Age', y='Video_Median_Velocity', data=analysis_df,
               scatter_kws={'alpha': 0.6, 's': 30}, line_kws={'color': 'red'})
    
    # Calculate correlation for the annotation
    corr = analysis_df['Age'].corr(analysis_df['Video_Median_Velocity'])
    p_value = stats.pearsonr(analysis_df['Age'], analysis_df['Video_Median_Velocity'])[1]
    corr_text = f"r = {corr:.3f}, p = {p_value:.3f}" if p_value >= 0.001 else f"r = {corr:.3f}, p < 0.001"
    
    if source_sans:
        plt.xlabel('Age (years)', fontproperties=source_sans)
        plt.ylabel('Velocity (mm/s)', fontproperties=source_sans)
        plt.title('Age Effect on Velocity', fontproperties=source_sans)
        plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, 
               fontproperties=source_sans, fontsize=6)
    else:
        plt.xlabel('Age (years)')
        plt.ylabel('Velocity (mm/s)')
        plt.title('Age Effect on Velocity')
        plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_regression.png'), dpi=300)
    plt.close()
    
    # Plot 3: 3D scatter plot with regression plane (if we have matplotlib 3D support)
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        # Fit the model
        model = ols('Video_Median_Velocity ~ FingerSizeBottom + Age', data=analysis_df).fit()
        
        # Create meshgrid for prediction
        finger_min, finger_max = analysis_df['FingerSizeBottom'].min(), analysis_df['FingerSizeBottom'].max()
        age_min, age_max = analysis_df['Age'].min(), analysis_df['Age'].max()
        finger_range = np.linspace(finger_min, finger_max, 20)
        age_range = np.linspace(age_min, age_max, 20)
        finger_mesh, age_mesh = np.meshgrid(finger_range, age_range)
        
        # Predict velocity for the meshgrid
        predict_df = pd.DataFrame({
            'FingerSizeBottom': finger_mesh.flatten(),
            'Age': age_mesh.flatten()
        })
        predicted = model.predict(predict_df).values.reshape(finger_mesh.shape)
        
        # Create the 3D plot
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surface = ax.plot_surface(finger_mesh, age_mesh, predicted, 
                                 alpha=0.5, cmap='viridis', edgecolor='none')
        
        # Plot the data points
        scatter = ax.scatter(analysis_df['FingerSizeBottom'], analysis_df['Age'], 
                           analysis_df['Video_Median_Velocity'], 
                           c='red', s=20, alpha=0.7)
        
        # Set labels
        if source_sans:
            ax.set_xlabel('Finger Size (mm)', fontproperties=source_sans)
            ax.set_ylabel('Age (years)', fontproperties=source_sans)
            ax.set_zlabel('Velocity (mm/s)', fontproperties=source_sans)
            plt.title('Finger Size and Age Effects on Velocity', fontproperties=source_sans)
        else:
            ax.set_xlabel('Finger Size (mm)')
            ax.set_ylabel('Age (years)')
            ax.set_zlabel('Velocity (mm/s)')
            plt.title('Finger Size and Age Effects on Velocity')
        
        # Add colorbar
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label='Predicted Velocity (mm/s)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3d_regression_surface.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not create 3D visualization: {e}")
    
    # Plot 4: Partial residual plots to show effects controlling for other variables
    try:
        fig = plt.figure(figsize=(7, 3))
        
        # Create partial residual plot for finger size
        ax1 = fig.add_subplot(121)
        sm.graphics.plot_partregress(endog='Video_Median_Velocity', 
                                    exog_i='FingerSizeBottom', 
                                    exog_others=['Age'], 
                                    data=analysis_df, 
                                    ax=ax1,
                                    obs_labels=False)
        
        if source_sans:
            ax1.set_xlabel('Finger Size (partial)', fontproperties=source_sans)
            ax1.set_ylabel('Velocity (partial)', fontproperties=source_sans)
            ax1.set_title('Finger Size Effect\n(controlling for Age)', fontproperties=source_sans)
        else:
            ax1.set_xlabel('Finger Size (partial)')
            ax1.set_ylabel('Velocity (partial)')
            ax1.set_title('Finger Size Effect\n(controlling for Age)')
        
        # Create partial residual plot for age
        ax2 = fig.add_subplot(122)
        sm.graphics.plot_partregress(endog='Video_Median_Velocity', 
                                    exog_i='Age', 
                                    exog_others=['FingerSizeBottom'], 
                                    data=analysis_df, 
                                    ax=ax2,
                                    obs_labels=False)
        
        if source_sans:
            ax2.set_xlabel('Age (partial)', fontproperties=source_sans)
            ax2.set_ylabel('Velocity (partial)', fontproperties=source_sans)
            ax2.set_title('Age Effect\n(controlling for Finger Size)', fontproperties=source_sans)
        else:
            ax2.set_xlabel('Age (partial)')
            ax2.set_ylabel('Velocity (partial)')
            ax2.set_title('Age Effect\n(controlling for Finger Size)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'partial_regression_plots.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not create partial regression plots: {e}")

def analyze_finger_size_correlation_mixed(merged_df: pd.DataFrame) -> Dict:
    """
    Analyze correlations between finger size (bottom circumference) and velocity
    using mixed-effects models to account for repeated measures from the same participant.
    
    Args:
        merged_df: DataFrame containing both finger metrics and velocity data
        
    Returns:
        Dictionary of correlation results by pressure level
    """
    # Group by pressure to analyze correlation at each pressure level
    pressure_levels = merged_df['Pressure'].unique()
    mixed_model_results = {}
    
    print("\nCalculating correlations between finger size (bottom) and velocity using mixed-effects models:")
    
    for pressure in pressure_levels:
        pressure_df = merged_df[merged_df['Pressure'] == pressure]
        
        # Ensure we have enough data for meaningful analysis
        if len(pressure_df) < 10 or pressure_df['FingerSizeBottom'].isna().all() or pressure_df['Participant'].nunique() < 3:
            print(f"- Insufficient data for pressure {pressure} psi")
            mixed_model_results[pressure] = {
                'coef': np.nan,
                'p': np.nan,
                'n_obs': len(pressure_df),
                'n_participants': pressure_df['Participant'].nunique()
            }
            continue
        
        # Drop any rows with missing values
        valid_data = pressure_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity'])
        
        # Fit mixed-effects model with participant as random effect
        try:
            # Formula: velocity ~ finger_size + (1|participant)
            model = mixedlm("Video_Median_Velocity ~ FingerSizeBottom", 
                           valid_data, 
                           groups=valid_data["Participant"])
            result = model.fit()
            
            # Extract coefficient and p-value for finger size
            coef = result.params["FingerSizeBottom"]
            p_value = result.pvalues["FingerSizeBottom"]
            
            mixed_model_results[pressure] = {
                'coef': coef,
                'p': p_value,
                'n_obs': len(valid_data),
                'n_participants': valid_data['Participant'].nunique(),
                'result': result
            }
            
            print(f"- {pressure} psi: coef = {coef:.3f}, p = {p_value:.3f}, observations = {len(valid_data)}, participants = {valid_data['Participant'].nunique()}")
            
        except Exception as e:
            print(f"- Error analyzing pressure {pressure} psi: {e}")
            mixed_model_results[pressure] = {
                'coef': np.nan,
                'p': np.nan,
                'n_obs': len(valid_data),
                'n_participants': valid_data['Participant'].nunique(),
                'error': str(e)
            }
    
    # Also run a mixed model on all data together
    try:
        all_valid_data = merged_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity'])
        
        # Formula: velocity ~ finger_size + pressure + (1|participant)
        overall_model = mixedlm("Video_Median_Velocity ~ FingerSizeBottom + C(Pressure)", 
                               all_valid_data, 
                               groups=all_valid_data["Participant"])
        overall_result = overall_model.fit()
        
        # Extract coefficient and p-value for finger size
        overall_coef = overall_result.params["FingerSizeBottom"]
        overall_p = overall_result.pvalues["FingerSizeBottom"]
        
        mixed_model_results['overall'] = {
            'coef': overall_coef,
            'p': overall_p,
            'n_obs': len(all_valid_data),
            'n_participants': all_valid_data['Participant'].nunique(),
            'result': overall_result
        }
        
        print(f"- Overall (all pressures): coef = {overall_coef:.3f}, p = {overall_p:.3f}, observations = {len(all_valid_data)}, participants = {all_valid_data['Participant'].nunique()}")
        
    except Exception as e:
        print(f"- Error analyzing overall data: {e}")
        mixed_model_results['overall'] = {
            'coef': np.nan,
            'p': np.nan,
            'n_obs': len(merged_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity'])),
            'n_participants': merged_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity'])['Participant'].nunique(),
            'error': str(e)
        }
    
    return mixed_model_results

def perform_anova_analysis_mixed(merged_df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """
    Perform mixed-effects model analysis to compare the effects of finger size and age on velocity
    while accounting for repeated measures from the same participant.
    
    Args:
        merged_df: DataFrame containing finger metrics, velocity data, and demographic info
        
    Returns:
        Tuple containing dictionary of mixed model results and summary DataFrame
    """
    print("\nPerforming mixed-effects analysis for finger size vs age effects on velocity:")
    
    # Ensure we have complete data for analysis
    analysis_df = merged_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity', 'Age'])
    
    if len(analysis_df) < 10 or analysis_df['Participant'].nunique() < 3:
        print("Insufficient data for mixed-effects analysis")
        return {}, pd.DataFrame()
    
    # Create results dictionary
    mixed_results = {}
    
    # Model 1: Finger size only (with random effect for participant)
    try:
        model_finger = mixedlm('Video_Median_Velocity ~ FingerSizeBottom', 
                              data=analysis_df, 
                              groups=analysis_df["Participant"])
        result_finger = model_finger.fit()
        
        mixed_results['finger_size'] = {
            'coef': result_finger.params['FingerSizeBottom'],
            'p': result_finger.pvalues['FingerSizeBottom'],
            'aic': result_finger.aic,
            'bic': result_finger.bic,
            'n_obs': len(analysis_df),
            'n_participants': analysis_df['Participant'].nunique(),
            'result': result_finger
        }
        
        print(f"Finger Size only: coef = {mixed_results['finger_size']['coef']:.3f}, p = {mixed_results['finger_size']['p']:.4f}")
        
    except Exception as e:
        print(f"Error in finger size model: {e}")
        mixed_results['finger_size'] = {'error': str(e)}
    
    # Model 2: Age only (with random effect for participant)
    try:
        model_age = mixedlm('Video_Median_Velocity ~ Age', 
                           data=analysis_df, 
                           groups=analysis_df["Participant"])
        result_age = model_age.fit()
        
        mixed_results['age'] = {
            'coef': result_age.params['Age'],
            'p': result_age.pvalues['Age'],
            'aic': result_age.aic,
            'bic': result_age.bic,
            'n_obs': len(analysis_df),
            'n_participants': analysis_df['Participant'].nunique(),
            'result': result_age
        }
        
        print(f"Age only: coef = {mixed_results['age']['coef']:.3f}, p = {mixed_results['age']['p']:.4f}")
        
    except Exception as e:
        print(f"Error in age model: {e}")
        mixed_results['age'] = {'error': str(e)}
    
    # Model 3: Both variables (additive model with random effect for participant)
    try:
        model_additive = mixedlm('Video_Median_Velocity ~ FingerSizeBottom + Age', 
                                data=analysis_df, 
                                groups=analysis_df["Participant"])
        result_additive = model_additive.fit()
        
        mixed_results['finger_size_in_model'] = {
            'coef': result_additive.params['FingerSizeBottom'],
            'p': result_additive.pvalues['FingerSizeBottom'],
            'aic': result_additive.aic,
            'bic': result_additive.bic,
            'n_obs': len(analysis_df),
            'n_participants': analysis_df['Participant'].nunique()
        }
        
        mixed_results['age_in_model'] = {
            'coef': result_additive.params['Age'],
            'p': result_additive.pvalues['Age'],
            'aic': result_additive.aic,
            'bic': result_additive.bic,
            'n_obs': len(analysis_df),
            'n_participants': analysis_df['Participant'].nunique()
        }
        
        mixed_results['additive_model'] = {
            'result': result_additive,
            'aic': result_additive.aic,
            'bic': result_additive.bic
        }
        
        print(f"Additive model - Finger Size: coef = {mixed_results['finger_size_in_model']['coef']:.3f}, p = {mixed_results['finger_size_in_model']['p']:.4f}")
        print(f"Additive model - Age: coef = {mixed_results['age_in_model']['coef']:.3f}, p = {mixed_results['age_in_model']['p']:.4f}")
        
    except Exception as e:
        print(f"Error in additive model: {e}")
        mixed_results['additive_model'] = {'error': str(e)}
    
    # Create summary DataFrame for easier reporting
    summary_data = []
    
    if 'finger_size' in mixed_results and 'error' not in mixed_results['finger_size']:
        summary_data.append({
            'Factor': 'Finger Size (alone)',
            'Coefficient': mixed_results['finger_size']['coef'],
            'p': mixed_results['finger_size']['p'],
            'AIC': mixed_results['finger_size']['aic'],
            'BIC': mixed_results['finger_size']['bic'],
            'Observations': mixed_results['finger_size']['n_obs'],
            'Participants': mixed_results['finger_size']['n_participants']
        })
    
    if 'age' in mixed_results and 'error' not in mixed_results['age']:
        summary_data.append({
            'Factor': 'Age (alone)',
            'Coefficient': mixed_results['age']['coef'],
            'p': mixed_results['age']['p'],
            'AIC': mixed_results['age']['aic'],
            'BIC': mixed_results['age']['bic'],
            'Observations': mixed_results['age']['n_obs'],
            'Participants': mixed_results['age']['n_participants']
        })
    
    if 'finger_size_in_model' in mixed_results:
        summary_data.append({
            'Factor': 'Finger Size (additive model)',
            'Coefficient': mixed_results['finger_size_in_model']['coef'],
            'p': mixed_results['finger_size_in_model']['p'],
            'AIC': mixed_results['finger_size_in_model']['aic'],
            'BIC': mixed_results['finger_size_in_model']['bic'],
            'Observations': mixed_results['finger_size_in_model']['n_obs'],
            'Participants': mixed_results['finger_size_in_model']['n_participants']
        })
    
    if 'age_in_model' in mixed_results:
        summary_data.append({
            'Factor': 'Age (additive model)',
            'Coefficient': mixed_results['age_in_model']['coef'],
            'p': mixed_results['age_in_model']['p'],
            'AIC': mixed_results['age_in_model']['aic'],
            'BIC': mixed_results['age_in_model']['bic'],
            'Observations': mixed_results['age_in_model']['n_obs'],
            'Participants': mixed_results['age_in_model']['n_participants']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    return mixed_results, summary_df

def analyze_finger_size_pressure_interaction(merged_df: pd.DataFrame) -> Dict:
    """
    Create mixed-effects models that analyze how finger size, pressure, and age
    interact to affect capillary velocity, using multiple model specifications.
    
    Args:
        merged_df: DataFrame containing finger metrics, velocity data, and demographic info
        
    Returns:
        Dictionary containing model results for different specifications
    """
    print("\nAnalyzing interaction between finger size and pressure using mixed-effects models:")
    
    # Ensure we have complete data for analysis
    analysis_df = merged_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity', 'Pressure'])
    
    if len(analysis_df) < 10 or analysis_df['Participant'].nunique() < 3:
        print("Insufficient data for finger size × pressure interaction analysis")
        return {'error': 'Insufficient data'}
    
    # Create results dictionary
    models = {}
    
    # Treat Pressure as numeric for model fitting, particularly for random slopes.
    # If distinct levels were few and non-linear effects expected, C(Pressure) could be used 
    # in formulas, but re_formula="~C(Pressure)" is not directly supported for random slopes.
    
    # Basic data summary
    print(f"\nData summary:")
    print(f"Number of observations: {len(analysis_df)}")
    print(f"Number of unique participants: {analysis_df['Participant'].nunique()}")
    print(f"Number of fingers: {analysis_df['Finger'].nunique()}")
    print(f"Finger types: {', '.join(analysis_df['Finger'].unique())}")
    print(f"Pressure levels: {', '.join(map(str, analysis_df['Pressure'].unique()))}")
    
    # Model 1: Basic mixed model with finger size and pressure as main effects (Pressure numeric)
    try:
        print("\nModel 1: Basic fixed effects (finger size + pressure)")
        formula = "Video_Median_Velocity ~ FingerSizeBottom + Pressure"
        model = sm.formula.mixedlm(formula, analysis_df, groups=analysis_df["Participant"])
        models['basic'] = model.fit()
        print(models['basic'].summary())
    except Exception as e:
        print(f"Error in basic model: {e}")
        models['basic'] = {'error': str(e)}
    
    # Model 2: Add interaction between finger size and pressure (Pressure numeric)
    try:
        print("\nModel 2: Finger size × Pressure interaction")
        formula = "Video_Median_Velocity ~ FingerSizeBottom * Pressure"
        model = sm.formula.mixedlm(formula, analysis_df, groups=analysis_df["Participant"])
        models['interaction'] = model.fit()
        print(models['interaction'].summary())
    except Exception as e:
        print(f"Error in interaction model: {e}")
        models['interaction'] = {'error': str(e)}
    
    # Model 3: Random slopes for pressure (Pressure numeric)
    try:
        print("\nModel 3: Random slopes for Pressure")
        formula = "Video_Median_Velocity ~ FingerSizeBottom + Pressure"
        model = sm.formula.mixedlm(formula, analysis_df, groups=analysis_df["Participant"], 
                                  re_formula="~Pressure")
        models['random_slopes'] = model.fit()
        print(models['random_slopes'].summary())
    except Exception as e:
        print(f"Error in random slopes model: {e}")
        models['random_slopes'] = {'error': str(e)}
    
    # Model 4: Full model with age if available (Pressure numeric)
    if 'Age' in analysis_df.columns and not analysis_df['Age'].isna().all():
        try:
            print("\nModel 4: Full model with Age")
            formula = "Video_Median_Velocity ~ FingerSizeBottom * Pressure + Age"
            model = sm.formula.mixedlm(formula, analysis_df, groups=analysis_df["Participant"])
            models['full'] = model.fit()
            print(models['full'].summary())
        except Exception as e:
            print(f"Error in full model: {e}")
            models['full'] = {'error': str(e)}
    
    # Model 5: Log-transformed velocity (Pressure numeric)
    try:
        print("\nModel 5: Log-transformed velocity")
        # Use log(y+1) 
        analysis_df['Log_Video_Median_Velocity'] = np.log(analysis_df['Video_Median_Velocity'] + 1)
        formula = "Log_Video_Median_Velocity ~ FingerSizeBottom * Pressure"
        model = sm.formula.mixedlm(formula, analysis_df, groups=analysis_df["Participant"])
        models['log_velocity'] = model.fit()
        print(models['log_velocity'].summary())
    except Exception as e:
        print(f"Error in log velocity model: {e}")
        models['log_velocity'] = {'error': str(e)}
    
    # Model comparison if multiple models fitted successfully
    valid_models = [name for name, result in models.items() 
                   if isinstance(result, sm.regression.linear_model.RegressionResultsWrapper)]
    
    if len(valid_models) > 1:
        print("\nModel comparison (AIC and BIC):")
        comparison_data = []
        for name in valid_models:
            comparison_data.append({
                'Model': name,
                'AIC': models[name].aic,
                'BIC': models[name].bic,
                'Log-Likelihood': models[name].llf
            })
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df)
    
    return models

def plot_mixed_model_diagnostics(models: Dict, analysis_df: pd.DataFrame, output_dir: str) -> None:
    """
    Creates diagnostic plots for the mixed effects models.
    
    Args:
        models: Dictionary of fitted model results
        analysis_df: DataFrame containing the original data
        output_dir: Directory to save output plots
    """
    print("\nCreating diagnostic plots for mixed models...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each valid model
    for model_name, result in models.items():
        # Skip if not a valid model result
        if not isinstance(result, sm.regression.linear_model.RegressionResultsWrapper):
            continue
            
        print(f"  Creating diagnostics for {model_name} model...")
        
        # Get residuals and fitted values
        residuals = result.resid
        fitted = result.fittedvalues
        
        # 1. Basic diagnostic plots (2×2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Diagnostic Plots - {model_name} Model')
        
        # Residuals vs fitted
        sns.scatterplot(x=fitted, y=residuals, ax=axes[0, 0])
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        
        # QQ plot
        sm.graphics.qqplot(residuals, line='45', fit=True, ax=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot')
        
        # Histogram of residuals
        sns.histplot(residuals, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].set_xlabel('Residual')
        
        # Observed vs predicted
        if model_name == 'log_velocity':
            observed = np.exp(analysis_df['Log_Video_Median_Velocity']) - 1
            predicted = np.exp(fitted) - 1
            axes[1, 1].set_title('Observed vs Predicted (Back-transformed)')
        else:
            observed = analysis_df['Video_Median_Velocity']
            predicted = fitted
            axes[1, 1].set_title('Observed vs Predicted')
            
        sns.scatterplot(x=observed, y=predicted, ax=axes[1, 1])
        
        # Add a 45-degree reference line
        min_val = min(observed.min(), predicted.min())
        max_val = max(observed.max(), predicted.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1, 1].set_xlabel('Observed')
        axes[1, 1].set_ylabel('Predicted')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_diagnostics.png'), dpi=300)
        plt.close()
        
        # 2. Random effects plots for random slopes model
        if model_name == 'random_slopes' and hasattr(result, 'random_effects'):
            random_effects = result.random_effects
            
            # Convert random effects to DataFrame
            re_df = pd.DataFrame.from_dict(random_effects, orient='index')
            if re_df.shape[1] >= 2:  # If we have both intercept and slope
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
                plt.savefig(os.path.join(output_dir, f'{model_name}_random_effects.png'), dpi=300)
                plt.close()

def plot_finger_size_pressure_interaction(models: Dict, merged_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create visualizations of the interaction between finger size and pressure.
    
    Args:
        models: Dictionary of fitted model results
        merged_df: DataFrame containing finger metrics and velocity data
        output_dir: Directory to save output plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have valid model results
    if 'interaction' not in models or not isinstance(models['interaction'], 
                                                    sm.regression.linear_model.RegressionResultsWrapper):
        print("Cannot create interaction plots: interaction model results not available")
        return
    
    print("\nCreating interaction plots...")
    
    # Prepare data for plotting
    analysis_df = merged_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity', 'Pressure'])
    # Keep Pressure numeric for consistency with model fitting
    # analysis_df['Pressure'] = analysis_df['Pressure'].astype('category') 
    
    # Get the interaction model
    interaction_model = models['interaction']
    
    # Plot 1: Scatterplot with regression lines for each pressure level
    # ... (Plot 1 code remains largely the same, but interprets numeric Pressure)
    plt.figure(figsize=(10, 6))
    
    # Create scatterplot colored by pressure (treating Pressure as continuous for color mapping)
    scatter = sns.scatterplot(x='FingerSizeBottom', y='Video_Median_Velocity', 
                           hue='Pressure', palette='viridis', 
                           data=analysis_df, alpha=0.7, s=50, legend='brief')
    
    # Add regression lines for specific pressure levels (e.g., min, mean, max)
    min_pressure = analysis_df['Pressure'].min()
    mean_pressure = analysis_df['Pressure'].mean()
    max_pressure = analysis_df['Pressure'].max()
    example_pressures = [min_pressure, mean_pressure, max_pressure]
    
    finger_range = np.linspace(analysis_df['FingerSizeBottom'].min(), analysis_df['FingerSizeBottom'].max(), 100)
    
    for pressure in example_pressures:
        pred_data = pd.DataFrame({
            'FingerSizeBottom': finger_range,
            'Pressure': pressure
        })
        predicted = interaction_model.predict(pred_data)
        plt.plot(finger_range, predicted, linestyle='--', 
                 label=f'Predicted at {pressure:.2f} PSI')
    
    if source_sans:
        plt.xlabel('Finger Bottom Circumference (mm)', fontproperties=source_sans)
        plt.ylabel('Velocity (mm/s)', fontproperties=source_sans)
        plt.title('Finger Size vs Velocity by Pressure', fontproperties=source_sans)
        plt.legend(prop=source_sans)
    else:
        plt.xlabel('Finger Bottom Circumference (mm)')
        plt.ylabel('Velocity (mm/s)')
        plt.title('Finger Size vs Velocity by Pressure')
        plt.legend()
    
    # Add colorbar
    norm = plt.Normalize(analysis_df['Pressure'].min(), analysis_df['Pressure'].max())
    sm_ = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm_.set_array([])
    scatter.figure.colorbar(sm_, label='Pressure (PSI)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'finger_size_velocity_by_pressure.png'), dpi=300)
    plt.close()
    
    # Plot 2: Predicted values from the mixed model interaction
    plt.figure(figsize=(10, 6))
    
    # Create finger size range for predictions
    finger_min = analysis_df['FingerSizeBottom'].min()
    finger_max = analysis_df['FingerSizeBottom'].max()
    finger_range = np.linspace(finger_min, finger_max, 100)
    
    # Plot predicted lines for different pressures
    pressures_to_plot = np.linspace(analysis_df['Pressure'].min(), analysis_df['Pressure'].max(), 5) # Plot 5 lines
    colors = sns.color_palette('viridis_r', n_colors=len(pressures_to_plot))

    for i, pressure in enumerate(pressures_to_plot):
        pred_data = pd.DataFrame({
            'FingerSizeBottom': finger_range,
            'Pressure': pressure
        })
        
        # Use the model's predict method
        predictions = interaction_model.predict(pred_data)
        
        plt.plot(finger_range, predictions, linewidth=2, 
                 color=colors[i], label=f'Pressure = {pressure:.2f} PSI')
    
    if source_sans:
        plt.xlabel('Finger Bottom Circumference (mm)', fontproperties=source_sans)
        plt.ylabel('Predicted Velocity (mm/s)', fontproperties=source_sans)
        plt.title('Mixed Model Predictions: Finger Size × Pressure Interaction', fontproperties=source_sans)
        plt.legend(prop=source_sans)
    else:
        plt.xlabel('Finger Bottom Circumference (mm)')
        plt.ylabel('Predicted Velocity (mm/s)')
        plt.title('Mixed Model Predictions: Finger Size × Pressure Interaction')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mixed_model_predictions.png'), dpi=300)
    plt.close()
    
    # Plot 3: Visualize the predicted slope of FingerSizeBottom vs Pressure
    try:
        plt.figure(figsize=(8, 6))
        
        # Extract relevant coefficients from the interaction model
        params = interaction_model.params
        beta_fs = params.get('FingerSizeBottom', 0)
        beta_fs_p = params.get('FingerSizeBottom:Pressure', 0)
        
        # Define the pressure range for plotting the slope
        pressure_range_plot = np.linspace(analysis_df['Pressure'].min(), analysis_df['Pressure'].max(), 100)
        
        # Calculate the predicted slope of FingerSizeBottom at each pressure
        predicted_slope = beta_fs + beta_fs_p * pressure_range_plot
        
        # Plot the predicted slope
        plt.plot(pressure_range_plot, predicted_slope, linewidth=2)
        
        # Add zero line for reference
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        
        # Add labels and title
        if source_sans:
            plt.xlabel('Pressure (PSI)', fontproperties=source_sans)
            plt.ylabel('Predicted Slope of Finger Size Effect', fontproperties=source_sans)
            plt.title('How Finger Size Effect Changes with Pressure', fontproperties=source_sans)
        else:
            plt.xlabel('Pressure (PSI)')
            plt.ylabel('Predicted Slope of Finger Size Effect')
            plt.title('How Finger Size Effect Changes with Pressure')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'finger_size_slope_vs_pressure.png'), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating slope visualization plot: {e}")

def plot_random_slopes_visualization(models: Dict, merged_df: pd.DataFrame, output_dir: str) -> None:
    """
    Creates visualizations for the random slopes model to show individual participant variations.
    
    Args:
        models: Dictionary of fitted model results
        merged_df: DataFrame containing finger metrics and velocity data
        output_dir: Directory to save output plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have valid random slopes model
    if 'random_slopes' not in models or not isinstance(models['random_slopes'], 
                                                      sm.regression.linear_model.RegressionResultsWrapper):
        print("Cannot create random slopes visualization: model results not available")
        return
    
    print("\nCreating random slopes visualization...")
    
    # Get the random slopes model
    random_slopes_model = models['random_slopes']
    
    # Check if we have random effects
    if not hasattr(random_slopes_model, 'random_effects'):
        print("Random effects not available in the model")
        return
    
    random_effects = random_slopes_model.random_effects
    
    # Prepare data for plotting
    analysis_df = merged_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity', 'Pressure'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot individual participant pressure slopes
    pressure_range = np.linspace(analysis_df['Pressure'].min(), analysis_df['Pressure'].max(), 100)
    
    # Plot for a subset of participants to avoid overcrowding
    participant_subset = list(random_effects.keys())[:10]
    
    # Plot individual participant trajectories
    for participant in participant_subset:
        re = random_effects[participant]
        
        # Check model structure for fixed effects (numeric Pressure assumed)
        fe_intercept = random_slopes_model.fe_params.get('Intercept', 0)
        fe_pressure_slope = random_slopes_model.fe_params.get('Pressure', 0)
        
        # Make sure we have both intercept and slope in random effects
        if len(re) >= 2:
            re_intercept = re[0]
            re_pressure_slope = re[1] # Random slope for numeric Pressure
            
            # Calculate predicted values for this participant
            # Predicted = (FE_Intercept + RE_Intercept) + (FE_Slope_Pressure + RE_Slope_Pressure) * Pressure
            predicted = (fe_intercept + re_intercept + 
                        (fe_pressure_slope + re_pressure_slope) * pressure_range)
            
            ax1.plot(pressure_range, predicted, alpha=0.5) # Removed label to avoid clutter
        elif len(re) == 1: # Only random intercept
            re_intercept = re[0]
            predicted = (fe_intercept + re_intercept + 
                         fe_pressure_slope * pressure_range)
            ax1.plot(pressure_range, predicted, alpha=0.5, linestyle=':') # Dashed line for intercept-only
    
    ax1.set_xlabel('Pressure (PSI)')
    ax1.set_ylabel('Predicted Blood Flow Velocity')
    ax1.set_title('Individual Participant Pressure Responses\n(Random Slopes Model)')
    ax1.grid(True, alpha=0.3)
    
    # Plot the distribution of slopes
    slopes = [re[1] for re in random_effects.values() if len(re) >= 2]
    
    if slopes:
        sns.histplot(slopes, kde=True, ax=ax2)
        # Add fixed effect slope for reference
        ax2.axvline(x=fe_pressure_slope, color='r', linestyle='--', label='Fixed Effect Slope') 
        ax2.set_xlabel('Random Pressure Slope Component') # Clarify label
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Random Pressure Slopes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No random slopes found in model.", ha='center', va='center')
        ax2.set_title('Distribution of Random Pressure Slopes')

def main():
    """Main function for finger size (bottom) analysis."""
    print("\nRunning finger size (bottom) analysis for capillary velocity data...")
    
    # Load data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Standardize finger column names
    df['Finger'] = df['Finger'].str[1:]
    df['Finger'] = df['Finger'].str.lower()
    df['Finger'] = df['Finger'].str.capitalize()
    df['Finger'] = df['Finger'].replace('Mid', 'Middle')
    df['Finger'] = df['Finger'].replace('Index', 'Pointer')
    print(f"Fingers in dataset: {df['Finger'].unique()}")

    # Load finger stats data
    finger_stats_df = pd.read_csv(os.path.join(cap_flow_path, 'finger_stats.csv'))
    
    # Merge with velocity data
    merged_df = pd.merge(df, finger_stats_df, on='Participant', how='left')
    merged_df = merged_df.dropna(subset=['Pointer bottom'])
    
    # Map from 'Finger' string to the column name holding the bottom size
    bottom_col_map = {f: f"{f} bottom" for f in ['Pointer', 'Middle', 'Ring', 'Pinky']}

    # Define a helper function for robust lookup within the row
    def get_finger_size(row, col_map):
        finger = row['Finger']
        col_name = col_map.get(finger)
        # Check if the finger name is valid and the corresponding column exists in the row
        if col_name and col_name in row.index:
            return row[col_name]
        # Return NaN if finger name is invalid or column doesn't exist
        return np.nan

    # Apply the function to get the bottom sizes directly from merged_df columns
    merged_df['FingerSizeBottom'] = merged_df.apply(lambda row: get_finger_size(row, bottom_col_map), axis=1)
    
    # Calculate log of velocity using log(y+1)
    merged_df['Log_Video_Median_Velocity'] = np.log(merged_df['Video_Median_Velocity']+1)
    
    # Save merged data for reference
    merged_df.to_csv(os.path.join(cap_flow_path, 'finger_size_analysis_df.csv'), index=False)
    
    # Filter for control data
    # controls_df = merged_df[merged_df['SET'] == 'set01']
    controls_df = merged_df
    # Check how many participants have both finger and velocity data
    participants_with_data = controls_df.dropna(subset=['FingerSizeBottom', 'Video_Median_Velocity'])
    unique_participants = participants_with_data['Participant'].nunique()
    
    print(f"Found {unique_participants} participants with both finger size and velocity data")
    print(f"Total observations: {len(participants_with_data)}")
    print(f"participants: {controls_df['Participant'].unique()}")
    
    # Set up plotting
    setup_plotting()
    
    # Regular velocity analysis
    # Create output directory
    output_dir = os.path.join(cap_flow_path, 'results', 'FingerSizeBottom')
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze correlations between finger size and velocity
    correlations = analyze_finger_size_correlation(controls_df)
    
    # Plot and save results
    plot_finger_size_correlations(controls_df, correlations, output_dir)
    
    # Save correlation results as text
    with open(os.path.join(output_dir, 'finger_size_correlations.txt'), 'w') as f:
        f.write("Correlation between finger bottom size and velocity:\n\n")
        f.write("Pressure\tr\tp-value\tn\n")
        for pressure, stats in correlations.items():
            f.write(f"{pressure}\t{stats['r']:.3f}\t{stats['p']:.3f}\t{stats['n']}\n")
    
    print(f"\nFinger size analysis complete. Results saved to: {output_dir}")
    
    # Log velocity analysis
    # Create output directory for log analysis
    log_output_dir = os.path.join(cap_flow_path, 'results', 'FingerSizeBottom_Log')
    os.makedirs(log_output_dir, exist_ok=True)
    
    # Analyze correlations between finger size and log velocity
    log_correlations = analyze_finger_size_log_correlation(controls_df)
    
    # Plot and save log results
    plot_finger_size_log_correlations(controls_df, log_correlations, log_output_dir)
    
    # Save log correlation results as text
    with open(os.path.join(log_output_dir, 'finger_size_log_correlations.txt'), 'w') as f:
        f.write("Correlation between finger bottom size and log velocity:\n\n")
        f.write("Pressure\tr\tp-value\tn\n")
        for pressure, stats in log_correlations.items():
            f.write(f"{pressure}\t{stats['r']:.3f}\t{stats['p']:.3f}\t{stats['n']}\n")
    
    print(f"\nFinger size log velocity analysis complete. Results saved to: {log_output_dir}")
    
    # ANOVA analysis comparing finger size and age effects
    anova_output_dir = os.path.join(cap_flow_path, 'results', 'FingerSize_vs_Age_ANOVA')
    os.makedirs(anova_output_dir, exist_ok=True)
    
    # Perform ANOVA analysis
    anova_results, summary_df = perform_anova_analysis(controls_df)
    
    # Create visualizations
    plot_anova_results(controls_df, anova_output_dir)
    
    # Save ANOVA results as CSV
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(anova_output_dir, 'anova_summary.csv'), index=False)
        
        # Also save as text for easier reading
        with open(os.path.join(anova_output_dir, 'anova_summary.txt'), 'w') as f:
            f.write("ANOVA Analysis: Comparing Effects of Finger Size and Age on Velocity\n\n")
            f.write("Factor\tF\tp-value\tdf\n")
            for _, row in summary_df.iterrows():
                f.write(f"{row['Factor']}\t{row['F']:.3f}\t{row['p']:.4f}\t{row['df']}\n")
    
    print(f"\nANOVA analysis complete. Results saved to: {anova_output_dir}")
    
    # NEW: Mixed-effects model analysis to account for repeated measures
    mixed_output_dir = os.path.join(cap_flow_path, 'results', 'FingerSize_Mixed_Effects')
    os.makedirs(mixed_output_dir, exist_ok=True)
    
    # Analyze correlations using mixed-effects models
    mixed_correlations = analyze_finger_size_correlation_mixed(controls_df)
    
    # Save mixed-effects model results as text
    with open(os.path.join(mixed_output_dir, 'finger_size_mixed_effects.txt'), 'w') as f:
        f.write("Mixed-effects model analysis of finger bottom size and velocity:\n\n")
        f.write("Pressure\tCoefficient\tp-value\tObservations\tParticipants\n")
        for pressure, stats in mixed_correlations.items():
            f.write(f"{pressure}\t{stats['coef']:.3f}\t{stats['p']:.3f}\t{stats['n_obs']}\t{stats['n_participants']}\n")
            
            # If we have a full result object, write the summary
            if 'result' in stats:
                f.write("\nModel summary for pressure {pressure}:\n")
                f.write(str(stats['result'].summary()) + "\n\n")
    
    # Perform mixed-effects analysis comparing finger size and age
    mixed_results, mixed_summary_df = perform_anova_analysis_mixed(controls_df)
    
    # Save mixed-effects results as CSV
    if not mixed_summary_df.empty:
        mixed_summary_df.to_csv(os.path.join(mixed_output_dir, 'mixed_effects_summary.csv'), index=False)
        
        # Also save as text for easier reading
        with open(os.path.join(mixed_output_dir, 'mixed_effects_summary.txt'), 'w') as f:
            f.write("Mixed-Effects Model Analysis: Comparing Effects of Finger Size and Age on Velocity\n\n")
            f.write("Factor\tCoefficient\tp-value\tAIC\tBIC\tObservations\tParticipants\n")
            for _, row in mixed_summary_df.iterrows():
                f.write(f"{row['Factor']}\t{row['Coefficient']:.3f}\t{row['p']:.4f}\t{row['AIC']:.1f}\t{row['BIC']:.1f}\t{row['Observations']}\t{row['Participants']}\n")
            
            # Write full model summaries if available
            f.write("\n\nDetailed Model Summaries:\n\n")
            for model_name, model_data in mixed_results.items():
                if 'result' in model_data:
                    f.write(f"\n--- {model_name} ---\n")
                    f.write(str(model_data['result'].summary()) + "\n\n")
    
    print(f"\nMixed-effects model analysis complete. Results saved to: {mixed_output_dir}")
    
    # NEW: Analyze interaction between finger size and pressure
    interaction_output_dir = os.path.join(cap_flow_path, 'results', 'FingerSize_Pressure_Interaction')
    os.makedirs(interaction_output_dir, exist_ok=True)
    
    # Analyze interaction using mixed-effects models - updated approach
    interaction_models = analyze_finger_size_pressure_interaction(controls_df)
    
    # Create diagnostic plots
    plot_mixed_model_diagnostics(interaction_models, controls_df, interaction_output_dir)
    
    # Create visualizations for interaction effects
    plot_finger_size_pressure_interaction(interaction_models, controls_df, interaction_output_dir)
    
    # Create random slopes visualization
    plot_random_slopes_visualization(interaction_models, controls_df, interaction_output_dir)
    
    # Save model comparison and summaries
    with open(os.path.join(interaction_output_dir, 'model_summaries.txt'), 'w') as f:
        f.write("Mixed-Effects Model Analysis of Finger Size × Pressure Interaction\n\n")
        
        # Write model comparison
        valid_models = [name for name, result in interaction_models.items() 
                      if isinstance(result, sm.regression.linear_model.RegressionResultsWrapper)]
        
        if len(valid_models) > 1:
            f.write("Model Comparison:\n")
            f.write("Model\tAIC\tBIC\tLog-Likelihood\n")
            for name in valid_models:
                f.write(f"{name}\t{interaction_models[name].aic:.2f}\t{interaction_models[name].bic:.2f}\t{interaction_models[name].llf:.2f}\n")
            f.write("\n\n")
        
        # Write detailed model summaries
        for model_name in valid_models:
            f.write(f"--- {model_name} Model ---\n")
            f.write(str(interaction_models[model_name].summary()) + "\n\n")
    
    print(f"\nFinger size × pressure interaction analysis complete. Results saved to: {interaction_output_dir}")
    
    return 0

if __name__ == "__main__":
    main() 