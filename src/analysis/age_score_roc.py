"""
Filename: src/analysis/age_score_roc.py
--------------------------------------------------

This script calculates age scores and generates ROC curves for different age thresholds.
Specifically designed to generate ROC curves for age thresholds of 29 and 59.

By: Assistant based on plot_big.py functions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import stats
from typing import List, Tuple, Optional

# Import paths from config
from src.config import PATHS

def load_and_preprocess_data() -> pd.DataFrame:
    """
    Loads and preprocesses the data for age score analysis.
    
    Returns:
        DataFrame containing the preprocessed data
    """
    print("\nLoading and preprocessing data...")
    
    # Load data (same path as other analyses)
    data_filepath = os.path.join(PATHS['cap_flow'], 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Basic cleaning
    df = df.dropna(subset=['Age', 'Corrected Velocity'])
    df = df.drop_duplicates(subset=['Participant', 'Video'])
    
    print(f"Loaded {len(df)} records")
    return df

def calculate_cdf_area(data: pd.DataFrame, start: float = 10, end: float = 700) -> Tuple[float, float]:
    """
    Calculate the area under the CDF curve for velocity data.
    
    Args:
        data: DataFrame containing velocity data
        start: Start value for area calculation
        end: End value for area calculation
        
    Returns:
        Tuple of (area, log_area)
    """
    velocities = data['Corrected Velocity'].dropna()
    
    # Generate a linear space from start to end
    x_values = np.linspace(start, end, 1000)
    
    # Calculate empirical CDF
    sorted_velocities = np.sort(velocities)
    cdf_values = []
    
    for x in x_values:
        cdf_value = np.sum(sorted_velocities <= x) / len(sorted_velocities)
        cdf_values.append(cdf_value)
    
    # Calculate area under CDF using trapezoidal rule
    area = np.trapz(cdf_values, x_values)
    
    # Calculate log area (using log of velocities)
    log_velocities = np.log(velocities + 1)  # Add 1 to avoid log(0)
    log_x_values = np.log(x_values + 1)
    sorted_log_velocities = np.sort(log_velocities)
    
    log_cdf_values = []
    for x in log_x_values:
        log_cdf_value = np.sum(sorted_log_velocities <= x) / len(sorted_log_velocities)
        log_cdf_values.append(log_cdf_value)
    
    log_area = np.trapz(log_cdf_values, log_x_values)
    
    return area, log_area

def calculate_age_scores(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate age scores for each participant based on CDF area differences from control.
    
    Args:
        data: DataFrame containing all participant data
        
    Returns:
        DataFrame with age scores for each participant
    """
    print("Calculating age scores...")
    
    # Get control data (set01)
    control_data = data[data['SET'] == 'set01']
    
    # Calculate reference area from control data
    control_area, control_log_area = calculate_cdf_area(control_data)
    print(f"Control area: {control_area:.2f}, Control log area: {control_log_area:.2f}")
    
    # Calculate age scores for each participant
    age_scores = []
    
    for participant in data['Participant'].unique():
        participant_df = data[data['Participant'] == participant]
        
        # Calculate participant's area
        participant_area, participant_log_area = calculate_cdf_area(participant_df)
        
        # Calculate age score as difference from control
        age_score = participant_area - control_area
        log_age_score = participant_log_area - control_log_area
        
        # Get participant's age (assuming constant per participant)
        participant_age = participant_df['Age'].iloc[0]
        
        age_scores.append({
            'Participant': participant,
            'Age': participant_age,
            'Age-Score': age_score,
            'Log Age-Score': log_age_score
        })
    
    age_scores_df = pd.DataFrame(age_scores)
    print(f"Calculated age scores for {len(age_scores_df)} participants")
    
    return age_scores_df

def calculate_auc_ci_delong(y_true: np.ndarray, y_scores: np.ndarray, alpha: float = 0.95) -> Tuple[float, float, float, float]:
    """
    Calculate AUC and confidence interval using DeLong's method.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        alpha: Confidence level
        
    Returns:
        Tuple of (auc, auc_var, ci_lower, ci_upper)
    """
    auc_score = roc_auc_score(y_true, y_scores)
    
    # Simplified variance calculation (full DeLong implementation would be more complex)
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    
    # Approximate variance using Hanley-McNeil method
    q1 = auc_score / (2 - auc_score)
    q2 = 2 * auc_score**2 / (1 + auc_score)
    
    auc_var = (auc_score * (1 - auc_score) + (n1 - 1) * (q1 - auc_score**2) + 
               (n0 - 1) * (q2 - auc_score**2)) / (n1 * n0)
    
    auc_std = np.sqrt(auc_var)
    z_score = stats.norm.ppf(1 - (1 - alpha) / 2)
    
    ci_lower = max(0, auc_score - z_score * auc_std)
    ci_upper = min(1, auc_score + z_score * auc_std)
    
    return auc_score, auc_var, ci_lower, ci_upper

def make_roc_curve_age_threshold(df: pd.DataFrame, feature: str, age_threshold: int, 
                               flip: bool = False, plot: bool = False, write: bool = False, 
                               n_bootstraps: int = 1000, ci_percentile: float = 95) -> dict:
    """
    Generate ROC curve for age classification using specified threshold.
    
    Args:
        df: DataFrame with age scores and ages
        feature: Feature to use for prediction ('Age-Score' or 'Log Age-Score')
        age_threshold: Age threshold for binary classification
        flip: Whether to flip the classification (young=1, old=0)
        plot: Whether to display the plot
        write: Whether to save the plot
        n_bootstraps: Number of bootstrap iterations
        ci_percentile: Confidence interval percentile
        
    Returns:
        Dictionary with ROC results
    """
    print(f"Generating ROC curve for {feature} with age threshold {age_threshold}")
    
    # Create binary age categories
    if flip:
        df['Age Category'] = (df['Age'] < age_threshold).astype(int)
    else:
        df['Age Category'] = (df['Age'] >= age_threshold).astype(int)
    
    # Calculate ROC curve
    y_true = df['Age Category'].values
    y_scores = df[feature].values
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate confidence interval using DeLong's method
    auc_score, auc_var, ci_lower, ci_upper = calculate_auc_ci_delong(y_true, y_scores)
    
    print(f'AUC: {auc_score:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    
    # Bootstrap for additional confidence intervals
    bootstrapped_aucs = []
    bootstrapped_tprs = []
    
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        # Resample with replacement
        indices = rng.choice(len(df), size=len(df), replace=True)
        df_resampled = df.iloc[indices]
        
        y_true_boot = df_resampled['Age Category'].values
        y_scores_boot = df_resampled[feature].values
        
        try:
            fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_scores_boot)
            bootstrapped_auc = auc(fpr_boot, tpr_boot)
            bootstrapped_aucs.append(bootstrapped_auc)
            
            # Interpolate TPR for consistent FPR values
            tpr_interp = np.interp(np.linspace(0, 1, 100), fpr_boot, tpr_boot)
            bootstrapped_tprs.append(tpr_interp)
        except:
            continue  # Skip failed bootstrap iterations
    
    bootstrapped_aucs = np.array(bootstrapped_aucs)
    boot_ci_lower = np.percentile(bootstrapped_aucs, (100 - ci_percentile) / 2)
    boot_ci_upper = np.percentile(bootstrapped_aucs, 100 - (100 - ci_percentile) / 2)
    
    print(f'Bootstrap AUC: {np.mean(bootstrapped_aucs):.3f}, 95% CI: [{boot_ci_lower:.3f}, {boot_ci_upper:.3f}]')
    
    # Plot ROC curve
    plot_roc_with_ci(fpr, tpr, roc_auc, bootstrapped_tprs, ci_lower, ci_upper, 
                     feature, age_threshold, plot=plot, write=write)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_ci_lower': boot_ci_lower,
        'bootstrap_ci_upper': boot_ci_upper
    }

def plot_roc_with_ci(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, 
                     bootstrapped_tprs: List[np.ndarray], ci_lower: float, ci_upper: float,
                     feature: str, age_threshold: int, plot: bool = False, write: bool = False):
    """
    Plot ROC curve with confidence intervals.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: AUC value
        bootstrapped_tprs: Bootstrap TPR values
        ci_lower: Lower confidence interval
        ci_upper: Upper confidence interval
        feature: Feature name
        age_threshold: Age threshold used
        plot: Whether to display plot
        write: Whether to save plot
    """
    # Set up style and font
    sns.set_style("whitegrid")
    
    # Try to load font, fall back to default if not available
    font_path = 'C:\\Users\\gt8mar\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf'
    if os.path.exists(font_path):
        source_sans = FontProperties(fname=font_path)
    else:
        source_sans = FontProperties()  # Use default font
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 4, 'lines.linewidth': 0.5
    })

    base_color = '#1f77b4'

    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    
    # Ensure consistent font sizes by setting them directly on the axes
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    
    # Plot the ROC curve
    ax.plot(fpr, tpr, marker='', linestyle='-', markersize=2, color=base_color, 
            label=f'AUC = {roc_auc:.2f}')
    
    # Plot confidence interval
    if len(bootstrapped_tprs) > 1:
        tprs_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
        tprs_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)
        ax.fill_between(np.linspace(0, 1, 100), tprs_lower, tprs_upper, 
                       color=base_color, alpha=0.2, label=f'95% CI')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', markersize=2, label='Random')
    
    # Set labels and title with consistent font sizes
    ax.set_xlabel('False Positive Rate', fontproperties=source_sans, fontsize=7)
    ax.set_ylabel('True Positive Rate', fontproperties=source_sans, fontsize=7)
    ax.set_title(f'Age Classification using {feature} (threshold={age_threshold})', 
                fontproperties=source_sans, fontsize=8)
    ax.legend(loc='lower right', prop=source_sans, fontsize=4)
    ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    
    if write:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(PATHS['cap_flow'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save plot
        filename = f'roc_curve_{feature.replace(" ", "_").replace("-", "_")}_threshold_{age_threshold}.png'
        plt.savefig(os.path.join(results_dir, filename), dpi=600, bbox_inches='tight')
        print(f"Saved plot: {filename}")
    
    if plot:
        plt.show()
    else:
        plt.close()

def main():
    """Main function to run age score ROC analysis."""
    print("Starting age score ROC analysis...")
    
    # Load data
    df = load_and_preprocess_data()
    
    # Calculate age scores
    age_scores_df = calculate_age_scores(df)
    
    # Generate ROC curves for different thresholds
    thresholds = [29, 59]
    features = ['Age-Score', 'Log Age-Score']
    
    results = {}
    
    for threshold in thresholds:
        print(f"\n=== Age Threshold: {threshold} ===")
        results[threshold] = {}
        
        for feature in features:
            print(f"\nAnalyzing {feature}...")
            
            # Generate ROC curve
            roc_results = make_roc_curve_age_threshold(
                age_scores_df, 
                feature, 
                age_threshold=threshold,
                flip=True,  # Young=1, Old=0 (as in original code)
                plot=False, 
                write=True
            )
            
            results[threshold][feature] = roc_results
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    
    for threshold in thresholds:
        print(f"\nAge Threshold: {threshold}")
        print("-" * 30)
        
        for feature in features:
            res = results[threshold][feature]
            print(f"{feature}:")
            print(f"  AUC: {res['auc']:.3f}")
            print(f"  95% CI: [{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]")
            print(f"  Bootstrap CI: [{res['bootstrap_ci_lower']:.3f}, {res['bootstrap_ci_upper']:.3f}]")

if __name__ == '__main__':
    main()
