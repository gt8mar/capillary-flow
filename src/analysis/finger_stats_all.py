"""
Filename: src/analysis/finger_stats.py

File for analyzing finger-specific metrics in capillary velocity data.

This script:
1. Loads required data and fonts
2. Provides functions for analyzing finger-specific statistics
3. Analyzes correlation between finger size and capillary velocity
4. Calculates hysteresis and pressure-specific velocity impacts
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.font_manager import FontProperties
from typing import Tuple, List, Dict, Optional

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

def calculate_finger_metrics(finger_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived metrics from finger measurements.
    
    Args:
        finger_df: DataFrame containing finger circumference measurements
        
    Returns:
        DataFrame with additional finger metrics
    """
    # Create a copy to avoid modifying the original
    df = finger_df.copy()
    
    # Calculate average finger circumference per participant
    df['Avg_Finger_Circ'] = df[['Pointer top', 'Pointer bottom', 'Middle top', 'Middle bottom', 
                               'Ring top', 'Ring bottom', 'Pinky top', 'Pinky bottom']].mean(axis=1)
    
    # Calculate ratios (top/bottom) for each finger
    df['Pointer_Ratio'] = df['Pointer top'] / df['Pointer bottom']
    df['Middle_Ratio'] = df['Middle top'] / df['Middle bottom']
    df['Ring_Ratio'] = df['Ring top'] / df['Ring bottom']
    df['Pinky_Ratio'] = df['Pinky top'] / df['Pinky bottom']
    
    # Calculate average ratio
    df['Avg_Ratio'] = df[['Pointer_Ratio', 'Middle_Ratio', 'Ring_Ratio', 'Pinky_Ratio']].mean(axis=1)
    
    # Calculate tapering (difference between top and bottom)
    df['Pointer_Taper'] = df['Pointer top'] - df['Pointer bottom']
    df['Middle_Taper'] = df['Middle top'] - df['Middle bottom']
    df['Ring_Taper'] = df['Ring top'] - df['Ring bottom']
    df['Pinky_Taper'] = df['Pinky top'] - df['Pinky bottom']
    
    # Calculate average taper
    df['Avg_Taper'] = df[['Pointer_Taper', 'Middle_Taper', 'Ring_Taper', 'Pinky_Taper']].mean(axis=1)
    
    return df

def calculate_hysteresis(participant_df: pd.DataFrame) -> float:
    """
    Calculate velocity hysteresis (up_down_diff) for a participant.
    
    Args:
        participant_df: DataFrame containing participant data with UpDown column
        
    Returns:
        Hysteresis value (difference between up and down velocities)
    """
    # Calculate up/down velocity differences
    up_velocities = participant_df[participant_df['UpDown'] == 'U']['Video_Median_Velocity']
    down_velocities = participant_df[participant_df['UpDown'] == 'D']['Video_Median_Velocity']
    
    # Calculate hysteresis as the difference between mean up and down velocities
    up_down_diff = np.mean(up_velocities) - np.mean(down_velocities) if len(up_velocities) > 0 and len(down_velocities) > 0 else np.nan
    
    return up_down_diff

def calculate_pressure_specific_velocities(participant_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate velocity at each pressure level.
    
    Args:
        participant_df: DataFrame containing participant data with Pressure column
        
    Returns:
        DataFrame with pressure-specific velocities
    """
    # Basic velocity statistics for each pressure
    pressure_stats = participant_df.pivot_table(
        index='Participant',
        columns='Pressure',
        values='Video_Median_Velocity',
        aggfunc=['mean']
    ).fillna(0)
    
    # Flatten multi-index columns
    pressure_stats.columns = [f'velocity_at_{pressure}psi' 
                            for (_, pressure) in pressure_stats.columns]
    
    return pressure_stats

def calculate_finger_velocity_metrics(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate hysteresis and pressure-specific velocities for each participant.
    
    Args:
        merged_df: DataFrame containing finger metrics and velocity data
        
    Returns:
        DataFrame with additional metrics
    """
    participant_data = []
    
    for participant in merged_df['Participant'].unique():
        participant_df = merged_df[merged_df['Participant'] == participant]
        
        # Calculate hysteresis
        hysteresis = calculate_hysteresis(participant_df)
        
        # Calculate pressure-specific velocities
        pressure_stats = calculate_pressure_specific_velocities(participant_df)
        
        # Get finger measurements
        finger_metrics = participant_df[['Avg_Finger_Circ', 'Avg_Ratio', 'Avg_Taper', 
                                       'Pointer top', 'Pointer bottom', 'Middle top', 
                                       'Middle bottom', 'Ring top', 'Ring bottom', 
                                       'Pinky top', 'Pinky bottom']].iloc[0].to_dict()
        
        # Basic statistics
        stats = {
            'Participant': participant,
            'Hysteresis': hysteresis,
            'Age': participant_df['Age'].iloc[0],
            'Sex': participant_df['Sex'].iloc[0] if 'Sex' in participant_df.columns else None
        }
        
        # Add finger metrics
        stats.update(finger_metrics)
        
        # Add pressure-specific velocities
        stats.update(pressure_stats.iloc[0].to_dict())
        
        participant_data.append(stats)
    
    return pd.DataFrame(participant_data)

def analyze_correlations(merged_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze correlations between finger metrics and velocity.
    
    Args:
        merged_df: DataFrame containing both finger metrics and velocity data
        
    Returns:
        Dictionary of correlation results
    """
    # Ensure we have participants with both finger and velocity data
    participants_with_finger_data = merged_df['Participant'].unique()
    print(f"Found {len(participants_with_finger_data)} participants with finger data")
    
    # Calculate participant-level metrics for correlation analysis
    participant_metrics = merged_df.groupby('Participant', as_index=False).agg({
        'Video_Median_Velocity': 'median',  # Median velocity per participant
        'Avg_Finger_Circ': 'first',         # First value (should be same for all rows per participant)
        'Avg_Ratio': 'first',
        'Avg_Taper': 'first',
        'Pointer top': 'first',
        'Pointer bottom': 'first',
        'Middle top': 'first',
        'Middle bottom': 'first',
        'Ring top': 'first',
        'Ring bottom': 'first',
        'Pinky top': 'first',
        'Pinky bottom': 'first',
        'Age': 'first',
        'Sex': 'first'
    })
    
    # Replace inf values with nan values safely
    cleaned_metrics = participant_metrics.copy()
    for col in cleaned_metrics.select_dtypes(include=['float', 'int']).columns:
        cleaned_metrics[col] = cleaned_metrics[col].replace([np.inf, -np.inf], np.nan)
        
    # Drop rows with NaN or inf values to avoid correlation calculation errors
    cleaned_metrics = cleaned_metrics.dropna(subset=['Video_Median_Velocity', 'Avg_Finger_Circ'])
    print(f"Using {len(cleaned_metrics)} participants for correlation analysis after removing NaN/inf values")
    
    # Calculate correlations for different metrics
    correlations = {}
    
    # Correlation between average finger circumference and velocity
    corr_avg_circ = stats.pearsonr(
        cleaned_metrics['Avg_Finger_Circ'],
        cleaned_metrics['Video_Median_Velocity']
    )
    
    # Correlation between finger ratios and velocity
    corr_avg_ratio = stats.pearsonr(
        cleaned_metrics['Avg_Ratio'],
        cleaned_metrics['Video_Median_Velocity']
    )
    
    # Correlation between taper and velocity
    corr_avg_taper = stats.pearsonr(
        cleaned_metrics['Avg_Taper'],
        cleaned_metrics['Video_Median_Velocity']
    )
    
    # Store results in dictionary
    correlations['summary'] = pd.DataFrame({
        'Metric': ['Average Circumference', 'Average Ratio', 'Average Taper'],
        'Correlation': [corr_avg_circ[0], corr_avg_ratio[0], corr_avg_taper[0]],
        'p-value': [corr_avg_circ[1], corr_avg_ratio[1], corr_avg_taper[1]]
    })
    
    # Individual finger correlations
    finger_parts = [
        'Pointer top', 'Pointer bottom', 'Middle top', 'Middle bottom',
        'Ring top', 'Ring bottom', 'Pinky top', 'Pinky bottom'
    ]
    
    finger_corr = []
    finger_p = []
    
    for part in finger_parts:
        corr = stats.pearsonr(
            cleaned_metrics[part],
            cleaned_metrics['Video_Median_Velocity']
        )
        finger_corr.append(corr[0])
        finger_p.append(corr[1])
    
    correlations['fingers'] = pd.DataFrame({
        'Finger_Part': finger_parts,
        'Correlation': finger_corr,
        'p-value': finger_p
    })
    
    # Print correlation results
    print("\nCorrelation Summary:")
    print(correlations['summary'])
    
    print("\nIndividual Finger Correlations:")
    print(correlations['fingers'])
    
    return correlations, cleaned_metrics

def plot_correlation_results(correlations: Dict[str, pd.DataFrame], 
                           participant_metrics: pd.DataFrame,
                           output_dir: str) -> None:
    """
    Create visualization of correlation results.
    
    Args:
        correlations: Dictionary of correlation DataFrames
        participant_metrics: DataFrame with participant-level metrics
        output_dir: Directory to save output plots
    """
    # Check if we have enough data points for meaningful visualization
    if len(participant_metrics) < 3:
        print("WARNING: Not enough valid data points for meaningful correlation plots")
        return
        
    # 1. Bar plot of correlation coefficients for different metrics
    plt.figure(figsize=(3.5, 2.5))
    summary_df = correlations['summary']
    
    # Create color palette based on significance
    colors = ['#1f77b4' if p > 0.05 else '#2ca02c' for p in summary_df['p-value']]
    
    # Create the bar plot
    ax = plt.bar(summary_df['Metric'], summary_df['Correlation'], color=colors)
    
    if source_sans:
        plt.xlabel('Finger Metric', fontproperties=source_sans)
        plt.ylabel('Correlation with Velocity', fontproperties=source_sans)
        plt.title('Correlation between Finger Metrics and Velocity', fontproperties=source_sans)
    else:
        plt.xlabel('Finger Metric')
        plt.ylabel('Correlation with Velocity')
        plt.title('Correlation between Finger Metrics and Velocity')
    
    # Add significance annotation
    for i, p in enumerate(summary_df['p-value']):
        if p < 0.05:
            plt.text(i, summary_df['Correlation'][i] + 0.02, '*', 
                    ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'finger_metrics_correlation.png'), dpi=300)
    plt.close()
    
    # 2. Scatter plot of average finger circumference vs velocity
    plt.figure(figsize=(3.0, 2.5))
    
    # Add text labels for each point (participant ID)
    ax = sns.scatterplot(
        x='Avg_Finger_Circ', 
        y='Video_Median_Velocity', 
        data=participant_metrics,
        s=40
    )
    
    # Add regression line
    sns.regplot(
        x='Avg_Finger_Circ', 
        y='Video_Median_Velocity', 
        data=participant_metrics,
        scatter=False,
        ci=None,
        line_kws={'color': 'red', 'linestyle': '--'}
    )
    
    # Add participant labels next to points
    for i, row in participant_metrics.iterrows():
        ax.text(row['Avg_Finger_Circ'] + 0.05, row['Video_Median_Velocity'], 
               f"P{i+1}", fontsize=6)
    
    # Add correlation info
    corr_value = correlations['summary'].loc[0, 'Correlation']
    p_value = correlations['summary'].loc[0, 'p-value']
    p_text = f"r = {corr_value:.2f}, p = {p_value:.3f}" if p_value >= 0.001 else f"r = {corr_value:.2f}, p < 0.001"
    
    if source_sans:
        plt.xlabel('Average Finger Circumference (cm)', fontproperties=source_sans)
        plt.ylabel('Median Velocity (um/s)', fontproperties=source_sans)
        plt.title('Finger Size vs Capillary Velocity', fontproperties=source_sans)
        ax.text(0.05, 0.95, p_text, transform=ax.transAxes, 
               fontproperties=source_sans, fontsize=6)
    else:
        plt.xlabel('Average Finger Circumference (cm)')
        plt.ylabel('Median Velocity (um/s)')
        plt.title('Finger Size vs Capillary Velocity')
        ax.text(0.05, 0.95, p_text, transform=ax.transAxes, fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'finger_size_vs_velocity.png'), dpi=300)
    plt.close()
    
    # 3. Heatmap of individual finger correlations
    plt.figure(figsize=(3.5, 2.5))
    
    # Reshape the data for the heatmap
    fingers = ['Pointer', 'Middle', 'Ring', 'Pinky']
    positions = ['top', 'bottom']
    
    heatmap_data = np.zeros((len(fingers), len(positions)))
    
    for i, finger in enumerate(fingers):
        for j, pos in enumerate(positions):
            part = f"{finger} {pos}"
            idx = correlations['fingers'][correlations['fingers']['Finger_Part'] == part].index
            if len(idx) > 0:
                heatmap_data[i, j] = correlations['fingers'].loc[idx[0], 'Correlation']
    
    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".2f", 
        xticklabels=positions,
        yticklabels=fingers,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        annot_kws={"size": 6}
    )
    
    if source_sans:
        plt.xlabel('Finger Position', fontproperties=source_sans)
        plt.ylabel('Finger', fontproperties=source_sans)
        plt.title('Correlation between Finger Parts and Velocity', fontproperties=source_sans)
    else:
        plt.xlabel('Finger Position')
        plt.ylabel('Finger')
        plt.title('Correlation between Finger Parts and Velocity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'finger_parts_correlation_heatmap.png'), dpi=300)
    plt.close()

def analyze_finger_effects_by_age_sex(merged_df: pd.DataFrame, output_dir: str) -> None:
    """
    Analyze if finger size effects differ by age and sex without categorization.
    
    Args:
        merged_df: DataFrame containing both finger metrics and velocity data
        output_dir: Directory to save output plots
    """
    # Clean data before analysis - fix the way inf values are replaced
    cleaned_df = merged_df.copy()
    # Replace inf values with nan values safely
    for col in cleaned_df.select_dtypes(include=['float', 'int']).columns:
        cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
    
    cleaned_df = cleaned_df.dropna(subset=['Video_Median_Velocity', 'Avg_Finger_Circ', 'Age', 'Sex'])
    
    if len(cleaned_df) == 0:
        print("WARNING: No valid data available for age/sex analysis after removing NaN values")
        return
    
    # Get unique participants with all required data
    participant_metrics = cleaned_df.groupby('Participant', as_index=False).agg({
        'Video_Median_Velocity': 'median',
        'Avg_Finger_Circ': 'first',
        'Age': 'first',
        'Sex': 'first'
    })
    
    unique_participants = participant_metrics['Participant'].nunique()
    print(f"Using {unique_participants} participants for age/sex analysis after removing NaN values")
    
    # Warning: this analysis may not be meaningful with only 8 participants
    # when split by age and sex
    print("\nWARNING: Analysis by age and sex subgroups may not be reliable with only 8 participants")
    
    # Check if we have enough data
    if unique_participants < 3:
        print("WARNING: Not enough valid data points for meaningful age/sex analysis")
        return
    
    # Create scatter plot with points colored by sex
    plt.figure(figsize=(3.5, 2.5))
    
    # Create the scatter plot by sex
    for sex in participant_metrics['Sex'].unique():
        subset = participant_metrics[participant_metrics['Sex'] == sex]
        
        if len(subset) > 0:
            marker = 'o' if sex == 'F' else '^'
            color = '#1f77b4' if sex == 'F' else '#d62728'
            label = f"{sex}"
            
            plt.scatter(
                subset['Avg_Finger_Circ'],
                subset['Video_Median_Velocity'],
                marker=marker,
                color=color,
                s=50,
                label=label
            )
            
            # Add participant IDs as text labels
            for i, row in subset.iterrows():
                plt.text(row['Avg_Finger_Circ'] + 0.05, row['Video_Median_Velocity'], 
                       f"P{i+1}", fontsize=6)
    
    # Add legend and labels
    if source_sans:
        plt.xlabel('Average Finger Circumference (cm)', fontproperties=source_sans)
        plt.ylabel('Median Velocity (um/s)', fontproperties=source_sans)
        plt.title('Finger Size vs Velocity by Sex', fontproperties=source_sans)
        plt.legend(prop=source_sans)
    else:
        plt.xlabel('Average Finger Circumference (cm)')
        plt.ylabel('Median Velocity (um/s)')
        plt.title('Finger Size vs Velocity by Sex')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'finger_size_by_sex.png'), dpi=300)
    plt.close()
    
    # Create a scatter plot with a regression line for age as a continuous variable
    plt.figure(figsize=(3.5, 2.5))
    
    # Create scatter plot with points colored by age
    scatter = plt.scatter(
        participant_metrics['Avg_Finger_Circ'],
        participant_metrics['Video_Median_Velocity'],
        c=participant_metrics['Age'],
        cmap='viridis',
        s=50,
        alpha=0.7
    )
    
    # Add colorbar for age
    cbar = plt.colorbar(scatter)
    cbar.set_label('Age (years)')
    
    # Add participant IDs as text labels
    for i, row in participant_metrics.iterrows():
        plt.text(row['Avg_Finger_Circ'] + 0.05, row['Video_Median_Velocity'], 
               f"P{i+1}", fontsize=6)
    
    # Add regression line
    x = participant_metrics['Avg_Finger_Circ']
    y = participant_metrics['Video_Median_Velocity']
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linestyle='--')
    
    # Add correlation info
    corr = stats.pearsonr(x, y)
    corr_text = f"r = {corr[0]:.2f}, p = {corr[1]:.3f}"
    
    if source_sans:
        plt.xlabel('Average Finger Circumference (cm)', fontproperties=source_sans)
        plt.ylabel('Median Velocity (um/s)', fontproperties=source_sans)
        plt.title('Finger Size vs Velocity (Age as Continuous)', fontproperties=source_sans)
        plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, 
               fontproperties=source_sans, fontsize=6)
    else:
        plt.xlabel('Average Finger Circumference (cm)')
        plt.ylabel('Median Velocity (um/s)')
        plt.title('Finger Size vs Velocity (Age as Continuous)')
        plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'finger_size_by_age_continuous.png'), dpi=300)
    plt.close()

def perform_regression_analysis(merged_df: pd.DataFrame) -> None:
    """
    Perform multivariate regression to assess finger size effect while
    controlling for age and sex as continuous variables.
    
    Args:
        merged_df: DataFrame containing both finger metrics and velocity data
    """
    # Clean data before analysis
    cleaned_df = merged_df.copy()
    # Replace inf values with nan values safely
    for col in cleaned_df.select_dtypes(include=['float', 'int']).columns:
        cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
    
    cleaned_df = cleaned_df.dropna(subset=['Video_Median_Velocity', 'Avg_Finger_Circ', 'Age', 'Sex'])
    
    if len(cleaned_df) == 0:
        print("\nWARNING: No valid data available for regression analysis after removing NaN values")
        return
    
    # Prepare participant-level data for regression
    participant_data = cleaned_df.groupby('Participant', as_index=False).agg({
        'Video_Median_Velocity': 'median',
        'Avg_Finger_Circ': 'first',
        'Age': 'first',
        'Sex': 'first'
    })
    
    unique_participants = participant_data['Participant'].nunique()
    print(f"\nUsing {unique_participants} participants for regression analysis after removing NaN values")
    
    # Warning about sample size
    print("\nWARNING: Multiple regression with n={} may not produce reliable results".format(unique_participants))
    print("The following analysis is provided for exploratory purposes only")
    
    # Check if we have enough data
    if unique_participants < 3:
        print("WARNING: Not enough valid data points for meaningful regression analysis")
        return
    
    # One-hot encode sex
    participant_data['Sex_F'] = (participant_data['Sex'] == 'F').astype(int)
    
    # Univariate models
    print("\nUnivariate Regressions:")
    
    # 1. Finger size only
    from statsmodels.formula.api import ols
    model_finger = ols('Video_Median_Velocity ~ Avg_Finger_Circ', data=participant_data).fit()
    print("\nFinger Size Only:")
    print(model_finger.summary().tables[1])
    
    # 2. Age only
    model_age = ols('Video_Median_Velocity ~ Age', data=participant_data).fit()
    print("\nAge Only:")
    print(model_age.summary().tables[1])
    
    # 3. Sex only
    model_sex = ols('Video_Median_Velocity ~ Sex_F', data=participant_data).fit()
    print("\nSex Only:")
    print(model_sex.summary().tables[1])
    
    # 4. Multivariate model (with strong caution about overfitting)
    print("\nMultivariate Model (CAUTION - likely overfitted):")
    model_multi = ols('Video_Median_Velocity ~ Avg_Finger_Circ + Age + Sex_F', 
                     data=participant_data).fit()
    print(model_multi.summary().tables[1])
    
    # Calculate adjusted R-squared for each model
    print("\nModel Comparison:")
    print(f"Finger Size Only - R²: {model_finger.rsquared:.3f}, Adj. R²: {model_finger.rsquared_adj:.3f}")
    print(f"Age Only - R²: {model_age.rsquared:.3f}, Adj. R²: {model_age.rsquared_adj:.3f}")
    print(f"Sex Only - R²: {model_sex.rsquared:.3f}, Adj. R²: {model_sex.rsquared_adj:.3f}")
    print(f"Multivariate - R²: {model_multi.rsquared:.3f}, Adj. R²: {model_multi.rsquared_adj:.3f}")
    
    # 5. Interaction model between Age and Finger Circumference (using continuous variables)
    print("\n\nInteraction Analysis:")
    print("=" * 50)
    
    try:
        # Run model with interaction term
        model_interaction = ols('Video_Median_Velocity ~ Avg_Finger_Circ * Age', 
                             data=participant_data).fit()
        
        print("\nRegression with Interaction Term (Avg_Finger_Circ * Age):")
        print(model_interaction.summary().tables[1])
        
        # ANOVA comparison between model with and without interaction
        from statsmodels.stats.anova import anova_lm
        
        # Create model without interaction for comparison
        model_no_interaction = ols('Video_Median_Velocity ~ Avg_Finger_Circ + Age', 
                                data=participant_data).fit()
        
        # Perform ANOVA comparison
        anova_results = anova_lm(model_no_interaction, model_interaction)
        print("\nANOVA Comparison:")
        print(anova_results)
        
        # Check if interaction is significant
        p_value = anova_results.iloc[1, 4]  # p-value for the model comparison
        if p_value < 0.05:
            print("\nResult: The interaction between age and finger circumference is significant (p < 0.05)")
            print("This suggests that the relationship between finger size and velocity differs across ages")
        else:
            print("\nResult: The interaction between age and finger circumference is not significant (p > 0.05)")
            print("This suggests that the relationship between finger size and velocity is consistent across ages")
        
        # Create Visualization of the Interaction Effect
        plt.figure(figsize=(4, 3))
        
        # Create scatter plot with data points
        plt.scatter(
            participant_data['Avg_Finger_Circ'], 
            participant_data['Video_Median_Velocity'],
            c=participant_data['Age'],
            cmap='viridis',
            alpha=0.7
        )
        
        # Add colorbar for age
        cbar = plt.colorbar()
        cbar.set_label('Age (years)')
        
        # Add regression line
        x = participant_data['Avg_Finger_Circ']
        y = participant_data['Video_Median_Velocity']
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x + b, color='red', linestyle='--')
        
        if source_sans:
            plt.xlabel('Average Finger Circumference (cm)', fontproperties=source_sans)
            plt.ylabel('Median Velocity (um/s)', fontproperties=source_sans)
            plt.title('Finger Size vs Velocity with Age Interaction', fontproperties=source_sans)
        else:
            plt.xlabel('Average Finger Circumference (cm)')
            plt.ylabel('Median Velocity (um/s)')
            plt.title('Finger Size vs Velocity with Age Interaction')
        
        plt.tight_layout()
        output_dir = os.path.join(cap_flow_path, 'results', 'FingerStats')
        plt.savefig(os.path.join(output_dir, 'age_finger_interaction_continuous.png'), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"\nWarning: Could not complete interaction analysis due to: {str(e)}")
        print("This may be due to insufficient data or collinearity issues.")

def perform_detailed_anova(merged_df: pd.DataFrame, output_dir: str) -> None:
    """
    Perform ANOVA to analyze how finger size impacts median velocity and its correlation with age.
    Treats all variables as continuous without categorization.
    
    Args:
        merged_df: DataFrame containing both finger metrics and velocity data
        output_dir: Directory to save output plots
    """
    # Clean the data
    cleaned_df = merged_df.copy()
    # Replace inf values with nan values safely
    for col in cleaned_df.select_dtypes(include=['float', 'int']).columns:
        cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
    
    cleaned_df = cleaned_df.dropna(subset=['Video_Median_Velocity', 'Avg_Finger_Circ', 'Age', 'Sex'])
    
    if len(cleaned_df) == 0:
        print("\nWARNING: No valid data available for ANOVA analysis after removing NaN values")
        return
    
    # Get participant-level data
    participant_data = cleaned_df.groupby('Participant', as_index=False).agg({
        'Video_Median_Velocity': 'median',
        'Avg_Finger_Circ': 'first',
        'Pointer top': 'first',
        'Pointer bottom': 'first',
        'Middle top': 'first',
        'Middle bottom': 'first',
        'Ring top': 'first',
        'Ring bottom': 'first',
        'Pinky top': 'first',
        'Pinky bottom': 'first',
        'Age': 'first',
        'Sex': 'first'
    })
    
    unique_participants = participant_data['Participant'].nunique()
    if unique_participants < 5:
        print(f"\nWARNING: Too few participants ({unique_participants}) for meaningful ANOVA analysis")
        print("Results may not be reliable and should be interpreted with caution")
         
    print("\n\nANOVA Analysis: Finger Size, Age, and Velocity")
    print("=" * 50)
    
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
        
        # One-hot encode sex
        participant_data['Sex_F'] = (participant_data['Sex'] == 'F').astype(int)
        
        # 1. ANCOVA: Using continuous variables (Age and Finger Size)
        print("\n1. ANCOVA: Using continuous variables (Age and Finger Size)")
        model_continuous = ols('Video_Median_Velocity ~ Age + Avg_Finger_Circ + Age:Avg_Finger_Circ', 
                            data=participant_data).fit()
        print(model_continuous.summary().tables[1])
        
        # 2. Compare model with interaction to model without interaction
        print("\n2. Comparing models with and without interaction:")
        model_no_interaction = ols('Video_Median_Velocity ~ Age + Avg_Finger_Circ', 
                                data=participant_data).fit()
        anova_results = anova_lm(model_no_interaction, model_continuous)
        print(anova_results)
        
        # Check if interaction is significant
        interaction_p_value = anova_results.iloc[1, 4]  # p-value for the model comparison
        if interaction_p_value < 0.05:
            print("\nResult: The interaction between age and finger size is significant (p < 0.05)")
            print("This suggests that the effect of finger size on velocity depends on age")
        else:
            print("\nResult: The interaction between age and finger size is not significant (p > 0.05)")
            print("This suggests that the effect of finger size on velocity is consistent across ages")
        
        # 3. Look at individual finger measurements and velocity
        finger_columns = ['Pointer top', 'Pointer bottom', 'Middle top', 'Middle bottom',
                        'Ring top', 'Ring bottom', 'Pinky top', 'Pinky bottom']
        
        print("\n3. Correlation between individual finger measurements and velocity:")
        correlation_results = []
        
        for finger in finger_columns:
            # Fix: Use Q() to properly quote column names with spaces
            formula = f'Video_Median_Velocity ~ Q("{finger}")'
            model = ols(formula, data=participant_data).fit()
            r_squared = model.rsquared
            # Fix: Use the column name directly instead of integer indexing
            p_value = model.pvalues.iloc[1]  # p-value for the finger measurement
            correlation_results.append({
                'Finger_Part': finger,
                'R_squared': r_squared,
                'p_value': p_value
            })
        
        corr_df = pd.DataFrame(correlation_results)
        print(corr_df.sort_values('R_squared', ascending=False))
        
        # 4. Visualizations
        # Create interaction plot: Finger size vs Velocity by Age
        plt.figure(figsize=(5, 4))
        
        # Scatter plot colored by age
        scatter = plt.scatter(
            participant_data['Avg_Finger_Circ'], 
            participant_data['Video_Median_Velocity'],
            c=participant_data['Age'],
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        
        # Add colorbar for age
        cbar = plt.colorbar(scatter)
        cbar.set_label('Age (years)')
        
        # Add regression line for all data
        x = participant_data['Avg_Finger_Circ']
        y = participant_data['Video_Median_Velocity']
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x + b, color='red', linestyle='--', label=f'Overall trend')
        
        # Add correlation info
        corr = stats.pearsonr(x, y)
        corr_text = f"r = {corr[0]:.2f}, p = {corr[1]:.3f}"
        
        if source_sans:
            plt.xlabel('Average Finger Circumference (cm)', fontproperties=source_sans)
            plt.ylabel('Median Velocity (um/s)', fontproperties=source_sans)
            plt.title('Finger Size vs Velocity with Age', fontproperties=source_sans)
            plt.legend(prop=source_sans)
            plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, 
                  fontproperties=source_sans, fontsize=6)
        else:
            plt.xlabel('Average Finger Circumference (cm)')
            plt.ylabel('Median Velocity (um/s)')
            plt.title('Finger Size vs Velocity with Age')
            plt.legend()
            plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, fontsize=6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'finger_size_vs_velocity_by_age_continuous.png'), dpi=300)
        plt.close()
        
        # Create heatmap of all correlations between finger measurements, age, and velocity
        plt.figure(figsize=(8, 6))
        
        # Get correlation matrix
        correlation_columns = ['Video_Median_Velocity', 'Age'] + finger_columns + ['Avg_Finger_Circ']
        correlation_matrix = participant_data[correlation_columns].corr()
        
        # Create mask to hide upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(
            correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            annot_kws={"size": 6}
        )
        
        if source_sans:
            plt.title('Correlation Matrix: Finger Size, Age, and Velocity', fontproperties=source_sans)
        else:
            plt.title('Correlation Matrix: Finger Size, Age, and Velocity')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"\nWarning: Could not complete ANOVA analysis due to: {str(e)}")
        print("This may be due to insufficient data points or other statistical constraints.")
        import traceback
        traceback.print_exc()

def analyze_pressure_specific_correlations(metrics_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze correlations between finger metrics and pressure-specific velocities.
    
    Args:
        metrics_df: DataFrame containing finger and velocity metrics
        
    Returns:
        Dictionary of correlation results
    """
    # Clean the data
    cleaned_df = metrics_df.copy()
    # Replace inf values with nan values safely
    for col in cleaned_df.select_dtypes(include=['float', 'int']).columns:
        cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
        
    cleaned_df = cleaned_df.dropna(subset=['Avg_Finger_Circ'])
    
    print(f"Analyzing correlations for {len(cleaned_df)} participants with valid data")
    
    # Identify pressure-specific velocity columns
    pressure_cols = [col for col in cleaned_df.columns if col.startswith('velocity_at_')]
    print(f"Found {len(pressure_cols)} pressure-specific velocity measurements")
    
    # Finger metric columns
    finger_metrics = [
        'Avg_Finger_Circ', 'Avg_Ratio', 'Avg_Taper',
        'Pointer top', 'Pointer bottom', 'Middle top', 'Middle bottom',
        'Ring top', 'Ring bottom', 'Pinky top', 'Pinky bottom'
    ]
    
    correlations = {}
    
    # 1. Correlations between average finger circumference and pressure-specific velocities
    pressure_correlations = []
    
    for pressure_col in pressure_cols:
        # Clean data for this specific correlation
        valid_data = cleaned_df.dropna(subset=[pressure_col])
        
        if len(valid_data) < 3:
            print(f"WARNING: Not enough valid data points for {pressure_col}")
            corr = np.nan
            p_val = np.nan
        else:
            # Calculate correlation
            corr_result = stats.pearsonr(
                valid_data['Avg_Finger_Circ'],
                valid_data[pressure_col]
            )
            corr = corr_result[0]
            p_val = corr_result[1]
        
        # Extract pressure value from column name
        pressure = pressure_col.replace('velocity_at_', '').replace('psi', '')
        
        pressure_correlations.append({
            'Pressure': float(pressure),
            'Correlation': corr,
            'p-value': p_val
        })
    
    correlations['pressure_specific'] = pd.DataFrame(pressure_correlations)
    
    # Sort by pressure
    correlations['pressure_specific'] = correlations['pressure_specific'].sort_values('Pressure')
    
    # 2. Correlations between all finger metrics and selected pressure velocities
    # Focus on 0.4 psi and 1.2 psi (which were identified as predictive in health_classifier)
    key_pressures = ['velocity_at_0.4psi', 'velocity_at_1.2psi']
    avail_pressures = [p for p in key_pressures if p in cleaned_df.columns]
    
    if not avail_pressures:
        print("WARNING: Key pressure points (0.4 psi, 1.2 psi) not available in data")
        # Find closest available pressures
        avail_pressure_vals = [float(p.replace('velocity_at_', '').replace('psi', '')) 
                             for p in pressure_cols]
        target_pressures = [0.4, 1.2]
        closest_pressures = []
        
        for target in target_pressures:
            if avail_pressure_vals:
                closest = min(avail_pressure_vals, key=lambda x: abs(x - target))
                closest_pressures.append(f'velocity_at_{closest}psi')
        
        avail_pressures = closest_pressures
        print(f"Using closest available pressures: {avail_pressures}")
    
    for pressure_col in avail_pressures:
        metric_correlations = []
        
        for metric in finger_metrics:
            # Clean data for this specific correlation
            valid_data = cleaned_df.dropna(subset=[pressure_col, metric])
            
            if len(valid_data) < 3:
                print(f"WARNING: Not enough valid data points for {metric} vs {pressure_col}")
                corr = np.nan
                p_val = np.nan
            else:
                # Calculate correlation
                corr_result = stats.pearsonr(
                    valid_data[metric],
                    valid_data[pressure_col]
                )
                corr = corr_result[0]
                p_val = corr_result[1]
            
            metric_correlations.append({
                'Finger_Metric': metric,
                'Correlation': corr,
                'p-value': p_val
            })
        
        # Extract pressure value from column name for dict key
        pressure = pressure_col.replace('velocity_at_', '').replace('psi', '')
        correlations[f'metrics_at_{pressure}psi'] = pd.DataFrame(metric_correlations)
    
    # 3. Correlation between hysteresis and finger metrics
    if 'Hysteresis' in cleaned_df.columns:
        hysteresis_correlations = []
        
        for metric in finger_metrics:
            # Clean data for this specific correlation
            valid_data = cleaned_df.dropna(subset=['Hysteresis', metric])
            
            if len(valid_data) < 3:
                print(f"WARNING: Not enough valid data points for {metric} vs Hysteresis")
                corr = np.nan
                p_val = np.nan
            else:
                # Calculate correlation
                corr_result = stats.pearsonr(
                    valid_data[metric],
                    valid_data['Hysteresis']
                )
                corr = corr_result[0]
                p_val = corr_result[1]
            
            hysteresis_correlations.append({
                'Finger_Metric': metric,
                'Correlation': corr,
                'p-value': p_val
            })
        
        correlations['hysteresis'] = pd.DataFrame(hysteresis_correlations)
    
    return correlations

def plot_pressure_specific_correlations(correlations: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Create visualization of pressure-specific correlation results.
    
    Args:
        correlations: Dictionary of correlation DataFrames
        output_dir: Directory to save output plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot correlation between finger size and velocity at different pressures
    if 'pressure_specific' in correlations:
        pressure_df = correlations['pressure_specific']
        
        if len(pressure_df) > 1:
            plt.figure(figsize=(4, 3))
            
            # Set significant correlations in green, non-significant in blue
            colors = ['#1f77b4' if p > 0.05 else '#2ca02c' for p in pressure_df['p-value']]
            
            # Create the line plot
            plt.plot(pressure_df['Pressure'], pressure_df['Correlation'], 'o-', color='#1f77b4')
            
            # Highlight significant correlations
            sig_mask = pressure_df['p-value'] < 0.05
            if sig_mask.any():
                plt.plot(
                    pressure_df.loc[sig_mask, 'Pressure'],
                    pressure_df.loc[sig_mask, 'Correlation'],
                    'o', color='#2ca02c', markersize=8
                )
            
            # Add reference line at y=0
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            if source_sans:
                plt.xlabel('Applied Pressure (psi)', fontproperties=source_sans)
                plt.ylabel('Correlation with Avg. Finger Circumference', fontproperties=source_sans)
                plt.title('Correlation Between Finger Size and Velocity by Pressure', fontproperties=source_sans)
            else:
                plt.xlabel('Applied Pressure (psi)')
                plt.ylabel('Correlation with Avg. Finger Circumference')
                plt.title('Correlation Between Finger Size and Velocity by Pressure')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'finger_pressure_correlations.png'), dpi=300)
            plt.close()
            
            # Save correlation results as text
            with open(os.path.join(output_dir, 'pressure_correlations.txt'), 'w') as f:
                f.write("Correlation between finger size and velocity at different pressures:\n\n")
                pressure_df.to_string(f, index=False, float_format='%.3f')
    
    # 2. Plot correlations for metrics at specific pressures
    for key, df in correlations.items():
        if key.startswith('metrics_at_'):
            pressure = key.replace('metrics_at_', '').replace('psi', '')
            
            if len(df) > 0:
                plt.figure(figsize=(5, 3))
                
                # Sort by correlation strength
                df_sorted = df.sort_values('Correlation')
                
                # Set significant correlations in green, non-significant in blue
                colors = ['#1f77b4' if p > 0.05 else '#2ca02c' for p in df_sorted['p-value']]
                
                # Create the bar plot
                bars = plt.barh(df_sorted['Finger_Metric'], df_sorted['Correlation'], color=colors)
                
                # Add reference line at x=0
                plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                
                if source_sans:
                    plt.xlabel(f'Correlation with Velocity at {pressure} psi', fontproperties=source_sans)
                    plt.ylabel('Finger Metric', fontproperties=source_sans)
                    plt.title(f'Correlation Between Finger Metrics and Velocity at {pressure} psi', 
                             fontproperties=source_sans)
                else:
                    plt.xlabel(f'Correlation with Velocity at {pressure} psi')
                    plt.ylabel('Finger Metric')
                    plt.title(f'Correlation Between Finger Metrics and Velocity at {pressure} psi')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'finger_metrics_at_{pressure}psi.png'), dpi=300)
                plt.close()
                
                # Save correlation results as text
                with open(os.path.join(output_dir, f'metrics_at_{pressure}psi_correlations.txt'), 'w') as f:
                    f.write(f"Correlation between finger metrics and velocity at {pressure} psi:\n\n")
                    df.to_string(f, index=False, float_format='%.3f')
    
    # 3. Plot correlations between finger metrics and hysteresis
    if 'hysteresis' in correlations:
        hysteresis_df = correlations['hysteresis']
        
        if len(hysteresis_df) > 0:
            plt.figure(figsize=(5, 3))
            
            # Sort by correlation strength
            df_sorted = hysteresis_df.sort_values('Correlation')
            
            # Set significant correlations in green, non-significant in blue
            colors = ['#1f77b4' if p > 0.05 else '#2ca02c' for p in df_sorted['p-value']]
            
            # Create the bar plot
            bars = plt.barh(df_sorted['Finger_Metric'], df_sorted['Correlation'], color=colors)
            
            # Add reference line at x=0
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            if source_sans:
                plt.xlabel('Correlation with Hysteresis', fontproperties=source_sans)
                plt.ylabel('Finger Metric', fontproperties=source_sans)
                plt.title('Correlation Between Finger Metrics and Hysteresis', fontproperties=source_sans)
            else:
                plt.xlabel('Correlation with Hysteresis')
                plt.ylabel('Finger Metric')
                plt.title('Correlation Between Finger Metrics and Hysteresis')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'finger_hysteresis_correlations.png'), dpi=300)
            plt.close()
            
            # Save correlation results as text
            with open(os.path.join(output_dir, 'hysteresis_correlations.txt'), 'w') as f:
                f.write("Correlation between finger metrics and hysteresis:\n\n")
                hysteresis_df.to_string(f, index=False, float_format='%.3f')


def main():
    """Main function for finger statistics analysis."""
    print("\nRunning finger statistics analysis for capillary velocity data...")
    
    # Load data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    # standardize finger column by removing first letter of string
    df['Finger'] = df['Finger'].str[1:]
    # make all strings lowercase
    df['Finger'] = df['Finger'].str.lower()
    # make the first letter uppercase
    df['Finger'] = df['Finger'].str.capitalize()
    print(df['Finger'].unique())
    df['Finger'] = df['Finger'].replace('Mid', 'Middle')
    df['Finger'] = df['Finger'].replace('Index', 'Pointer')
    print(df['Finger'].unique())


    # Load finger stats data
    finger_stats_df = pd.read_csv(os.path.join(cap_flow_path, 'finger_stats.csv'))
    
    # Add derived finger metrics
    finger_stats_df = calculate_finger_metrics(finger_stats_df)
    
    # Merge with velocity data
    merged_df = pd.merge(df, finger_stats_df, on='Participant', how='left')

    merged_df = merged_df.dropna(subset=['Pointer top'])
    # save csv as a test
    merged_df.to_csv(os.path.join(cap_flow_path, 'finger_test_df.csv'), index=False)
    # Map from 'Finger' string to the column name holding the bottom size
    bottom_col_map = {f: f"{f} bottom" for f in ['Pointer', 'Middle', 'Ring', 'Pinky']}
    top_col_map = {f: f"{f} top" for f in ['Pointer', 'Middle', 'Ring', 'Pinky']}

    # Define a helper function for robust lookup within the row
    def get_finger_size(row, col_map):
        finger = row['Finger']
        col_name = col_map.get(finger)
        # Check if the finger name is valid and the corresponding column exists in the row
        if col_name and col_name in row.index:
            return row[col_name]
        # Return NaN if finger name is invalid or column doesn't exist (e.g., due to left merge)
        return np.nan

    # Apply the function to get the sizes directly from merged_df columns
    merged_df['FingerSizeBottom'] = merged_df.apply(lambda row: get_finger_size(row, bottom_col_map), axis=1)
    merged_df['FingerSizeTop'] = merged_df.apply(lambda row: get_finger_size(row, top_col_map), axis=1)

    print(merged_df['FingerSizeBottom'])
    print(merged_df['FingerSizeTop'])

    # Filter for control data if needed
    controls_df = merged_df[merged_df['SET'] == 'set01']
    
    # Check how many participants have both finger and velocity data
    participants_with_data = controls_df.dropna(subset=['Video_Median_Velocity'])
    unique_participants = participants_with_data['Participant'].nunique()
    
    print(f"Found {unique_participants} participants with both finger size and velocity data")
    
    if unique_participants < 8:
        print("WARNING: Some participants with finger measurements do not have velocity data")
    
    # # Set up plotting
    # setup_plotting()
    
    # # Create main output directory
    # output_dir = os.path.join(cap_flow_path, 'results', 'FingerStats')
    # os.makedirs(output_dir, exist_ok=True)
    
    # # print("\nRunning analysis using average finger circumference metrics...")
    
    # # Run correlation analysis
    # correlations, participant_metrics = analyze_correlations(controls_df)
    
    # # Plot correlation results
    # plot_correlation_results(correlations, participant_metrics, output_dir)
    
    # # Analyze by age and sex
    # analyze_finger_effects_by_age_sex(controls_df, output_dir)
    
    # # Perform regression analysis
    # perform_regression_analysis(controls_df)
    
    # # Perform detailed ANOVA analysis
    # perform_detailed_anova(controls_df, output_dir)
    
    # # print("\nStandard finger statistics analysis complete.")
    
    # # New analyses for hysteresis and pressure-specific velocities
    # print("\nRunning extended analyses for hysteresis and pressure-specific velocities...")
    
    # # Calculate finger stats with hysteresis and pressure-specific velocities
    # finger_velocity_metrics = calculate_finger_velocity_metrics(controls_df)
    
    # # Create subdirectories for specialized analyses
    # hysteresis_dir = os.path.join(output_dir, 'Hysteresis')
    # pressure_dir = os.path.join(output_dir, 'PressureSpecific')
    # os.makedirs(hysteresis_dir, exist_ok=True)
    # os.makedirs(pressure_dir, exist_ok=True)
    
    # # Analyze correlations between finger metrics and pressure-specific velocities
    # pressure_correlations = analyze_pressure_specific_correlations(finger_velocity_metrics)
    
    # # Plot and save results
    # plot_pressure_specific_correlations(pressure_correlations, pressure_dir)
    
    # # Save summary statistics of finger velocity metrics
    # with open(os.path.join(output_dir, 'finger_velocity_metrics_summary.txt'), 'w') as f:
    #     f.write("Summary statistics for finger velocity metrics:\n\n")
    #     f.write(f"Number of participants: {len(finger_velocity_metrics)}\n\n")
        
    #     # Write summary statistics for hysteresis
    #     f.write("Hysteresis (up-down velocity difference):\n")
    #     f.write(f"Mean: {finger_velocity_metrics['Hysteresis'].mean():.3f}\n")
    #     f.write(f"Std: {finger_velocity_metrics['Hysteresis'].std():.3f}\n")
    #     f.write(f"Min: {finger_velocity_metrics['Hysteresis'].min():.3f}\n")
    #     f.write(f"Max: {finger_velocity_metrics['Hysteresis'].max():.3f}\n\n")
        
    #     # Write summary statistics for pressure-specific velocities
    #     pressure_cols = [col for col in finger_velocity_metrics.columns if col.startswith('velocity_at_')]
    #     f.write("Pressure-specific velocities:\n")
    #     for col in pressure_cols:
    #         pressure = col.replace('velocity_at_', '').replace('psi', '')
    #         f.write(f"Velocity at {pressure} psi:\n")
    #         f.write(f"  Mean: {finger_velocity_metrics[col].mean():.3f}\n")
    #         f.write(f"  Std: {finger_velocity_metrics[col].std():.3f}\n")
    #         f.write(f"  Min: {finger_velocity_metrics[col].min():.3f}\n")
    #         f.write(f"  Max: {finger_velocity_metrics[col].max():.3f}\n\n")
    
    # # Save the finger velocity metrics DataFrame for future analysis
    # finger_velocity_metrics.to_csv(os.path.join(output_dir, 'finger_velocity_metrics.csv'), index=False)
    
    # print("Extended analyses complete. Results saved to:")
    # print(f"- {hysteresis_dir}")
    # print(f"- {pressure_dir}")

    """
    --------------------- now try with just the actual finger measurements -------------------------------
    """
    two_df = controls_df[controls_df['Pressure'] == 0.2]
    four_df = controls_df[controls_df['Pressure'] == 0.4]
    twelve_df = controls_df[controls_df['Pressure'] == 1.2]
    
    # # plot the velocity vs the bottom circumference
    x = two_df['FingerSizeBottom']
    y = two_df['Video_Median_Velocity']
    plt.scatter(x, y)
    plt.title('Velocity vs Bottom Circumference at 0.2 psi')
    plt.xlabel('Bottom Circumference (cm)')
    plt.ylabel('Velocity (um/s)')
    plt.show()
    
    # plot the velocity vs the bottom circumference for the other pressures
    x = four_df['FingerSizeBottom']
    y = four_df['Video_Median_Velocity']
    plt.scatter(x, y)
    plt.title('Velocity vs Bottom Circumference at 0.4 psi')
    plt.xlabel('Bottom Circumference (cm)')
    plt.ylabel('Velocity (um/s)')
    plt.show() 

    # plot the velocity vs the bottom circumference for the other pressures
    x = twelve_df['FingerSizeBottom']
    y = twelve_df['Video_Median_Velocity']
    plt.scatter(x, y)
    plt.title('Velocity vs Bottom Circumference at 1.2 psi')
    plt.xlabel('Bottom Circumference (cm)')
    plt.ylabel('Velocity (um/s)')
    plt.show()
    return 0


if __name__ == "__main__":
    main() 