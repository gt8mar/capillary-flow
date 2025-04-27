"""
Filename: src/analysis/anova.py

File for analyzing the effects of age, blood pressure, and sex on capillary velocity data
using ANOVA statistical testing.

This script:
1. Loads capillary velocity data and computes participant-level medians
2. Performs ANOVA tests to measure the effects of age, blood pressure, and sex
3. Creates visualization of the main effects and interactions
4. Provides LaTeX-formatted tables for publication
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.font_manager import FontProperties
from typing import Tuple, List, Dict, Optional
import argparse
import statsmodels.api as sm
from statsmodels.formula.api import ols
import colorsys

# Import paths from config
from src.config import PATHS, load_source_sans
from src.tools.plotting_utils import create_monochromatic_palette, adjust_brightness_of_colors

cap_flow_path = PATHS['cap_flow']
source_sans = load_source_sans()

# Define consistent color schemes
AGE_COLOR = '#1f77b4'  # Blue
SEX_COLOR = '#674F92'  # Purple
BP_COLOR = '#2ca02c'  # Green

def setup_plotting():
    """Set up matplotlib parameters for consistent plotting."""
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

def summarize_participants(df: pd.DataFrame, age_threshold: int = 50) -> pd.DataFrame:
    """
    Create a dataframe with one row per participant, aggregating velocity data.
    
    Args:
        df: Input DataFrame with participant-level data
        age_threshold: Age threshold for grouping (default: 50)
        
    Returns:
        DataFrame with one row per participant containing median velocities
    """
    # Group by participant
    participant_medians = df.groupby('Participant', as_index=False).agg({
        'Video_Median_Velocity': 'median',  # Median velocity for each participant
        'Age': 'first',                     # Age is constant per participant
        'Sex': 'first',                     # Sex is constant per participant
        'SYS_BP': 'median'                  # Take median blood pressure
    })
        
    # Create categorical variables for grouping
    participant_medians['Age_Group'] = participant_medians['Age'].apply(
        lambda x: f'Above {age_threshold}' if x >= age_threshold else f'Below {age_threshold}'
    )
    participant_medians['BP_Group'] = participant_medians['SYS_BP'].apply(
        lambda x: 'High BP (≥120)' if x >= 120 else 'Normal BP (<120)'
    )
    
    return participant_medians

def print_descriptive_statistics(df: pd.DataFrame) -> None:
    """
    Print descriptive statistics for the dataset.
    
    Args:
        df: DataFrame with participant data
    """
    print("\nDescriptive Statistics:")
    print(f"Total participants: {len(df)}")
    
    # Sex distribution
    sex_counts = df['Sex'].value_counts()
    print(f"Sex distribution: {sex_counts.to_dict()}")
    
    # Age groups
    age_groups = df['Age_Group'].value_counts()
    print(f"Age groups: {age_groups.to_dict()}")
    
    # BP groups
    bp_groups = df['BP_Group'].value_counts()
    print(f"Blood pressure groups: {bp_groups.to_dict()}")
    
    # Velocity statistics by group
    for group in ['Sex', 'Age_Group', 'BP_Group']:
        print(f"\nVelocity statistics by {group}:")
        for subgroup in df[group].unique():
            subgroup_data = df[df[group] == subgroup]['Video_Median_Velocity']
            print(f"  {subgroup} (n={len(subgroup_data)}):")
            print(f"    Median: {subgroup_data.median():.2f} mm/s")
            print(f"    Mean: {subgroup_data.mean():.2f} mm/s")
            print(f"    Standard deviation: {subgroup_data.std():.2f} mm/s")
            print(f"    IQR: {subgroup_data.quantile(0.25):.2f} to {subgroup_data.quantile(0.75):.2f} mm/s")

def perform_anova_analysis(df: pd.DataFrame, log_transform: bool = False) -> Dict:
    """
    Perform ANOVA analysis for effects of age, sex, and blood pressure on velocity.
    
    Args:
        df: DataFrame with participant data
        log_transform: Whether to use log-transformed velocity
        
    Returns:
        Dictionary containing model, results and p-values
    """
    print("\nRunning ANOVA analysis...")
    
    # Choose dependent variable based on log transform flag
    dependent_var = 'Log_Video_Median_Velocity' if log_transform else 'Video_Median_Velocity'
    print(f"Using {'log-transformed' if log_transform else 'raw'} velocity as dependent variable")
    
    # First model: Main effects only
    formula_main = f"{dependent_var} ~ Age + C(Sex) + SYS_BP"
    model_main = ols(formula_main, data=df).fit()
    anova_main = sm.stats.anova_lm(model_main, typ=2)
    
    print("\nANOVA Results (Main Effects):")
    print(anova_main)
    
    # Second model: With interactions
    formula_full = f"{dependent_var} ~ Age + C(Sex) + SYS_BP + Age:C(Sex) + Age:SYS_BP + C(Sex):SYS_BP"
    model_full = ols(formula_full, data=df).fit()
    anova_full = sm.stats.anova_lm(model_full, typ=2)
    
    print("\nANOVA Results (With Interactions):")
    print(anova_full)
    
    return {
        'model_main': model_main,
        'anova_main': anova_main,
        'model_full': model_full,
        'anova_full': anova_full
    }

def plot_main_effects(df: pd.DataFrame, output_dir: str, log_transform: bool = False) -> None:
    """
    Create boxplots for main effects of age, sex, and blood pressure on velocity.
    
    Args:
        df: DataFrame with participant data
        output_dir: Directory to save plots
        log_transform: Whether to plot log-transformed velocity
    """
    dependent_var = 'Log_Video_Median_Velocity' if log_transform else 'Video_Median_Velocity'
    y_label = 'Log Velocity (mm/s)' if log_transform else 'Median Velocity (mm/s)'
    
    # Create color palettes using plotting_utils functions
    age_palette = create_monochromatic_palette(base_color=AGE_COLOR, n_colors=5)
    age_palette = adjust_brightness_of_colors(age_palette, brightness_scale=0.1)
    
    sex_palette = create_monochromatic_palette(base_color=SEX_COLOR, n_colors=5)
    sex_palette = adjust_brightness_of_colors(sex_palette, brightness_scale=0.1)
    
    bp_palette = create_monochromatic_palette(base_color=BP_COLOR, n_colors=5)
    bp_palette = adjust_brightness_of_colors(bp_palette, brightness_scale=0.1)
    
    # Plot effect of Age Group
    plt.figure(figsize=(2.4, 2.0))
    # Fix deprecation warning by assigning hue instead of palette
    ax = sns.boxplot(
        x='Age_Group', 
        y=dependent_var,
        hue='Age_Group', 
        data=df,
        palette=[age_palette[4], age_palette[1]],
        width=0.6,
        fliersize=3,
        legend=False
    )
    sns.swarmplot(
        x='Age_Group', 
        y=dependent_var, 
        data=df, 
        color='black', 
        size=3, 
        alpha=0.7
    )
    
    if source_sans:
        ax.set_title('Effect of Age on Blood Cell Velocity', fontproperties=source_sans, fontsize=8)
        ax.set_xlabel('Age Group', fontproperties=source_sans)
        ax.set_ylabel(y_label, fontproperties=source_sans)
    else:
        ax.set_title('Effect of Age on Blood Cell Velocity', fontsize=8)
        ax.set_xlabel('Age Group')
        ax.set_ylabel(y_label)
    
    # Add counts to x-axis labels - fix set_ticklabels warning
    counts = df['Age_Group'].value_counts().sort_index()
    # Set ticks first, then labels
    ticks = range(len(ax.get_xticklabels()))
    ax.set_xticks(ticks)
    
    xtick_labels = []
    for i, label in enumerate(ax.get_xticklabels()):
        label_text = label.get_text()
        if label_text in counts.index:
            xtick_labels.append(f"{label_text}\n(n={counts[label_text]})")
        else:
            xtick_labels.append(label_text)
    
    ax.set_xticklabels(xtick_labels)
    
    # Add p-value from t-test
    group1 = df[df['Age_Group'].str.startswith('Below')][dependent_var]
    group2 = df[df['Age_Group'].str.startswith('Above')][dependent_var]
    _, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    p_text = f"p = {p_value:.3f}" if p_value >= 0.001 else "p < 0.001"
    
    if source_sans:
        ax.text(0.5, 0.95, p_text, transform=ax.transAxes, 
               ha='center', fontproperties=source_sans, fontsize=6)
    else:
        ax.text(0.5, 0.95, p_text, transform=ax.transAxes, 
               ha='center', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'velocity_by_age{"_log" if log_transform else ""}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot effect of Sex
    plt.figure(figsize=(2.4, 2.0))
    # Fix deprecation warning
    ax = sns.boxplot(
        x='Sex', 
        y=dependent_var,
        hue='Sex', 
        data=df,
        palette=[sex_palette[4], sex_palette[1]],
        width=0.6,
        fliersize=3,
        legend=False
    )
    sns.swarmplot(
        x='Sex', 
        y=dependent_var, 
        data=df, 
        color='black', 
        size=3, 
        alpha=0.7
    )
    
    if source_sans:
        ax.set_title('Effect of Sex on Blood Cell Velocity', fontproperties=source_sans, fontsize=8)
        ax.set_xlabel('Sex', fontproperties=source_sans)
        ax.set_ylabel(y_label, fontproperties=source_sans)
    else:
        ax.set_title('Effect of Sex on Blood Cell Velocity', fontsize=8)
        ax.set_xlabel('Sex')
        ax.set_ylabel(y_label)
    
    # Add counts to x-axis labels - fix set_ticklabels warning
    counts = df['Sex'].value_counts().sort_index()
    # Set ticks first, then labels
    ticks = range(len(ax.get_xticklabels()))
    ax.set_xticks(ticks)
    
    xtick_labels = []
    for i, label in enumerate(ax.get_xticklabels()):
        label_text = label.get_text()
        if label_text in counts.index:
            xtick_labels.append(f"{label_text}\n(n={counts[label_text]})")
        else:
            xtick_labels.append(label_text)
    
    ax.set_xticklabels(xtick_labels)
    
    # Add p-value from t-test
    group1 = df[df['Sex'] == 'M'][dependent_var]
    group2 = df[df['Sex'] == 'F'][dependent_var]
    _, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    p_text = f"p = {p_value:.3f}" if p_value >= 0.001 else "p < 0.001"
    
    if source_sans:
        ax.text(0.5, 0.95, p_text, transform=ax.transAxes, 
               ha='center', fontproperties=source_sans, fontsize=6)
    else:
        ax.text(0.5, 0.95, p_text, transform=ax.transAxes, 
               ha='center', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'velocity_by_sex{"_log" if log_transform else ""}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot effect of Blood Pressure
    plt.figure(figsize=(2.4, 2.0))
    # Fix deprecation warning
    ax = sns.boxplot(
        x='BP_Group', 
        y=dependent_var,
        hue='BP_Group', 
        data=df,
        palette=[bp_palette[4], bp_palette[1]],
        width=0.6,
        fliersize=3,
        legend=False
    )
    sns.swarmplot(
        x='BP_Group', 
        y=dependent_var, 
        data=df, 
        color='black', 
        size=3, 
        alpha=0.7
    )
    
    if source_sans:
        ax.set_title('Effect of Blood Pressure on Cell Velocity', fontproperties=source_sans, fontsize=8)
        ax.set_xlabel('Blood Pressure Group', fontproperties=source_sans)
        ax.set_ylabel(y_label, fontproperties=source_sans)
    else:
        ax.set_title('Effect of Blood Pressure on Cell Velocity', fontsize=8)
        ax.set_xlabel('Blood Pressure Group')
        ax.set_ylabel(y_label)
    
    # Add counts to x-axis labels - fix set_ticklabels warning
    counts = df['BP_Group'].value_counts().sort_index()
    # Set ticks first, then labels
    ticks = range(len(ax.get_xticklabels()))
    ax.set_xticks(ticks)
    
    xtick_labels = []
    for i, label in enumerate(ax.get_xticklabels()):
        label_text = label.get_text()
        if label_text in counts.index:
            xtick_labels.append(f"{label_text}\n(n={counts[label_text]})")
        else:
            xtick_labels.append(label_text)
    
    ax.set_xticklabels(xtick_labels)
    
    # Add p-value from t-test
    group1 = df[df['BP_Group'] == 'Normal BP (<120)'][dependent_var]
    group2 = df[df['BP_Group'] == 'High BP (≥120)'][dependent_var]
    _, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    p_text = f"p = {p_value:.3f}" if p_value >= 0.001 else "p < 0.001"
    
    if source_sans:
        ax.text(0.5, 0.95, p_text, transform=ax.transAxes, 
               ha='center', fontproperties=source_sans, fontsize=6)
    else:
        ax.text(0.5, 0.95, p_text, transform=ax.transAxes, 
               ha='center', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'velocity_by_bp{"_log" if log_transform else ""}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def plot_interaction_effects(df: pd.DataFrame, output_dir: str, log_transform: bool = False) -> None:
    """
    Create interaction plots to visualize how factors interact in affecting velocity.
    
    Args:
        df: DataFrame with participant data
        output_dir: Directory to save plots
        log_transform: Whether to plot log-transformed velocity
    """
    dependent_var = 'Log_Video_Median_Velocity' if log_transform else 'Video_Median_Velocity'
    y_label = 'Log Velocity (mm/s)' if log_transform else 'Median Velocity (mm/s)'
    
    # Create color palettes using plotting_utils functions
    sex_palette = create_monochromatic_palette(base_color=SEX_COLOR, n_colors=5)
    sex_palette = adjust_brightness_of_colors(sex_palette, brightness_scale=0.1)
    
    bp_palette = create_monochromatic_palette(base_color=BP_COLOR, n_colors=5)
    bp_palette = adjust_brightness_of_colors(bp_palette, brightness_scale=0.1)
    
    # Age x Sex interaction
    plt.figure(figsize=(3.2, 2.4))
    ax = sns.boxplot(
        x='Age_Group', 
        y=dependent_var,
        hue='Sex',
        data=df,
        palette=[sex_palette[4], sex_palette[1]]
    )
    
    if source_sans:
        ax.set_title('Interaction of Age and Sex on Blood Cell Velocity', 
                    fontproperties=source_sans, fontsize=8)
        ax.set_xlabel('Age Group', fontproperties=source_sans)
        ax.set_ylabel(y_label, fontproperties=source_sans)
        plt.legend(title='Sex', prop=source_sans)
    else:
        ax.set_title('Interaction of Age and Sex on Blood Cell Velocity', fontsize=8)
        ax.set_xlabel('Age Group')
        ax.set_ylabel(y_label)
        plt.legend(title='Sex')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'age_sex_interaction{"_log" if log_transform else ""}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Age x Blood Pressure interaction
    plt.figure(figsize=(3.2, 2.4))
    ax = sns.boxplot(
        x='Age_Group', 
        y=dependent_var,
        hue='BP_Group',
        data=df,
        palette=[bp_palette[4], bp_palette[1]]
    )
    
    if source_sans:
        ax.set_title('Interaction of Age and Blood Pressure on Cell Velocity', 
                    fontproperties=source_sans, fontsize=8)
        ax.set_xlabel('Age Group', fontproperties=source_sans)
        ax.set_ylabel(y_label, fontproperties=source_sans)
        plt.legend(title='Blood Pressure', prop=source_sans)
    else:
        ax.set_title('Interaction of Age and Blood Pressure on Cell Velocity', fontsize=8)
        ax.set_xlabel('Age Group')
        ax.set_ylabel(y_label)
        plt.legend(title='Blood Pressure')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'age_bp_interaction{"_log" if log_transform else ""}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sex x Blood Pressure interaction
    plt.figure(figsize=(3.2, 2.4))
    ax = sns.boxplot(
        x='Sex', 
        y=dependent_var,
        hue='BP_Group',
        data=df,
        palette=[bp_palette[4], bp_palette[1]]
    )
    
    if source_sans:
        ax.set_title('Interaction of Sex and Blood Pressure on Cell Velocity', 
                    fontproperties=source_sans, fontsize=8)
        ax.set_xlabel('Sex', fontproperties=source_sans)
        ax.set_ylabel(y_label, fontproperties=source_sans)
        plt.legend(title='Blood Pressure', prop=source_sans)
    else:
        ax.set_title('Interaction of Sex and Blood Pressure on Cell Velocity', fontsize=8)
        ax.set_xlabel('Sex')
        ax.set_ylabel(y_label)
        plt.legend(title='Blood Pressure')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sex_bp_interaction{"_log" if log_transform else ""}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def export_to_latex(anova_results: Dict, output_dir: str, log_transform: bool = False) -> None:
    """
    Export ANOVA results to LaTeX format.
    
    Args:
        anova_results: Dictionary containing ANOVA results
        output_dir: Directory to save LaTeX output
        log_transform: Whether log-transformed velocity was used
    """
    # Format the main effects table
    main_df = anova_results['anova_main'].reset_index()
    main_df.columns = ['Factor', 'Sum_Sq', 'DF', 'F', 'PR(>F)']
    
    # Create LaTeX table for main effects
    latex_main = f"""
\\begin{{table}}[h]
    \\centering
    \\caption{{ANOVA Results for Main Effects on {"Log-Transformed" if log_transform else "Raw"} Blood Cell Velocity}}
    \\begin{{tabular}}{{lrrrr}}
    \\toprule
    \\textbf{{Factor}} & \\textbf{{Sum Sq}} & \\textbf{{DF}} & \\textbf{{F Value}} & \\textbf{{Pr($>$F)}} \\\\
    \\midrule
"""
    
    for _, row in main_df.iterrows():
        p_value = row['PR(>F)']
        stars = ""
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"
        
        factor_name = row['Factor']
        if factor_name == 'C(Sex)':
            factor_name = 'Sex'
        elif factor_name == 'Residual':
            factor_name = 'Residual'
        
        latex_main += f"    {factor_name} & {row['Sum_Sq']:.3f} & {row['DF']:.0f} & {row['F']:.3f} & {p_value:.4f}{stars} \\\\\n"
    
    latex_main += """    \\bottomrule
    \\multicolumn{5}{l}{\\footnotesize{Significance codes: *** p<0.001, ** p<0.01, * p<0.05}}
    \\end{tabular}
    \\label{tab:anova_main_effects}
\\end{table}
"""
    
    # Format the interaction effects table
    full_df = anova_results['anova_full'].reset_index()
    full_df.columns = ['Factor', 'Sum_Sq', 'DF', 'F', 'PR(>F)']
    
    # Create LaTeX table for full model with interactions
    latex_full = f"""
\\begin{{table}}[h]
    \\centering
    \\caption{{ANOVA Results Including Interaction Effects on {"Log-Transformed" if log_transform else "Raw"} Blood Cell Velocity}}
    \\begin{{tabular}}{{lrrrr}}
    \\toprule
    \\textbf{{Factor}} & \\textbf{{Sum Sq}} & \\textbf{{DF}} & \\textbf{{F Value}} & \\textbf{{Pr($>$F)}} \\\\
    \\midrule
"""
    
    for _, row in full_df.iterrows():
        p_value = row['PR(>F)']
        stars = ""
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"
        
        factor_name = row['Factor']
        if factor_name == 'C(Sex)':
            factor_name = 'Sex'
        elif factor_name == 'Age:C(Sex)':
            factor_name = 'Age:Sex'
        elif factor_name == 'C(Sex):SYS_BP':
            factor_name = 'Sex:SYS_BP'
        elif factor_name == 'Residual':
            factor_name = 'Residual'
        
        latex_full += f"    {factor_name} & {row['Sum_Sq']:.3f} & {row['DF']:.0f} & {row['F']:.3f} & {p_value:.4f}{stars} \\\\\n"
    
    latex_full += """    \\bottomrule
    \\multicolumn{5}{l}{\\footnotesize{Significance codes: *** p<0.001, ** p<0.01, * p<0.05}}
    \\end{tabular}
    \\label{tab:anova_interaction_effects}
\\end{table}
"""
    
    # Print the LaTeX tables to console
    print("\nLaTeX Tables:")
    print(latex_main)
    print(latex_full)
    
    # Save LaTeX to files
    with open(os.path.join(output_dir, f'anova_main_effects{"_log" if log_transform else ""}.tex'), 'w') as f:
        f.write(latex_main)
    
    with open(os.path.join(output_dir, f'anova_interaction_effects{"_log" if log_transform else ""}.tex'), 'w') as f:
        f.write(latex_full)

def main(age_threshold: int = 50):
    """
    Main function for ANOVA analysis of capillary velocity data.
    
    Args:
        age_threshold: Age threshold for grouping participants (default: 50)
    """
    print("\nRunning ANOVA analysis of capillary velocity data...")
    print(f"Using age threshold of {age_threshold} years")
    
    # Create output directory
    output_dir = os.path.join(cap_flow_path, 'results', 'ANOVA_Analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Filter for control group
    controls_df = df[df['SET'] == 'set01'].copy()
    
    # Find velocity column - both names appear in dataset
    if 'Video_Median_Velocity' not in controls_df.columns and 'Video Median Velocity' in controls_df.columns:
        controls_df.rename(columns={'Video Median Velocity': 'Video_Median_Velocity'}, inplace=True)
    
    # Check for required columns
    required_cols = ['Participant', 'Age', 'Sex', 'SYS_BP', 'Video_Median_Velocity']
    
    if not all(col in controls_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in controls_df.columns]
        print(f"Error: Missing required columns: {missing}")
        return 1
    
    # Remove rows with missing values in key columns
    controls_df = controls_df.dropna(subset=required_cols)
    
    # Set up plotting parameters
    setup_plotting()
    
    # Create participant summary with specified age threshold
    participant_df = summarize_participants(controls_df, age_threshold=age_threshold)
    
    # If log transform is needed, add log-transformed velocity column
    participant_df['Log_Video_Median_Velocity'] = np.log(participant_df['Video_Median_Velocity'])
    
    # Print descriptive statistics
    print_descriptive_statistics(participant_df)
    
    # Run ANOVA analysis (without log transform)
    anova_results = perform_anova_analysis(participant_df, log_transform=False)
    
    # Create main effect plots
    plot_main_effects(participant_df, output_dir, log_transform=False)
    
    # Create interaction plots
    plot_interaction_effects(participant_df, output_dir, log_transform=False)
    
    # Export to LaTeX
    export_to_latex(anova_results, output_dir, log_transform=False)
    
    print(f"\nANOVA analysis complete. Results saved to {output_dir}")
    return 0

if __name__ == "__main__":
    # You can change the age threshold here (default is 50)
    main(age_threshold=50) 