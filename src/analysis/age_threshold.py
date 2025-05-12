"""
Filename: src/analysis/age_threshold.py

File for analyzing age thresholds in capillary velocity data.

This script:
1. Determines the optimal age threshold for differentiating velocity distributions
2. Creates boxplots of velocities by age groups
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
from src.tools.plotting_utils import plot_CI_multiple_bands

cap_flow_path = PATHS['cap_flow']
source_sans = load_source_sans()

def create_age_cdf_plot(df: pd.DataFrame, threshold: int, 
                        output_path: str, ax=None, average_by_participant: bool = False) -> Optional[float]:
    """
    Creates a CDF plot for velocities split by age groups based on the given threshold.
    
    Args:
        df: DataFrame containing Age and Video_Median_Velocity columns
        threshold: Age threshold to split groups (younger vs older)
        output_path: Path for finding fonts and saving results (Note: not used for saving if ax is provided)
        ax: Matplotlib axis object to plot on (optional)
        average_by_participant: Flag indicating if data was averaged by participant.
    
    Returns:
        KS statistic measuring the difference between distributions
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.0))
    
    # Create age groups
    df['Age_Group'] = df['Age'].apply(lambda x: f'<{threshold} years' if x < threshold else f'≥{threshold} years')
    
    # Group data
    young_group = df[df['Age'] < threshold]['Video_Median_Velocity']
    old_group = df[df['Age'] >= threshold]['Video_Median_Velocity']

    
    # Check if we have enough data in both groups
    if len(young_group) < 3 or len(old_group) < 3:
        print(f"Warning: Not enough data for age threshold {threshold}")
        return None
    
    # Calculate empirical CDFs
    x_young = np.sort(young_group)
    y_young = np.arange(1, len(x_young) + 1) / len(x_young)
    
    x_old = np.sort(old_group)
    y_old = np.arange(1, len(x_old) + 1) / len(x_old)
    
    # Plot CDFs
    ax.plot(x_young, y_young, 'b-', linewidth=1, label=f'<{threshold} years (n={len(young_group)})')
    ax.plot(x_old, y_old, 'r-', linewidth=1, label=f'≥{threshold} years (n={len(old_group)})')
    
    # Run KS test
    ks_stat, p_value = stats.ks_2samp(young_group, old_group)

    print(f"KS statistic for threshold {threshold}: {ks_stat}")
    print(f"p-value for threshold {threshold}: {p_value}")
    print(f"The median velocity for the low group is {young_group.median()} and the high group is {old_group.median()}")
    print(f"The interquartile range for the low group is {young_group.quantile(0.25)} to {young_group.quantile(0.75)} and the high group is {old_group.quantile(0.25)} to {old_group.quantile(0.75)}")
        
    
    # Add KS test result to plot
    p_text = f"p = {p_value:.3f}" if p_value >= 0.001 else "p < 0.001"
    ax.text(0.05, 0.95, f"KS stat: {ks_stat:.3f}\n{p_text}", 
           transform=ax.transAxes, fontsize=6, va='top')
    
    title_suffix = " (Participant Avg)" if average_by_participant else ""
    
    # Try to use Source Sans font if available
    try:
        source_sans_font = FontProperties(fname=os.path.join(PATHS['downloads'], 
                                                       'Source_Sans_3', 'static', 
                                                       'SourceSans3-Regular.ttf'))
        ax.set_xlabel('Velocity (um/s)', fontproperties=source_sans_font)
        ax.set_ylabel('Cumulative Probability', fontproperties=source_sans_font)
        ax.set_title(f'Velocity CDF by Age (Threshold: {threshold}){title_suffix}', 
                    fontproperties=source_sans_font)
        ax.legend(prop=source_sans_font)
    except:
        # Fall back to default font
        ax.set_xlabel('Velocity (um/s)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Velocity CDF by Age (Threshold: {threshold}){title_suffix}')
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    return ks_stat


def threshold_analysis(df: pd.DataFrame, average_by_participant: bool = False) -> int:
    """
    Analyzes different age thresholds to find the one that best differentiates
    velocity distributions.
    
    Args:
        df: DataFrame containing Age and Video_Median_Velocity columns
        average_by_participant: Flag indicating if data was averaged by participant.
    
    Returns:
        The best age threshold for differentiating velocity distributions
    """
    # Create output directory for plots
    output_dir = os.path.join(cap_flow_path, 'results', 'AgeThreshold')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nAnalyzing age thresholds for velocity distributions...")
    
    # Ensure we have age data
    if 'Age' not in df.columns or df['Age'].isna().all():
        print("Error: Age data is missing or all null. Cannot create age-based plots.")
        return None
    
    # Create a copy of the dataframe for age analysis
    age_df = df.dropna(subset=['Video_Median_Velocity', 'Age']).copy()
    
    # Print age statistics
    print(f"Age range in data: {age_df['Age'].min()} to {age_df['Age'].max()} years")
    print(f"Mean age: {age_df['Age'].mean():.2f} years")
    print(f"Median age: {age_df['Age'].median():.2f} years")
    
    # Test different age thresholds
    thresholds = [25, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79]
    
    print(f"Testing age thresholds: {thresholds}")
    
    # Create individual plots for each threshold
    ks_results = {}
    plot_suffix = "_avg_participant" if average_by_participant else ""
    
    for threshold in thresholds:
        plt.close()
        # Set up style and font
        sns.set_style("whitegrid")
        
        plt.rcParams.update({
            'pdf.fonttype': 42, 'ps.fonttype': 42,
            'font.size': 7, 'axes.labelsize': 7,
            'xtick.labelsize': 6, 'ytick.labelsize': 6,
            'legend.fontsize': 5, 'lines.linewidth': 0.5
        })
        
        fig, ax = plt.subplots(figsize=(2.4, 2.0))
        ks_stat = create_age_cdf_plot(age_df, threshold, cap_flow_path, ax, average_by_participant)
        if ks_stat is not None:
            ks_results[threshold] = ks_stat
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'age_velocity_cdf_threshold_{threshold}{plot_suffix}.png'), 
                   dpi=600, bbox_inches='tight')
        plt.close()
    
    # Find the threshold with the maximum KS statistic (most different distributions)
    if ks_results:
        best_threshold = max(ks_results, key=ks_results.get)
        print(f"\nThreshold with most distinct velocity distributions: {best_threshold} years")
        print(f"KS statistic: {ks_results[best_threshold]:.3f}")
        
        # Create a plot showing KS statistic vs threshold
        if len(ks_results) > 1:
            plt.close()
            # Set up style and font
            sns.set_style("whitegrid")
            
            plt.rcParams.update({
                'pdf.fonttype': 42, 'ps.fonttype': 42,
                'font.size': 7, 'axes.labelsize': 7,
                'xtick.labelsize': 6, 'ytick.labelsize': 6,
                'legend.fontsize': 5, 'lines.linewidth': 0.5
            })
            
            fig, ax = plt.subplots(figsize=(2.4, 2.0))
            thresholds_list = list(ks_results.keys())
            ks_stats = list(ks_results.values())
            
            ax.plot(thresholds_list, ks_stats, 'o-', linewidth=0.5)
            ax.axvline(x=best_threshold, color='red', linestyle='--', 
                      label=f'Best threshold: {best_threshold} years')
            
            title_suffix = " (Participant Avg)" if average_by_participant else ""
            # Try to use Source Sans font if available
            try:
                source_sans_font = FontProperties(fname=os.path.join(PATHS['downloads'], 
                                                               'Source_Sans_3', 'static', 
                                                               'SourceSans3-Regular.ttf'))
                ax.set_xlabel('Age Threshold (years)', fontproperties=source_sans_font)
                ax.set_ylabel('KS Statistic', fontproperties=source_sans_font)
                ax.set_title(f'Capillary Velocity Similarity by Age{title_suffix}', fontproperties=source_sans_font)
                ax.legend(prop=source_sans_font)
            except:
                ax.set_xlabel('Age Threshold (years)')
                ax.set_ylabel('KS Statistic')
                ax.set_title(f'Capillary Velocity Similarity by Age{title_suffix}')
                ax.legend()
                
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'velocity_ks_statistic_vs_threshold{plot_suffix}.png'), 
                       dpi=600, bbox_inches='tight')
            plt.close()
            
        return best_threshold
    
    return None


def plot_velocity_boxplots(df: pd.DataFrame, best_threshold: Optional[int] = None, average_by_participant: bool = False) -> None:
    """
    Creates boxplots of video median velocities grouped by different age categories.
    
    Args:
        df: DataFrame containing Age and Video_Median_Velocity columns
        best_threshold: Optional optimal age threshold from threshold_analysis
        average_by_participant: Flag indicating if data was averaged by participant.
    """
    # Create output directory for plots
    output_dir = os.path.join(cap_flow_path, 'results', 'AgeThreshold')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nCreating velocity boxplots by age groups...")
    
    # Standard plot configuration with robust font loading
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
    
    plot_suffix = "_avg_participant" if average_by_participant else ""
    title_suffix = " (Participant Avg)" if average_by_participant else ""

    # 1. Create boxplot by predefined age groups
    plt.figure(figsize=(2.4, 2.0))
    
    # Create age groups
    age_df = df.dropna(subset=['Video_Median_Velocity', 'Age']).copy()
    age_df['Age_Group'] = pd.cut(
        age_df['Age'], 
        bins=[0, 30, 50, 60, 70, 100], 
        labels=['<30', '30-49', '50-59', '60-69', '70+'],
        include_lowest=True
    )
    
    # Create the boxplot
    if average_by_participant:
        ax = sns.boxplot(
            x='Age_Group',
            y='Video_Median_Velocity',
            data=age_df,
            color='#1f77b4',
            width=0.6,
            showfliers=False # Hide fliers when stripplot is shown
        )
        sns.stripplot(
            x='Age_Group',
            y='Video_Median_Velocity',
            data=age_df,
            color='black', 
            size=3,
            jitter=True,
            alpha=0.6,
            ax=ax
        )
    else:
        ax = sns.boxplot(
            x='Age_Group',
            y='Video_Median_Velocity',
            data=age_df,
            color='#1f77b4',
            width=0.6,
            fliersize=3
        )

    # Set title and labels
    if source_sans:
        ax.set_title(f'Video Median Velocity by Age Group{title_suffix}', fontproperties=source_sans, fontsize=8)
        ax.set_xlabel('Age Group', fontproperties=source_sans)
        ax.set_ylabel('Velocity (um/s)', fontproperties=source_sans)
    else:
        ax.set_title(f'Video Median Velocity by Age Group{title_suffix}', fontsize=8)
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Velocity (um/s)')
    
    # Get counts for each age group
    counts_30 = age_df[age_df['Age_Group'] == '<30']['Video_Median_Velocity'].count()
    counts_50 = age_df[age_df['Age_Group'] == '30-49']['Video_Median_Velocity'].count()
    counts_60 = age_df[age_df['Age_Group'] == '50-59']['Video_Median_Velocity'].count()
    counts_70 = age_df[age_df['Age_Group'] == '60-69']['Video_Median_Velocity'].count()
    counts_100 = age_df[age_df['Age_Group'] == '70+']['Video_Median_Velocity'].count()
    
    # Add counts to the x-tick labels
    xtick_labels = [
        f"<30\n(n={counts_30})",
        f"30-49\n(n={counts_50})",
        f"50-59\n(n={counts_60})",
        f"60-69\n(n={counts_70})",
        f"70+\n(n={counts_100})"
    ]
    ax.set_xticklabels(xtick_labels)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'velocity_by_age_groups{plot_suffix}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create boxplot by optimal threshold if provided
    if best_threshold is not None:
        plt.figure(figsize=(2.4, 2.0))
        
        # Create binary age groups based on threshold
        age_df['Threshold_Group'] = age_df['Age'].apply(
            lambda x: f'<{best_threshold} years' if x < best_threshold else f'≥{best_threshold} years'
        )
        
        if average_by_participant:
            # Create the boxplot
            ax = sns.boxplot(
                x='Threshold_Group',
                y='Video_Median_Velocity',
                data=age_df,
                color='#2ca02c',
                width=0.6, 
                showfliers=False
            )
        else: 
            ax = sns.boxplot(
                x='Threshold_Group',
                y='Video_Median_Velocity',
                data=age_df,
                color='#2ca02c',
                width=0.6,
                fliersize=3
            )
        if average_by_participant:
            sns.stripplot(
                x='Threshold_Group',
                y='Video_Median_Velocity',
                data=age_df,
                color='black', # Or a darker shade like '#1b5e20'
                size=3,
                jitter=True,
                alpha=0.6,
                ax=ax
            )
        
        # Set title and labels
        if source_sans:
            ax.set_title(f'Video Median Velocity by Age Threshold ({best_threshold} years){title_suffix}', 
                        fontproperties=source_sans, fontsize=8)
            ax.set_xlabel('Age Group', fontproperties=source_sans)
            ax.set_ylabel('Velocity (um/s)', fontproperties=source_sans)
        else:
            ax.set_title(f'Video Median Velocity by Age Threshold ({best_threshold} years){title_suffix}', fontsize=8)
            ax.set_xlabel('Age Group')
            ax.set_ylabel('Velocity (um/s)')
        
        # Get counts for each threshold group
        counts_young_threshold = age_df[age_df['Threshold_Group'] == f'<{best_threshold} years']['Video_Median_Velocity'].count()
        counts_old_threshold = age_df[age_df['Threshold_Group'] == f'≥{best_threshold} years']['Video_Median_Velocity'].count()
        
        # Add counts to the x-tick labels
        xtick_labels = [f"<{best_threshold}\n(n={counts_young_threshold})",
                        f">={best_threshold}\n(n={counts_old_threshold})"]
        ax.set_xticklabels(xtick_labels)
        
        # Add statistical annotation
        groups = age_df.groupby('Threshold_Group')['Video_Median_Velocity'].apply(list).to_dict()
        if len(groups) == 2:
            group_values = list(groups.values())
            stat, p_value = stats.mannwhitneyu(group_values[0], group_values[1])
            
            # Add p-value annotation
            p_text = f"p = {p_value:.3f}" if p_value >= 0.001 else "p < 0.001"
            if source_sans:
                ax.text(0.5, 0.95, p_text, transform=ax.transAxes, 
                       ha='center', fontproperties=source_sans, fontsize=6)
            else:
                ax.text(0.5, 0.95, p_text, transform=ax.transAxes, 
                       ha='center', fontsize=6)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'velocity_by_threshold_{best_threshold}{plot_suffix}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def analyze_pressure_specific_thresholds(df: pd.DataFrame, average_by_participant: bool = False) -> Dict[int, int]:
    """
    Analyzes age thresholds separately for each pressure level.
    
    Args:
        df: DataFrame containing Age, Pressure, and Video_Median_Velocity columns
        average_by_participant: Flag indicating if data was averaged by participant.
    
    Returns:
        Dictionary mapping pressure levels to their best age thresholds
    """
    # Create output directory for plots
    output_dir = os.path.join(cap_flow_path, 'results', 'AgeThreshold', 'PressureSpecific')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nAnalyzing pressure-specific age thresholds...")
    
    # Check if we have necessary columns
    if not all(col in df.columns for col in ['Age', 'Pressure', 'Video_Median_Velocity']):
        print("Error: Missing required columns for pressure-specific analysis")
        return {}
    
    # Get unique pressure levels
    pressures = sorted(df['Pressure'].unique())
    print(f"Found {len(pressures)} pressure levels: {pressures}")
    
    best_thresholds = {}
    ks_all_pressures = {}
    
    plot_suffix = "_avg_participant" if average_by_participant else ""
    
    for pressure in pressures:
        print(f"\nAnalyzing threshold for pressure: {pressure}")
        
        # Filter data for this pressure
        pressure_df = df[df['Pressure'] == pressure].dropna(subset=['Video_Median_Velocity', 'Age']).copy()
        
        if len(pressure_df) < 10:
            print(f"Skipping pressure {pressure} - insufficient data ({len(pressure_df)} samples)")
            continue
            
        # Print data size
        print(f"Samples for pressure {pressure}: {len(pressure_df)}")
        
         # Test different age thresholds
        thresholds = [25, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79]
        print(f"Testing thresholds for pressure {pressure}: {thresholds}")
        
        # Test each threshold
        ks_results = {}
        for threshold in thresholds:
            # Create age groups
            pressure_df['Age_Group'] = pressure_df['Age'].apply(
                lambda x: f'<{threshold}' if x < threshold else f'≥{threshold}'
            )
            
            # Get velocity data for each group
            young_group = pressure_df[pressure_df['Age'] < threshold]['Video_Median_Velocity']
            old_group = pressure_df[pressure_df['Age'] >= threshold]['Video_Median_Velocity']
            
            # Skip if not enough data in both groups
            if len(young_group) < 3 or len(old_group) < 3:
                print(f"Skipping threshold {threshold} - insufficient group sizes")
                continue
            
            # Calculate KS statistic
            ks_stat, p_value = stats.ks_2samp(young_group, old_group)
            ks_results[threshold] = ks_stat
            ks_all_pressures[(pressure, threshold)] = ks_stat
            
            # Create CDF plot for this pressure and threshold
            plt.close()
            sns.set_style("whitegrid")
            plt.rcParams.update({
                'pdf.fonttype': 42, 'ps.fonttype': 42,
                'font.size': 7, 'axes.labelsize': 7,
                'xtick.labelsize': 6, 'ytick.labelsize': 6,
                'legend.fontsize': 5, 'lines.linewidth': 0.5
            })
            
            fig, ax = plt.subplots(figsize=(2.4, 2.0))
            ks_stat2 =create_age_cdf_plot(pressure_df, threshold, cap_flow_path, ax, average_by_participant)
            print(f'ks_stat: {ks_stat}, ks_stat2: {ks_stat2}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'pressure_{pressure}_threshold_{threshold}{plot_suffix}.png'), 
                       dpi=600, bbox_inches='tight')
            plt.close()
        
        # Find best threshold for this pressure
        if ks_results:
            best_threshold = max(ks_results, key=ks_results.get)
            best_thresholds[pressure] = best_threshold
            print(f"Best threshold for pressure {pressure}: {best_threshold} years (KS: {ks_results[best_threshold]:.3f})")
    
    # Create summary plot of best thresholds by pressure
    if best_thresholds:
        plt.close()
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'pdf.fonttype': 42, 'ps.fonttype': 42,
            'font.size': 7, 'axes.labelsize': 7,
            'xtick.labelsize': 6, 'ytick.labelsize': 6,
            'legend.fontsize': 5, 'lines.linewidth': 0.5
        })
        
        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        
        pressures = list(best_thresholds.keys())
        thresholds = [best_thresholds[p] for p in pressures]
        
        ax.plot(pressures, thresholds, 'o-', linewidth=1)
        
        title_suffix = " (Participant Avg)" if average_by_participant else ""
        # Try to use Source Sans font if available
        if source_sans:
            ax.set_xlabel('Pressure (PSI)', fontproperties=source_sans)
            ax.set_ylabel('Best Age Threshold (years)', fontproperties=source_sans)
            ax.set_title(f'Optimal Age Threshold by Pressure{title_suffix}', fontproperties=source_sans)
        else:
            ax.set_xlabel('Pressure (PSI)')
            ax.set_ylabel('Best Age Threshold (years)')
            ax.set_title(f'Optimal Age Threshold by Pressure{title_suffix}')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'best_threshold_by_pressure{plot_suffix}.png'), 
                   dpi=600, bbox_inches='tight')
        plt.close()
    
    return best_thresholds


def plot_velocity_boxplots_by_pressure(
    df: pd.DataFrame,
    pressures: Optional[List[float]] = None,
    min_samples: int = 5,
    average_by_participant: bool = False,
) -> None:
    """Create boxplots of *Video_Median_Velocity* by predefined age bins **for each pressure level**.

    Each pressure-specific subset is handled independently, ensuring that the *n* displayed
    in the labels correspond to the exact pool of videos used for that pressure.

    Args:
        df: Complete dataframe containing at least *Age*, *Pressure* and
            *Video_Median_Velocity* columns.
        pressures: If provided, limits the plots to the specified pressure levels.
            When *None* (default) all distinct, non-null pressure values are processed.
        min_samples: Minimal number of rows required for a pressure level to be plotted.
            This avoids generating misleading plots for very sparse subsets.
        average_by_participant: Flag indicating if data was averaged by participant.
    """
    # Determine which pressure levels to plot
    if pressures is None:
        pressures = sorted(df['Pressure'].dropna().unique())

    output_dir = os.path.join(cap_flow_path, 'results', 'AgeThreshold', 'PressureSpecific', 'Boxplots')
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating pressure-specific velocity boxplots by age bins…")

    # Common seaborn / matplotlib style (reuse config from the other boxplot helper)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 5,
        'lines.linewidth': 0.5,
    })

    plot_suffix = "_avg_participant" if average_by_participant else ""
    title_suffix = " (Participant Avg)" if average_by_participant else ""

    for pressure in pressures:
        pressure_df = df[df['Pressure'] == pressure].dropna(subset=['Video_Median_Velocity', 'Age']).copy()

        if len(pressure_df) < min_samples:
            print(f"Skipping pressure {pressure} PSI – insufficient data (n={len(pressure_df)})")
            continue

        # Bin ages exactly the same way as in *plot_velocity_boxplots*
        pressure_df['Age_Group'] = pd.cut(
            pressure_df['Age'],
            bins=[0, 30, 50, 60, 70, 100],
            labels=['<30', '30-49', '50-59', '60-69', '70+'],
            include_lowest=True,
        )

        plt.figure(figsize=(2.4, 2.0))
        if average_by_participant:
            ax = sns.boxplot(
                x='Age_Group',
                y='Video_Median_Velocity',
                data=pressure_df,
                color='#1f77b4',
                width=0.6,
                showfliers=False # Hide fliers when stripplot is shown
            )
            sns.stripplot(
                x='Age_Group',
                y='Video_Median_Velocity',
                data=pressure_df,
                color='black', 
                size=3,
                jitter=True,
                alpha=0.6,
                ax=ax
            )
        else:
            ax = sns.boxplot(
                x='Age_Group',
                y='Video_Median_Velocity',
                data=pressure_df,
                color='#1f77b4',
                width=0.6,
                fliersize=3,
            )

        # Title & labels with Source Sans fallback
        if source_sans:
            ax.set_title(
                f'Velocity at {pressure} PSI by Age Group{title_suffix}',
                fontproperties=source_sans,
                fontsize=8,
            )
            ax.set_xlabel('Age Group', fontproperties=source_sans)
            ax.set_ylabel('Velocity (um/s)', fontproperties=source_sans)
        else:
            ax.set_title(f'Velocity at {pressure} PSI by Age Group{title_suffix}', fontsize=8)
            ax.set_xlabel('Age Group')
            ax.set_ylabel('Velocity (um/s)')

        # Get counts for each age group
        counts_30 = pressure_df[pressure_df['Age_Group'] == '<30']['Video_Median_Velocity'].count()
        counts_50 = pressure_df[pressure_df['Age_Group'] == '30-49']['Video_Median_Velocity'].count()
        counts_60 = pressure_df[pressure_df['Age_Group'] == '50-59']['Video_Median_Velocity'].count()
        counts_70 = pressure_df[pressure_df['Age_Group'] == '60-69']['Video_Median_Velocity'].count()
        counts_100 = pressure_df[pressure_df['Age_Group'] == '70+']['Video_Median_Velocity'].count()
        
        xtick_labels = [
            f"<30\n(n={counts_30})",
            f"30-49\n(n={counts_50})",
            f"50-59\n(n={counts_60})",
            f"60-69\n(n={counts_70})",
            f"70+\n(n={counts_100})"
        ]
        ax.set_xticklabels(xtick_labels)

        plt.tight_layout()
        outfile = os.path.join(
            output_dir,
            f'velocity_by_age_groups_pressure_{pressure}{plot_suffix}.png',
        )
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()

    print("Pressure-specific boxplots saved to:", output_dir)


def main(debug=False, average_by_participant: bool = False):
    """Main function for age threshold analysis."""
    print("\nRunning age threshold analysis for capillary velocity data...")
    
    # Load data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)

    controls_df = df[df['SET'] == 'set01']

    # drop rows with nan video_median_velocity
    controls_df = controls_df[controls_df['Video_Median_Velocity'].notna()]

    if average_by_participant:
        print("Averaging Video_Median_Velocity by Participant and Pressure...")
        # Assuming 'Participant' is the participant identifier and 'Age' is a stable characteristic of the participant.
        # We group by Participant and Pressure, calculate the median velocity,
        # and retain the 'Age' associated with that Participant.
        # If an Participant could have multiple Ages in the dataset, 'first()' takes the first one encountered.
        averaged_df = controls_df.groupby(['Participant', 'Pressure']).agg(
            Video_Median_Velocity=('Video_Median_Velocity', 'median'),
            Age=('Age', 'first')  # Assumes Age is fixed per Participant for the relevant dataset
        ).reset_index()
        
        # Ensure essential columns are not NaN after aggregation/merge
        controls_df = averaged_df.dropna(subset=['Video_Median_Velocity', 'Age'])
        print(f"Number of rows after averaging by participant and pressure: {len(controls_df)}")

    if debug:
        print(f'number of rows in controls_df: {len(controls_df)}')
        print(f'number of unique pressures in controls_df: {controls_df["Pressure"].nunique()}')
        print(f'unique pressures in controls_df: {controls_df["Pressure"].unique()}')
   
        # make slices of controls_df for each pressure level
        pressures = controls_df['Pressure'].unique()
        for pressure in pressures:
            print(f'pressure: {pressure}')
            pressure_df = controls_df[controls_df['Pressure'] == pressure]
            print(f'number of rows in pressure_df: {len(pressure_df)}')
            print(f'number of unique pressures in pressure_df: {pressure_df["Pressure"].nunique()}')
            print(f'unique pressures in pressure_df: {pressure_df["Pressure"].unique()}')

    # make a pressure df for each pressure level, 0.2 through 1.2 in increments of 0.2
    pressure_02_df = controls_df[controls_df['Pressure'] == 0.2]
    pressure_04_df = controls_df[controls_df['Pressure'] == 0.4]
    pressure_06_df = controls_df[controls_df['Pressure'] == 0.6]
    pressure_08_df = controls_df[controls_df['Pressure'] == 0.8]
    pressure_10_df = controls_df[controls_df['Pressure'] == 1.0]
    pressure_12_df = controls_df[controls_df['Pressure'] == 1.2]
    
    # Run threshold analysis to find optimal age cutoff
    best_threshold = threshold_analysis(controls_df, average_by_participant=average_by_participant)
    
    # Plot velocity by age groups
    plot_velocity_boxplots(controls_df, best_threshold, average_by_participant=average_by_participant)
    
    # Run pressure-specific threshold analysis
    pressure_thresholds = analyze_pressure_specific_thresholds(controls_df, average_by_participant=average_by_participant)
    
    # Generate pressure-specific boxplots
    plot_velocity_boxplots_by_pressure(controls_df, average_by_participant=average_by_participant)
    
    print("\nAge threshold analysis complete.")


    # # Plot the CI bands for age groups
    # # plot_CI_multiple_bands(controls_df, thresholds=[29, 49], variable='Age', method='bootstrap', 
    # #                        n_iterations=1000, ci_percentile=95, write=True, dimensionless=False, 
    # #                        video_median=False, log_scale=False, velocity_variable='Corrected Velocity')

    return 0


if __name__ == "__main__":
    main() 
    main(average_by_participant=True)