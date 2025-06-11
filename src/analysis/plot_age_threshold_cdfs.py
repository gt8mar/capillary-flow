"""
Filename: src/analysis/plot_age_threshold_cdfs.py
--------------------------------------------------

Generate CDF plots for two age thresholds:
1. ≤29 vs ≥30 years
2. ≤59 vs ≥60 years

Each threshold is plotted twice: linear x-axis and log x-axis (velocity +1).
Plots follow the same styling conventions used in plot_big age CDF figures.
All plots are saved to results/cdf_analysis/age_thresholds without displaying
on screen.

By: Marcus Forst
"""

# Standard libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker

# Project utilities
from src.config import PATHS, load_source_sans
from src.tools.plotting_utils import (
    calculate_cdf,
    create_monochromatic_palette,
    adjust_brightness_of_colors,
)

# Styling constants (aligned with docs/coding_standards.md)
BASE_COLOR = '#1f77b4'  # Standard blue for age analyses
LINEWIDTH = 0.75
FIGSIZE = (3.2, 2.4)

# Helper to compute empirical CDF

def empirical_cdf(arr):
    arr_sorted = np.sort(arr)
    cdf = np.arange(1, len(arr_sorted) + 1) / len(arr_sorted)
    return arr_sorted, cdf


def create_palette(base_color=BASE_COLOR):
    """Return two visually separated monochromatic colors."""
    base = sns.color_palette([base_color])[0]
    palette = sns.light_palette(base, n_colors=5, reverse=False)
    return palette[0], palette[3]  # control, comparison


def plot_cdf(ax, data1, data2, labels, colors, log_scale=False):
    """Plot two CDF curves on the provided axis."""
    if log_scale:
        data1 = data1 + 1
        data2 = data2 + 1

    x1, y1 = empirical_cdf(data1)
    x2, y2 = empirical_cdf(data2)

    ax.plot(x1, y1, color=colors[0], linewidth=LINEWIDTH, label=labels[0])
    ax.plot(x2, y2, color=colors[1], linewidth=LINEWIDTH, linestyle='--', label=labels[1])

    # grid & formatting handled by caller


def configure_plot(ax, xlabel, title, source_sans):
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    if source_sans:
        ax.set_xlabel(xlabel, fontproperties=source_sans)
        ax.set_ylabel('Cumulative Probability', fontproperties=source_sans)
        ax.set_title(title, fontproperties=source_sans, fontsize=8)
        ax.legend(prop=source_sans)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(title)
        ax.legend()


def main():
    print('Generating age-threshold CDF plots…')

    # Load dataset (same path as confidence-interval analyses)
    data_fp = os.path.join(PATHS['cap_flow'], 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_fp)

    # Basic cleaning (mirrors previous scripts)
    df = df.dropna(subset=['Age', 'Corrected Velocity'])
    df = df.drop_duplicates(subset=['Participant', 'Video'])

    # Use healthy controls (set01) only
    controls_df = df[df['SET'] == 'set01'].copy()

    thresholds = [(29, '≤29', '≥30'), (59, '≤59', '≥60')]
    scales = [('linear', False), ('log', True)]

    # Colors
    control_color, comp_color = create_palette()

    # Font
    source_sans = load_source_sans()

    # Output dir
    out_dir = os.path.join(PATHS['cap_flow'], 'results', 'cdf_analysis', 'age_thresholds')
    os.makedirs(out_dir, exist_ok=True)

    sns.set_style('whitegrid')
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'lines.linewidth': LINEWIDTH
    })

    for thresh, lab_low, lab_high in thresholds:
        # Split groups
        low_group = controls_df[controls_df['Age'] <= thresh]['Corrected Velocity'].dropna()
        high_group = controls_df[controls_df['Age'] > thresh]['Corrected Velocity'].dropna()

        # Skip if any group empty
        if low_group.empty or high_group.empty:
            print(f'Warning: one group empty for threshold {thresh}. Skipping.')
            continue

        # KS test info (printed for record)
        ks, p_val = ks_2samp(low_group, high_group)
        print(f'Threshold {thresh}: KS={ks:.3f}, p={p_val:.3e}')

        for scale_name, log_flag in scales:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            plot_cdf(ax, low_group, high_group, [lab_low, lab_high], [control_color, comp_color], log_scale=log_flag)

            xlabel = 'Velocity (μm/s)' if not log_flag else 'Velocity + 1 (μm/s)'
            title = f'CDF: {lab_low} vs {lab_high} (Age Threshold {thresh})'
            if log_flag:
                ax.set_xscale('log')
            configure_plot(ax, xlabel, title, source_sans)

            # Save
            fname = f'age_threshold_{thresh}_{scale_name}.png'
            fig.savefig(os.path.join(out_dir, fname), dpi=600, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {fname}')

        # ------------------------------------------------------------------
        # Generate *legacy* plots using the classic styling (plot_cdf_old)
        # ------------------------------------------------------------------

        # Prepare data & subsets in the format expected by plot_cdf_old
        all_data = controls_df['Corrected Velocity'].dropna()
        subsets = [low_group, high_group]

        for scale_name, log_flag in scales:
            legacy_title = f'CDF_oldStyle_age_thresh_{thresh}_{scale_name}'
            plot_cdf_old(
                all_data,
                subsets=subsets.copy(),  # pass a copy to avoid in-place shifts
                labels=['Entire Dataset', lab_low, lab_high],
                title=legacy_title,
                write=True,
                normalize=False,
                variable='Age',
                log=log_flag,
            )
            print(f'Saved legacy plot {legacy_title}.png')

    print('✅ Age threshold CDF plots created.')
    return 0


def plot_cdf_old(data, subsets, labels=['Entire Dataset', 'Subset'], title='CDF Comparison', 
             write=False, normalize=False, variable = 'Age', log = True):
    """
    Plots the CDF of the entire dataset and the inputted subsets.

    Args:
        data (array-like): The entire dataset
        subsets (list of array-like): The subsets to be compared
        labels (list of str): The labels for the entire dataset and the subsets
        title (str): The title of the plot
        write (bool): Whether to write the plot to a file
        normalize (bool): Whether to normalize the CDF
    
    Returns:
        0 if successful, 1 if no subsets provided
    """
    plt.close()
    # Set up style and font
    sns.set_style("whitegrid")
    source_sans = load_source_sans()  # Use the safe font loading function
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

    # Set color palette based on variable
    if variable == 'Age':
        base_color = '#1f77b4'##3FCA54''BDE4A7 #A1E5AB
    elif variable == 'SYS_BP':
        base_color = '2ca02c'#80C6C3 #ff7f0e
    elif variable == 'Sex':
        base_color = '674F92'#947EB0#2ca02c#CAC0D89467bd
    elif variable == 'Individual':
        base_color = '#1f77b4'
        individual_color = '#6B0F1A' #'#ff7f0e'
    elif variable == 'Diabetes_plot':
        base_color = '#ff7f0e' 
    elif variable == 'Hypertension_plot':
        base_color = '#d62728'
    else:
        raise ValueError(f"Unsupported variable: {variable}")

    palette = create_monochromatic_palette(base_color)
    # palette = adjust_saturation_of_colors(palette, saturation_scale=1.3)
    palette = adjust_brightness_of_colors(palette, brightness_scale=.2)
    sns.set_palette(palette)

    if not subsets:
        return 1

    fig, ax = plt.subplots(figsize=(2.4, 2.0))

    if log:
        data = data+1
        for i in range(len(subsets)):
            subsets[i]= subsets[i]+1
   
    # Plot main dataset
    x, y = calculate_cdf(data, normalize)
    ax.plot(x, y, label=labels[0])
    

    # Plot subsets
    for i in  range(len(subsets)):
        if i == 0:
            i_color = 0
            dot_color = 0
        elif i == 1:
            i_color = 3
            dot_color = 2
        elif i == 2:
            i_color = 4
            dot_color=3
        x, y = calculate_cdf(subsets[i], normalize)
        if variable == 'Individual':
            ax.plot(x, y, label=labels[i+1], linestyle='--', color=individual_color)
        else:
            ax.plot(x, y, label=labels[i+1], linestyle='--', color=palette[i_color])

    # Set labels and formatting with safe font handling
    if source_sans:
        ax.set_ylabel('CDF', fontproperties=source_sans)
        if log:
            ax.set_xlabel('Velocity + 1 (um/s)' if 'Pressure' not in title else 'Pressure (psi)', fontproperties=source_sans)
        else:
            ax.set_xlabel('Velocity (um/s)' if 'Pressure' not in title else 'Pressure (psi)', fontproperties=source_sans)
        ax.set_title(title, fontsize=8, fontproperties=source_sans)
    else:
        ax.set_ylabel('CDF')
        if log:
            ax.set_xlabel('Velocity + 1 (um/s)' if 'Pressure' not in title else 'Pressure (psi)')
        else:
            ax.set_xlabel('Velocity (um/s)' if 'Pressure' not in title else 'Pressure (psi)')
        ax.set_title(title, fontsize=8)

    if log:
        ax.set_xscale('log')
        # ax.set_xticklabels([1, 10, 100, 1000, 5000])
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    # Adjust legend with safe font handling
    if source_sans:
        ax.legend(loc='lower right', bbox_to_anchor=(1, 0.01), prop=source_sans, fontsize=6)
    else:
        ax.legend(loc='lower right', bbox_to_anchor=(1, 0.01), fontsize=6)
    
    ax.grid(True, linewidth=0.3)
    fig.set_dpi(300)
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    if write:
        save_plot(fig, title, dpi=300)
    else:
        plt.show()
    if write:
        plt.close()

    return 0


# ---------------------------------------------------------------------------
# Helper for legacy saving path inside plot_cdf_old
# ---------------------------------------------------------------------------

def save_plot(fig, title, dpi=300):
    """Save figure to the standard age-threshold results directory."""
    out_dir = os.path.join(PATHS['cap_flow'], 'results', 'cdf_analysis', 'age_thresholds')
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{title.replace(' ', '_')}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main() 