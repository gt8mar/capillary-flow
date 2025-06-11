"""
Filename: src/analysis/plot_participant20_cdf.py
---------------------------------------------------------

Plot the CDF (Cumulative Distribution Function) of participant 20 to explain 
how CDFs work, using the same data loading approach as confidence intervals.

By: Marcus Forst
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.config import PATHS, load_source_sans
from matplotlib.font_manager import FontProperties

# Define base colors (consistent with coding standards)
base_colors = {
    'default': '#1f77b4',      # Blue (entire dataset)
    'highlight': '#d62728'     # Red (participant 20)
}

def main():
    print("Creating CDF plot for participant 20...")
    
    # Load data same way as confidence intervals
    data_filepath = os.path.join(PATHS['cap_flow'], 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Data cleaning (same as figs_ci.py)
    df = df.dropna(subset=['Age'])
    df = df.dropna(subset=['Corrected Velocity'])
    df = df.drop_duplicates(subset=['Participant', 'Video'])
    
    print(f"Data loaded: {len(df)} rows, {len(df['Participant'].unique())} unique participants")
    
    # Filter for participant 20
    part20_data = df[df['Participant'] == 'part20']
    
    if part20_data.empty:
        print("Error: No data found for participant 20")
        return 1
    
    # Get participant 20's age
    part20_age = part20_data['Age'].iloc[0]
    print(f"Participant 20 age: {part20_age} years")
    
    # Setup plotting
    source_sans = load_source_sans()
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 6, 'lines.linewidth': 0.75
    })
    
    # Define control dataset (set01 healthy controls)
    controls_df = df[df['SET'] == 'set01']
    # Get velocity data
    part20_velocities = part20_data['Corrected Velocity'].dropna()
    control_velocities = controls_df['Corrected Velocity'].dropna()
    
    # Calculate CDFs
    part20_sorted = np.sort(part20_velocities)
    part20_cdf = np.arange(1, len(part20_sorted) + 1) / len(part20_sorted)
    
    control_sorted = np.sort(control_velocities)
    control_cdf = np.arange(1, len(control_sorted) + 1) / len(control_sorted)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    
    # Use standard colors from coding standards
    dataset_color = base_colors['default']  # Controls color
    part20_color = base_colors['highlight']
    
    # Line style: dashed to highlight participant 20
    part20_linestyle = '--'
    
    # Plot CDFs (controls vs participant20)
    ax.plot(control_sorted, control_cdf, color=dataset_color, alpha=0.8, linewidth=0.75,
            label='Controls')
    
    ax.plot(part20_sorted, part20_cdf, color=part20_color, linewidth=0.75,
            linestyle=part20_linestyle,
            label=f'Participant 20 (Age {part20_age:.0f})')
    
    # No shading under dataset CDF as requested
    
    # Add median lines
    part20_median = np.median(part20_velocities)
    control_median = np.median(control_velocities)
    
    ax.axvline(control_median, color=dataset_color, linestyle=':', alpha=0.8, linewidth=0.75)
    ax.axvline(part20_median, color=part20_color, linestyle=':', alpha=0.8, linewidth=1)
    ax.axhline(0.5, color='black', linestyle=':', alpha=0.3, linewidth=0.5)
    
    # Labels and formatting
    if source_sans:
        ax.set_xlabel('Velocity (μm/s)', fontproperties=source_sans)
        ax.set_ylabel('Cumulative Probability', fontproperties=source_sans)
        ax.set_title('CDF: Participant 20 vs Controls', fontproperties=source_sans, fontsize=8)
        ax.legend(prop=source_sans, loc='center right')
    else:
        ax.set_xlabel('Velocity (μm/s)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('CDF: Participant 20 vs Controls')
        ax.legend(loc='center right')
    
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(PATHS['cap_flow'], 'results', 'cdf_analysis', 'participant20')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = 'participant20_vs_controls_cdf.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    print(f"CDF plot saved: {filepath}")
    
    plt.show()
    
    # ------------------------------------------------------------
    # Distribution Plot (Histogram + KDE)
    # ------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(3.2, 2.4))

    # Histogram for controls dataset
    sns.histplot(control_velocities + 1, bins=30, color=dataset_color, alpha=0.4,
                 stat='density', kde=False, log_scale=True, ax=ax2, label='Controls')

    # KDE for controls dataset
    sns.kdeplot(control_velocities + 1, color=dataset_color, linewidth=0.75, ax=ax2)

    # Histogram for participant 20
    sns.histplot(part20_velocities + 1, bins=20, color=part20_color, alpha=0.4,
                 stat='density', kde=False, log_scale=True, ax=ax2, label='Participant 20')

    # KDE for participant 20
    sns.kdeplot(part20_velocities + 1, color=part20_color, linewidth=0.75, ax=ax2, linestyle='--')

    # Formatting
    if source_sans:
        ax2.set_xlabel('Velocity + 1 (μm/s)', fontproperties=source_sans)
        ax2.set_ylabel('Density', fontproperties=source_sans)
        ax2.set_title('Velocity Distribution: Participant 20 vs Controls', fontproperties=source_sans, fontsize=8)
        ax2.legend(prop=source_sans)
    else:
        ax2.set_xlabel('Velocity + 1 (μm/s)')
        ax2.set_ylabel('Density')
        ax2.set_title('Velocity Distribution: Participant 20 vs Controls')
        ax2.legend()

    ax2.grid(True, alpha=0.3, linewidth=0.3)
    plt.tight_layout()

    # Save distribution plot
    dist_filename = 'participant20_vs_controls_velocity_distribution.png'
    dist_filepath = os.path.join(output_dir, dist_filename)
    plt.savefig(dist_filepath, dpi=600, bbox_inches='tight')
    print(f"Distribution plot saved: {dist_filepath}")

    plt.show()
    
    # Print statistics
    print(f"\nSummary:")
    print(f"Participant 20: Age {part20_age:.0f}, Median velocity {part20_median:.1f} μm/s")
    print(f"Controls: Median velocity {control_median:.1f} μm/s")
    
    ks_stat, p_value = stats.ks_2samp(part20_velocities, control_velocities)
    print(f"KS test: D={ks_stat:.4f}, p={p_value:.4f}")
    
    # ------------------------------------------------------------
    # Participant-only plots (CDF and distribution)
    # ------------------------------------------------------------
    # CDF only
    fig_only, ax_only = plt.subplots(figsize=(3.2, 2.4))
    ax_only.plot(part20_sorted, part20_cdf, color=part20_color, linewidth=0.75, label='Participant 20')
    ax_only.axhline(0.5, color='black', linestyle=':', alpha=0.3, linewidth=0.5)
    ax_only.axvline(part20_median, color=part20_color, linestyle=':', alpha=0.8, linewidth=0.75)
    if source_sans:
        ax_only.set_xlabel('Velocity (μm/s)', fontproperties=source_sans)
        ax_only.set_ylabel('Cumulative Probability', fontproperties=source_sans)
        ax_only.set_title('CDF: Participant 20', fontproperties=source_sans, fontsize=8)
    else:
        ax_only.set_xlabel('Velocity (μm/s)')
        ax_only.set_ylabel('Cumulative Probability')
        ax_only.set_title('CDF: Participant 20')
    ax_only.set_ylim(0, 1)
    ax_only.grid(True, alpha=0.3, linewidth=0.3)
    plt.tight_layout()
    only_cdf_filename = 'participant20_cdf_only.png'
    plt.savefig(os.path.join(output_dir, only_cdf_filename), dpi=600, bbox_inches='tight')
    print(f"Participant-only CDF saved: {os.path.join(output_dir, only_cdf_filename)}")
    plt.show()

    # Distribution only
    fig_only2, ax_only2 = plt.subplots(figsize=(3.2, 2.4))
    sns.histplot(part20_velocities + 1, bins=20, color=part20_color, alpha=0.4,
                 stat='density', kde=False, log_scale=True, ax=ax_only2, label='Participant 20')
    sns.kdeplot(part20_velocities + 1, color=part20_color, linewidth=0.75, ax=ax_only2, linestyle='--')
    if source_sans:
        ax_only2.set_xlabel('Velocity + 1 (μm/s)', fontproperties=source_sans)
        ax_only2.set_ylabel('Density', fontproperties=source_sans)
        ax_only2.set_title('Velocity Distribution: Participant 20', fontproperties=source_sans, fontsize=8)
    else:
        ax_only2.set_xlabel('Velocity + 1 (μm/s)')
        ax_only2.set_ylabel('Density')
        ax_only2.set_title('Velocity Distribution: Participant 20')
    ax_only2.grid(True, alpha=0.3, linewidth=0.3)
    plt.tight_layout()
    only_dist_filename = 'participant20_velocity_distribution_only.png'
    plt.savefig(os.path.join(output_dir, only_dist_filename), dpi=600, bbox_inches='tight')
    print(f"Participant-only distribution saved: {os.path.join(output_dir, only_dist_filename)}")
    plt.show()
    
    # ------------------------------------------------------------
    # Linear-scale distribution plots (controls vs participant 20)
    # ------------------------------------------------------------
    fig_lin, ax_lin = plt.subplots(figsize=(3.2, 2.4))

    sns.histplot(control_velocities, bins=30, color=dataset_color, alpha=0.4,
                 stat='density', kde=False, ax=ax_lin, label='Controls')
    sns.kdeplot(control_velocities, color=dataset_color, linewidth=0.75, ax=ax_lin)

    sns.histplot(part20_velocities, bins=20, color=part20_color, alpha=0.4,
                 stat='density', kde=False, ax=ax_lin, label='Participant 20')
    sns.kdeplot(part20_velocities, color=part20_color, linewidth=0.75, ax=ax_lin, linestyle='--')

    if source_sans:
        ax_lin.set_xlabel('Velocity (μm/s)', fontproperties=source_sans)
        ax_lin.set_ylabel('Density', fontproperties=source_sans)
        ax_lin.set_title('Velocity Distribution (Linear): Participant 20 vs Controls', fontproperties=source_sans, fontsize=8)
        ax_lin.legend(prop=source_sans)
    else:
        ax_lin.set_xlabel('Velocity (μm/s)')
        ax_lin.set_ylabel('Density')
        ax_lin.set_title('Velocity Distribution (Linear): Participant 20 vs Controls')
        ax_lin.legend()

    ax_lin.grid(True, alpha=0.3, linewidth=0.3)
    plt.tight_layout()

    dist_lin_filename = 'participant20_vs_controls_velocity_distribution_linear.png'
    plt.savefig(os.path.join(output_dir, dist_lin_filename), dpi=600, bbox_inches='tight')
    print(f"Linear-scale distribution plot saved: {os.path.join(output_dir, dist_lin_filename)}")
    plt.show()

    # Participant-only linear distribution
    fig_only_lin, ax_only_lin = plt.subplots(figsize=(3.2, 2.4))
    sns.histplot(part20_velocities, bins=20, color=part20_color, alpha=0.4,
                 stat='density', kde=False, ax=ax_only_lin, label='Participant 20')
    sns.kdeplot(part20_velocities, color=part20_color, linewidth=0.75, ax=ax_only_lin, linestyle='--')

    if source_sans:
        ax_only_lin.set_xlabel('Velocity (μm/s)', fontproperties=source_sans)
        ax_only_lin.set_ylabel('Density', fontproperties=source_sans)
        ax_only_lin.set_title('Velocity Distribution (Linear): Participant 20', fontproperties=source_sans, fontsize=8)
        ax_only_lin.legend(prop=source_sans)
    else:
        ax_only_lin.set_xlabel('Velocity (μm/s)')
        ax_only_lin.set_ylabel('Density')
        ax_only_lin.set_title('Velocity Distribution (Linear): Participant 20')
        ax_only_lin.legend()

    ax_only_lin.grid(True, alpha=0.3, linewidth=0.3)
    plt.tight_layout()

    only_dist_lin_filename = 'participant20_velocity_distribution_only_linear.png'
    plt.savefig(os.path.join(output_dir, only_dist_lin_filename), dpi=600, bbox_inches='tight')
    print(f"Participant-only linear distribution saved: {os.path.join(output_dir, only_dist_lin_filename)}")
    plt.show()
    
    # ------------------------------------------------------------
    # Additional CDF plots WITHOUT median lines (requested)
    # ------------------------------------------------------------

    # Controls vs Participant 20 without medians
    fig_nom, ax_nom = plt.subplots(figsize=(3.2, 2.4))
    ax_nom.plot(control_sorted, control_cdf, color=dataset_color, alpha=0.8, linewidth=0.75, label='Controls')
    ax_nom.plot(part20_sorted, part20_cdf, color=part20_color, linewidth=0.75, linestyle=part20_linestyle, label=f'Participant 20 (Age {part20_age:.0f})')
    ax_nom.axhline(0.5, color='black', linestyle=':', alpha=0.3, linewidth=0.5)
    if source_sans:
        ax_nom.set_xlabel('Velocity (μm/s)', fontproperties=source_sans)
        ax_nom.set_ylabel('Cumulative Probability', fontproperties=source_sans)
        ax_nom.set_title('CDF: Participant 20 vs Controls (No Medians)', fontproperties=source_sans, fontsize=8)
        ax_nom.legend(prop=source_sans, loc='center right')
    else:
        ax_nom.set_xlabel('Velocity (μm/s)')
        ax_nom.set_ylabel('Cumulative Probability')
        ax_nom.set_title('CDF: Participant 20 vs Controls (No Medians)')
        ax_nom.legend(loc='center right')
    ax_nom.set_ylim(0, 1)
    ax_nom.grid(True, alpha=0.3, linewidth=0.3)
    plt.tight_layout()
    no_med_filename = 'participant20_vs_controls_cdf_no_medians.png'
    plt.savefig(os.path.join(output_dir, no_med_filename), dpi=600, bbox_inches='tight')
    print(f"CDF without medians saved: {os.path.join(output_dir, no_med_filename)}")
    plt.show()

    # Participant-only CDF without median line
    fig_only_nom, ax_only_nom = plt.subplots(figsize=(3.2, 2.4))
    ax_only_nom.plot(part20_sorted, part20_cdf, color=part20_color, linewidth=0.75, label='Participant 20')
    ax_only_nom.axhline(0.5, color='black', linestyle=':', alpha=0.3, linewidth=0.5)
    if source_sans:
        ax_only_nom.set_xlabel('Velocity (μm/s)', fontproperties=source_sans)
        ax_only_nom.set_ylabel('Cumulative Probability', fontproperties=source_sans)
        ax_only_nom.set_title('CDF: Participant 20 (No Median)', fontproperties=source_sans, fontsize=8)
    else:
        ax_only_nom.set_xlabel('Velocity (μm/s)')
        ax_only_nom.set_ylabel('Cumulative Probability')
        ax_only_nom.set_title('CDF: Participant 20 (No Median)')
    ax_only_nom.set_ylim(0, 1)
    ax_only_nom.grid(True, alpha=0.3, linewidth=0.3)
    plt.tight_layout()
    only_nom_filename = 'participant20_cdf_only_no_median.png'
    plt.savefig(os.path.join(output_dir, only_nom_filename), dpi=600, bbox_inches='tight')
    print(f"Participant-only CDF without median saved: {os.path.join(output_dir, only_nom_filename)}")
    plt.show()
    
    return 0

if __name__ == '__main__':
    main() 