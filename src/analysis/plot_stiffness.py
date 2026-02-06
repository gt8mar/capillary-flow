"""
Filename: src/analysis/plot_stiffness.py

File for creating paper-worthy plots comparing stiffness coefficients
to blood pressure and other demographic variables.
By: Marcus Forst
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from typing import Optional, Tuple
import warnings

# Import paths and font from config
from src.config import PATHS, load_source_sans

cap_flow_path = PATHS['cap_flow']
source_sans = load_source_sans()

# Set up publication-quality plotting style
plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5,
    'lines.linewidth': 0.5,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.3
})

sns.set_style("whitegrid")

# Color scheme
CONTROL_COLOR = '#1f77b4'  # Blue
DIABETES_COLOR = '#ff7f0e'  # Orange


def apply_font(ax, source_sans):
    """Apply Source Sans font to axis labels and title if available."""
    if source_sans:
        for label in ax.get_xticklabels():
            label.set_fontproperties(source_sans)
        for label in ax.get_yticklabels():
            label.set_fontproperties(source_sans)
        ax.xaxis.label.set_fontproperties(source_sans)
        ax.yaxis.label.set_fontproperties(source_sans)
        if ax.get_title():
            ax.title.set_fontproperties(source_sans)


def save_figure(fig, output_dir: str, filename: str, dpi: int = 300):
    """Save figure as both PDF and PNG.
    
    Args:
        fig: matplotlib figure object
        output_dir: Directory to save files
        filename: Base filename (without extension)
        dpi: Resolution for PNG (default: 300)
    """
    # Save PDF
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight')
    
    # Save PNG in png subfolder
    png_dir = os.path.join(output_dir, 'png')
    os.makedirs(png_dir, exist_ok=True)
    png_path = os.path.join(png_dir, f'{filename}.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def plot_map_by_group(results_df: pd.DataFrame, output_dir: str, figsize: Tuple[float, float] = (2.4, 2.0)):
    """Plot boxplot of MAP by group (Control vs Diabetic).
    
    Panel A: Boxplot showing no difference in MAP between groups.
    """
    # Filter to participants with MAP data
    df_plot = results_df[results_df['MAP'].notna()].copy()
    
    # Create group labels
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create boxplot
    box_colors = [CONTROL_COLOR, DIABETES_COLOR]
    sns.boxplot(data=df_plot, x='Group', y='MAP', ax=ax, palette=box_colors, width=0.6)
    
    # Add individual points
    for i, group in enumerate(['Control', 'Diabetic']):
        group_data = df_plot[df_plot['Group'] == group]['MAP']
        x_pos = i + np.random.normal(0, 0.05, len(group_data))
        ax.scatter(x_pos, group_data, alpha=0.4, s=10, color='gray', zorder=3)
    
    ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Mean Arterial Pressure (mmHg)', fontproperties=source_sans if source_sans else None)
    ax.set_title('MAP by Group', fontproperties=source_sans if source_sans else None)
    
    # Statistical test
    control_map = df_plot[df_plot['Group'] == 'Control']['MAP']
    diabetic_map = df_plot[df_plot['Group'] == 'Diabetic']['MAP']
    if len(control_map) > 0 and len(diabetic_map) > 0:
        stat, pval = stats.mannwhitneyu(control_map, diabetic_map, alternative='two-sided')
        ax.text(0.5, 0.95, f'p = {pval:.3f}', transform=ax.transAxes,
                ha='center', va='top', fontsize=6, fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    save_figure(fig, output_dir, 'stiffness_fig_MAP_by_group')
    plt.close()


def plot_sbp_by_group(results_df: pd.DataFrame, output_dir: str, figsize: Tuple[float, float] = (2.4, 2.0)):
    """Plot boxplot of SBP by group (Control vs Diabetic).
    
    Panel B: Boxplot showing no difference in SBP between groups.
    """
    # Filter to participants with SBP data
    df_plot = results_df[results_df['SYS_BP'].notna()].copy()
    
    # Create group labels
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create boxplot
    box_colors = [CONTROL_COLOR, DIABETES_COLOR]
    sns.boxplot(data=df_plot, x='Group', y='SYS_BP', ax=ax, palette=box_colors, width=0.6)
    
    # Add individual points
    for i, group in enumerate(['Control', 'Diabetic']):
        group_data = df_plot[df_plot['Group'] == group]['SYS_BP']
        x_pos = i + np.random.normal(0, 0.05, len(group_data))
        ax.scatter(x_pos, group_data, alpha=0.4, s=10, color='gray', zorder=3)
    
    ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Systolic Blood Pressure (mmHg)', fontproperties=source_sans if source_sans else None)
    ax.set_title('SBP by Group', fontproperties=source_sans if source_sans else None)
    
    # Statistical test
    control_sbp = df_plot[df_plot['Group'] == 'Control']['SYS_BP']
    diabetic_sbp = df_plot[df_plot['Group'] == 'Diabetic']['SYS_BP']
    if len(control_sbp) > 0 and len(diabetic_sbp) > 0:
        stat, pval = stats.mannwhitneyu(control_sbp, diabetic_sbp, alternative='two-sided')
        ax.text(0.5, 0.95, f'p = {pval:.3f}', transform=ax.transAxes,
                ha='center', va='top', fontsize=6, fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    save_figure(fig, output_dir, 'stiffness_fig_SBP_by_group')
    plt.close()


def plot_stiffness_by_group(results_df: pd.DataFrame, output_dir: str, 
                           stiffness_col: str = 'stiffness_coeff_averaged_04_12',
                           figsize: Tuple[float, float] = (2.4, 2.0)):
    """Plot boxplot of stiffness coefficient by group.
    
    Panel C: Boxplot showing clear difference in stiffness between groups.
    """
    # Filter to participants with stiffness data
    df_plot = results_df[results_df[stiffness_col].notna()].copy()
    
    # Create group labels
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create boxplot
    box_colors = [CONTROL_COLOR, DIABETES_COLOR]
    sns.boxplot(data=df_plot, x='Group', y=stiffness_col, ax=ax, palette=box_colors, width=0.6)
    
    # Add individual points
    for i, group in enumerate(['Control', 'Diabetic']):
        group_data = df_plot[df_plot['Group'] == group][stiffness_col]
        x_pos = i + np.random.normal(0, 0.05, len(group_data))
        ax.scatter(x_pos, group_data, alpha=0.4, s=10, color='gray', zorder=3)
    
    ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Stiffness Index (SI_AUC)', fontproperties=source_sans if source_sans else None)
    ax.set_title('Stiffness Index by Group', fontproperties=source_sans if source_sans else None)
    
    # Statistical test
    control_stiff = df_plot[df_plot['Group'] == 'Control'][stiffness_col]
    diabetic_stiff = df_plot[df_plot['Group'] == 'Diabetic'][stiffness_col]
    if len(control_stiff) > 0 and len(diabetic_stiff) > 0:
        stat, pval = stats.mannwhitneyu(control_stiff, diabetic_stiff, alternative='two-sided')
        ax.text(0.5, 0.95, f'p = {pval:.4f}', transform=ax.transAxes,
                ha='center', va='top', fontsize=6, fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    # Determine range label from column name
    if '02_12' in stiffness_col:
        range_label = '0.2-1.2'
    else:
        range_label = '0.4-1.2'
    
    if 'up' in stiffness_col:
        method_label = 'up'
    else:
        method_label = 'averaged'
    
    filename = f'stiffness_fig_SI_by_group_{method_label}_{range_label}'
    save_figure(fig, output_dir, filename)
    plt.close()


def plot_stiffness_vs_age(results_df: pd.DataFrame, output_dir: str,
                          stiffness_col: str = 'stiffness_coeff_averaged_04_12',
                          figsize: Tuple[float, float] = (2.4, 2.0)):
    """Plot stiffness coefficient vs age with regression line.
    
    Shows that age explains variance in stiffness.
    """
    # Filter to participants with both age and stiffness data
    df_plot = results_df[(results_df['Age'].notna()) & (results_df[stiffness_col].notna())].copy()
    
    # Create group labels
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points by group
    for group, color in [('Control', CONTROL_COLOR), ('Diabetic', DIABETES_COLOR)]:
        group_data = df_plot[df_plot['Group'] == group]
        ax.scatter(group_data['Age'], group_data[stiffness_col], 
                  alpha=0.6, s=20, color=color, label=group, zorder=3)
    
    # Fit regression line
    x = df_plot['Age'].values
    y = df_plot[stiffness_col].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7, label=f'R² = {r_value**2:.3f}')
        
        # Add correlation text
        ax.text(0.05, 0.95, f'R = {r_value:.3f}\np = {p_value:.4f}', 
                transform=ax.transAxes, ha='left', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Age (years)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Stiffness Index (SI_AUC)', fontproperties=source_sans if source_sans else None)
    ax.set_title('Stiffness Index vs Age', fontproperties=source_sans if source_sans else None)
    ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    # Determine range label from column name
    if '02_12' in stiffness_col:
        range_label = '0.2-1.2'
    else:
        range_label = '0.4-1.2'
    
    if 'up' in stiffness_col:
        method_label = 'up'
    else:
        method_label = 'averaged'
    
    filename = f'stiffness_fig_SI_vs_age_{method_label}_{range_label}'
    save_figure(fig, output_dir, filename)
    plt.close()


def plot_stiffness_vs_map(results_df: pd.DataFrame, output_dir: str,
                          stiffness_col: str = 'stiffness_coeff_averaged_04_12',
                          figsize: Tuple[float, float] = (2.4, 2.0)):
    """Plot stiffness coefficient vs MAP with regression line.
    
    Shows that BP does NOT explain variance in stiffness (slope ~ 0).
    """
    # Filter to participants with both MAP and stiffness data
    df_plot = results_df[(results_df['MAP'].notna()) & (results_df[stiffness_col].notna())].copy()
    
    # Create group labels
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points by group
    for group, color in [('Control', CONTROL_COLOR), ('Diabetic', DIABETES_COLOR)]:
        group_data = df_plot[df_plot['Group'] == group]
        ax.scatter(group_data['MAP'], group_data[stiffness_col], 
                  alpha=0.6, s=20, color=color, label=group, zorder=3)
    
    # Fit regression line
    x = df_plot['MAP'].values
    y = df_plot[stiffness_col].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7, label=f'R² = {r_value**2:.3f}')
        
        # Add correlation text
        ax.text(0.05, 0.95, f'R = {r_value:.3f}\np = {p_value:.4f}\nslope = {slope:.4f}', 
                transform=ax.transAxes, ha='left', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Mean Arterial Pressure (mmHg)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Stiffness Index (SI_AUC)', fontproperties=source_sans if source_sans else None)
    ax.set_title('Stiffness Index vs MAP', fontproperties=source_sans if source_sans else None)
    ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    # Determine range label from column name
    if '02_12' in stiffness_col:
        range_label = '0.2-1.2'
    else:
        range_label = '0.4-1.2'
    
    if 'up' in stiffness_col:
        method_label = 'up'
    else:
        method_label = 'averaged'
    
    filename = f'stiffness_fig_SI_vs_MAP_{method_label}_{range_label}'
    save_figure(fig, output_dir, filename)
    plt.close()


def plot_stiffness_vs_sbp(results_df: pd.DataFrame, output_dir: str,
                          stiffness_col: str = 'stiffness_coeff_averaged_04_12',
                          figsize: Tuple[float, float] = (2.4, 2.0)):
    """Plot stiffness coefficient vs SBP with regression line.
    
    Shows that BP does NOT explain variance in stiffness (slope ~ 0).
    """
    # Filter to participants with both SBP and stiffness data
    df_plot = results_df[(results_df['SYS_BP'].notna()) & (results_df[stiffness_col].notna())].copy()
    
    # Create group labels
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points by group
    for group, color in [('Control', CONTROL_COLOR), ('Diabetic', DIABETES_COLOR)]:
        group_data = df_plot[df_plot['Group'] == group]
        ax.scatter(group_data['SYS_BP'], group_data[stiffness_col], 
                  alpha=0.6, s=20, color=color, label=group, zorder=3)
    
    # Fit regression line
    x = df_plot['SYS_BP'].values
    y = df_plot[stiffness_col].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7, label=f'R² = {r_value**2:.3f}')
        
        # Add correlation text
        ax.text(0.05, 0.95, f'R = {r_value:.3f}\np = {p_value:.4f}\nslope = {slope:.4f}', 
                transform=ax.transAxes, ha='left', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Systolic Blood Pressure (mmHg)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Stiffness Index (SI_AUC)', fontproperties=source_sans if source_sans else None)
    ax.set_title('Stiffness Index vs SBP', fontproperties=source_sans if source_sans else None)
    ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    # Determine range label from column name
    if '02_12' in stiffness_col:
        range_label = '0.2-1.2'
    else:
        range_label = '0.4-1.2'
    
    if 'up' in stiffness_col:
        method_label = 'up'
    else:
        method_label = 'averaged'
    
    filename = f'stiffness_fig_SI_vs_SBP_{method_label}_{range_label}'
    save_figure(fig, output_dir, filename)
    plt.close()


def compare_up_vs_averaged(results_df: pd.DataFrame, output_dir: str,
                           range_label: str = '04_12',
                           figsize: Tuple[float, float] = (2.4, 2.0)):
    """Compare up-only vs averaged stiffness coefficients.
    
    Scatter plot showing correlation between the two methods.
    """
    up_col = f'stiffness_coeff_up_{range_label}'
    avg_col = f'stiffness_coeff_averaged_{range_label}'
    
    # Filter to participants with both metrics
    df_plot = results_df[(results_df[up_col].notna()) & (results_df[avg_col].notna())].copy()
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points by group
    for group, color in [('Control', CONTROL_COLOR), ('Diabetic', DIABETES_COLOR)]:
        group_data = df_plot[df_plot['Group'] == group]
        ax.scatter(group_data[up_col], group_data[avg_col], 
                  alpha=0.6, s=20, color=color, label=group, zorder=3)
    
    # Fit regression line
    x = df_plot[up_col].values
    y = df_plot[avg_col].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7)
        
        # Add 1:1 line
        min_val = min(x[valid].min(), y[valid].min())
        max_val = max(x[valid].max(), y[valid].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r:', linewidth=0.5, alpha=0.5, label='1:1')
        
        # Add correlation text
        ax.text(0.05, 0.95, f'R = {r_value:.3f}\np = {p_value:.4f}', 
                transform=ax.transAxes, ha='left', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('SI_AUC (Up Only)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('SI_AUC (Averaged)', fontproperties=source_sans if source_sans else None)
    ax.set_title('Up vs Averaged Stiffness', fontproperties=source_sans if source_sans else None)
    ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    filename = f'stiffness_fig_up_vs_averaged_{range_label}'
    save_figure(fig, output_dir, filename)
    plt.close()


def compare_ranges(results_df: pd.DataFrame, output_dir: str,
                  method: str = 'averaged',
                  figsize: Tuple[float, float] = (2.4, 2.0)):
    """Compare 0.2-1.2 vs 0.4-1.2 stiffness coefficients.
    
    Scatter plot showing correlation between the two ranges.
    """
    col_02 = f'stiffness_coeff_{method}_02_12'
    col_04 = f'stiffness_coeff_{method}_04_12'
    
    # Filter to participants with both metrics
    df_plot = results_df[(results_df[col_02].notna()) & (results_df[col_04].notna())].copy()
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points by group
    for group, color in [('Control', CONTROL_COLOR), ('Diabetic', DIABETES_COLOR)]:
        group_data = df_plot[df_plot['Group'] == group]
        ax.scatter(group_data[col_04], group_data[col_02], 
                  alpha=0.6, s=20, color=color, label=group, zorder=3)
    
    # Fit regression line
    x = df_plot[col_04].values
    y = df_plot[col_02].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7)
        
        # Add correlation text
        ax.text(0.05, 0.95, f'R = {r_value:.3f}\np = {p_value:.4f}', 
                transform=ax.transAxes, ha='left', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('SI_AUC (0.4-1.2 psi)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('SI_AUC (0.2-1.2 psi)', fontproperties=source_sans if source_sans else None)
    ax.set_title(f'Range Comparison ({method})', fontproperties=source_sans if source_sans else None)
    ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    filename = f'stiffness_fig_range_comparison_{method}'
    save_figure(fig, output_dir, filename)
    plt.close()


def plot_p50_vs_stiffness(results_df: pd.DataFrame, output_dir: str,
                          stiffness_col: str = 'stiffness_coeff_averaged_04_12',
                          figsize: Tuple[float, float] = (2.4, 2.0)):
    """Plot P50 vs stiffness coefficient to show they tell the same story.
    
    Supplement figure showing P50 as a secondary mechanical metric.
    """
    # Filter to participants with both metrics
    df_plot = results_df[(results_df['P50'].notna()) & (results_df[stiffness_col].notna())].copy()
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points by group
    for group, color in [('Control', CONTROL_COLOR), ('Diabetic', DIABETES_COLOR)]:
        group_data = df_plot[df_plot['Group'] == group]
        ax.scatter(group_data['P50'], group_data[stiffness_col], 
                  alpha=0.6, s=20, color=color, label=group, zorder=3)
    
    # Fit regression line
    x = df_plot['P50'].values
    y = df_plot[stiffness_col].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7)
        
        # Add correlation text
        ax.text(0.05, 0.95, f'R = {r_value:.3f}\np = {p_value:.4f}', 
                transform=ax.transAxes, ha='left', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('P50 (psi)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Stiffness Index (SI_AUC)', fontproperties=source_sans if source_sans else None)
    ax.set_title('P50 vs SI_AUC', fontproperties=source_sans if source_sans else None)
    ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    save_figure(fig, output_dir, 'stiffness_fig_P50_vs_SI')
    plt.close()


def plot_ev_lin_vs_stiffness(results_df: pd.DataFrame, output_dir: str,
                             stiffness_col: str = 'stiffness_coeff_averaged_04_12',
                             figsize: Tuple[float, float] = (2.4, 2.0)):
    """Plot EV_lin vs stiffness coefficient to show they tell the same story.
    
    Supplement figure showing EV_lin as a secondary mechanical metric.
    """
    # Filter to participants with both metrics
    df_plot = results_df[(results_df['EV_lin'].notna()) & (results_df[stiffness_col].notna())].copy()
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points by group
    for group, color in [('Control', CONTROL_COLOR), ('Diabetic', DIABETES_COLOR)]:
        group_data = df_plot[df_plot['Group'] == group]
        ax.scatter(group_data['EV_lin'], group_data[stiffness_col], 
                  alpha=0.6, s=20, color=color, label=group, zorder=3)
    
    # Fit regression line
    x = df_plot['EV_lin'].values
    y = df_plot[stiffness_col].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7)
        
        # Add correlation text
        ax.text(0.05, 0.95, f'R = {r_value:.3f}\np = {p_value:.4f}', 
                transform=ax.transAxes, ha='left', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('EV_lin ((um/s)/psi)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Stiffness Index (SI_AUC)', fontproperties=source_sans if source_sans else None)
    ax.set_title('EV_lin vs SI_AUC', fontproperties=source_sans if source_sans else None)
    ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    save_figure(fig, output_dir, 'stiffness_fig_EV_lin_vs_SI')
    plt.close()


def plot_p50_by_group(results_df: pd.DataFrame, output_dir: str,
                      figsize: Tuple[float, float] = (2.4, 2.0)):
    """Plot boxplot of P50 by group.
    
    Supplement figure showing P50 separation between groups.
    """
    # Filter to participants with P50 data
    df_plot = results_df[results_df['P50'].notna()].copy()
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create boxplot
    box_colors = [CONTROL_COLOR, DIABETES_COLOR]
    sns.boxplot(data=df_plot, x='Group', y='P50', ax=ax, palette=box_colors, width=0.6)
    
    # Add individual points
    for i, group in enumerate(['Control', 'Diabetic']):
        group_data = df_plot[df_plot['Group'] == group]['P50']
        x_pos = i + np.random.normal(0, 0.05, len(group_data))
        ax.scatter(x_pos, group_data, alpha=0.4, s=10, color='gray', zorder=3)
    
    ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('P50 (psi)', fontproperties=source_sans if source_sans else None)
    ax.set_title('P50 by Group', fontproperties=source_sans if source_sans else None)
    
    # Statistical test
    control_p50 = df_plot[df_plot['Group'] == 'Control']['P50']
    diabetic_p50 = df_plot[df_plot['Group'] == 'Diabetic']['P50']
    if len(control_p50) > 0 and len(diabetic_p50) > 0:
        stat, pval = stats.mannwhitneyu(control_p50, diabetic_p50, alternative='two-sided')
        ax.text(0.5, 0.95, f'p = {pval:.4f}', transform=ax.transAxes,
                ha='center', va='top', fontsize=6, fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    save_figure(fig, output_dir, 'stiffness_fig_P50_by_group')
    plt.close()


def plot_ev_lin_by_group(results_df: pd.DataFrame, output_dir: str,
                         figsize: Tuple[float, float] = (2.4, 2.0)):
    """Plot boxplot of EV_lin by group.
    
    Supplement figure showing EV_lin separation between groups.
    """
    # Filter to participants with EV_lin data
    df_plot = results_df[results_df['EV_lin'].notna()].copy()
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create boxplot
    box_colors = [CONTROL_COLOR, DIABETES_COLOR]
    sns.boxplot(data=df_plot, x='Group', y='EV_lin', ax=ax, palette=box_colors, width=0.6)
    
    # Add individual points
    for i, group in enumerate(['Control', 'Diabetic']):
        group_data = df_plot[df_plot['Group'] == group]['EV_lin']
        x_pos = i + np.random.normal(0, 0.05, len(group_data))
        ax.scatter(x_pos, group_data, alpha=0.4, s=10, color='gray', zorder=3)
    
    ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('EV_lin ((um/s)/psi)', fontproperties=source_sans if source_sans else None)
    ax.set_title('EV_lin by Group', fontproperties=source_sans if source_sans else None)
    
    # Statistical test
    control_ev = df_plot[df_plot['Group'] == 'Control']['EV_lin']
    diabetic_ev = df_plot[df_plot['Group'] == 'Diabetic']['EV_lin']
    if len(control_ev) > 0 and len(diabetic_ev) > 0:
        stat, pval = stats.mannwhitneyu(control_ev, diabetic_ev, alternative='two-sided')
        ax.text(0.5, 0.95, f'p = {pval:.4f}', transform=ax.transAxes,
                ha='center', va='top', fontsize=6, fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    save_figure(fig, output_dir, 'stiffness_fig_EV_lin_by_group')
    plt.close()


def calculate_correlations(results_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations for robustness checks.
    
    Returns DataFrame with correlation coefficients and p-values for:
    - Up vs Averaged
    - 0.2-1.2 vs 0.4-1.2
    - Correlation with age
    - Correlation with MAP
    - Correlation with SBP
    - Separation of diabetics vs controls
    """
    correlations = []
    
    # Compare up vs averaged
    for range_label in ['04_12', '02_12']:
        up_col = f'stiffness_coeff_up_{range_label}'
        avg_col = f'stiffness_coeff_averaged_{range_label}'
        df_valid = results_df[(results_df[up_col].notna()) & (results_df[avg_col].notna())]
        if len(df_valid) > 2:
            r, p = pearsonr(df_valid[up_col], df_valid[avg_col])
            correlations.append({
                'Comparison': f'Up vs Averaged ({range_label})',
                'R': r,
                'p_value': p,
                'N': len(df_valid)
            })
    
    # Compare ranges
    for method in ['up', 'averaged']:
        col_02 = f'stiffness_coeff_{method}_02_12'
        col_04 = f'stiffness_coeff_{method}_04_12'
        df_valid = results_df[(results_df[col_02].notna()) & (results_df[col_04].notna())]
        if len(df_valid) > 2:
            r, p = pearsonr(df_valid[col_02], df_valid[col_04])
            correlations.append({
                'Comparison': f'0.2-1.2 vs 0.4-1.2 ({method})',
                'R': r,
                'p_value': p,
                'N': len(df_valid)
            })
    
    # Correlation with age
    for col in ['stiffness_coeff_up_04_12', 'stiffness_coeff_up_02_12',
                'stiffness_coeff_averaged_04_12', 'stiffness_coeff_averaged_02_12']:
        df_valid = results_df[(results_df['Age'].notna()) & (results_df[col].notna())]
        if len(df_valid) > 2:
            r, p = pearsonr(df_valid['Age'], df_valid[col])
            correlations.append({
                'Comparison': f'{col} vs Age',
                'R': r,
                'p_value': p,
                'N': len(df_valid)
            })
    
    # Correlation with MAP
    for col in ['stiffness_coeff_up_04_12', 'stiffness_coeff_up_02_12',
                'stiffness_coeff_averaged_04_12', 'stiffness_coeff_averaged_02_12']:
        df_valid = results_df[(results_df['MAP'].notna()) & (results_df[col].notna())]
        if len(df_valid) > 2:
            r, p = pearsonr(df_valid['MAP'], df_valid[col])
            correlations.append({
                'Comparison': f'{col} vs MAP',
                'R': r,
                'p_value': p,
                'N': len(df_valid)
            })
    
    # Correlation with SBP
    for col in ['stiffness_coeff_up_04_12', 'stiffness_coeff_up_02_12',
                'stiffness_coeff_averaged_04_12', 'stiffness_coeff_averaged_02_12']:
        df_valid = results_df[(results_df['SYS_BP'].notna()) & (results_df[col].notna())]
        if len(df_valid) > 2:
            r, p = pearsonr(df_valid['SYS_BP'], df_valid[col])
            correlations.append({
                'Comparison': f'{col} vs SBP',
                'R': r,
                'p_value': p,
                'N': len(df_valid)
            })
    
    # Separation of diabetics vs controls (Mann-Whitney U test)
    for col in ['stiffness_coeff_up_04_12', 'stiffness_coeff_up_02_12',
                'stiffness_coeff_averaged_04_12', 'stiffness_coeff_averaged_02_12']:
        control = results_df[(results_df['Diabetes'] == False) & (results_df[col].notna())][col]
        diabetic = results_df[(results_df['Diabetes'] == True) & (results_df[col].notna())][col]
        if len(control) > 0 and len(diabetic) > 0:
            stat, p = stats.mannwhitneyu(control, diabetic, alternative='two-sided')
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control) - 1) * control.std()**2 + 
                                 (len(diabetic) - 1) * diabetic.std()**2) / 
                                (len(control) + len(diabetic) - 2))
            cohens_d = (diabetic.mean() - control.mean()) / pooled_std if pooled_std > 0 else 0
            correlations.append({
                'Comparison': f'{col} (Diabetic vs Control)',
                'R': cohens_d,  # Using Cohen's d as effect size
                'p_value': p,
                'N': len(control) + len(diabetic)
            })
    
    return pd.DataFrame(correlations)


def plot_composite_stiffness_by_group(results_df: pd.DataFrame, output_dir: str,
                                     figsize: Tuple[float, float] = (2.4, 2.0)):
    """Plot boxplot of composite stiffness by group.
    
    Shows composite stiffness score using classifier weights.
    """
    if 'composite_stiffness' not in results_df.columns:
        print("Warning: composite_stiffness not found in results")
        return
    
    df_plot = results_df[results_df['composite_stiffness'].notna()].copy()
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    box_colors = [CONTROL_COLOR, DIABETES_COLOR]
    sns.boxplot(data=df_plot, x='Group', y='composite_stiffness', ax=ax, palette=box_colors, width=0.6)
    
    for i, group in enumerate(['Control', 'Diabetic']):
        group_data = df_plot[df_plot['Group'] == group]['composite_stiffness']
        x_pos = i + np.random.normal(0, 0.05, len(group_data))
        ax.scatter(x_pos, group_data, alpha=0.4, s=10, color='gray', zorder=3)
    
    ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Composite Stiffness Score', fontproperties=source_sans if source_sans else None)
    ax.set_title('Composite Stiffness by Group', fontproperties=source_sans if source_sans else None)
    
    control_stiff = df_plot[df_plot['Group'] == 'Control']['composite_stiffness']
    diabetic_stiff = df_plot[df_plot['Group'] == 'Diabetic']['composite_stiffness']
    if len(control_stiff) > 0 and len(diabetic_stiff) > 0:
        stat, pval = stats.mannwhitneyu(control_stiff, diabetic_stiff, alternative='two-sided')
        ax.text(0.5, 0.95, f'p = {pval:.4f}', transform=ax.transAxes,
                ha='center', va='top', fontsize=6, fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    save_figure(fig, output_dir, 'stiffness_fig_composite_by_group')
    plt.close()


def plot_log_stiffness_by_group(results_df: pd.DataFrame, output_dir: str,
                                stiffness_col: str = 'log_stiffness_coeff_averaged_04_12',
                                figsize: Tuple[float, float] = (2.4, 2.0)):
    """Plot boxplot of log-transformed stiffness by group."""
    if stiffness_col not in results_df.columns:
        return
    
    df_plot = results_df[results_df[stiffness_col].notna()].copy()
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    box_colors = [CONTROL_COLOR, DIABETES_COLOR]
    sns.boxplot(data=df_plot, x='Group', y=stiffness_col, ax=ax, palette=box_colors, width=0.6)
    
    for i, group in enumerate(['Control', 'Diabetic']):
        group_data = df_plot[df_plot['Group'] == group][stiffness_col]
        x_pos = i + np.random.normal(0, 0.05, len(group_data))
        ax.scatter(x_pos, group_data, alpha=0.4, s=10, color='gray', zorder=3)
    
    ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('log(SI_AUC + 1)', fontproperties=source_sans if source_sans else None)
    ax.set_title('Log-Transformed Stiffness by Group', fontproperties=source_sans if source_sans else None)
    
    control_stiff = df_plot[df_plot['Group'] == 'Control'][stiffness_col]
    diabetic_stiff = df_plot[df_plot['Group'] == 'Diabetic'][stiffness_col]
    if len(control_stiff) > 0 and len(diabetic_stiff) > 0:
        stat, pval = stats.mannwhitneyu(control_stiff, diabetic_stiff, alternative='two-sided')
        ax.text(0.5, 0.95, f'p = {pval:.4f}', transform=ax.transAxes,
                ha='center', va='top', fontsize=6, fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    # Determine range label from column name
    if '02_12' in stiffness_col:
        range_label = '0.2-1.2'
    else:
        range_label = '0.4-1.2'
    
    if 'up' in stiffness_col:
        method_label = 'up'
    else:
        method_label = 'averaged'
    
    filename = f'stiffness_fig_log_SI_by_group_{method_label}_{range_label}'
    save_figure(fig, output_dir, filename)
    plt.close()


def _ensure_diabetes_bool(results_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Diabetes column is boolean for group comparisons."""
    df = results_df.copy()
    if 'Diabetes' not in df.columns:
        return df
    if df['Diabetes'].dtype == object or df['Diabetes'].dtype.name == 'str':
        df['Diabetes'] = df['Diabetes'].astype(str).str.upper() == 'TRUE'
    return df


def collect_significance_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Collect all significance test results into a single DataFrame for export.

    Includes: group comparisons (Mann-Whitney U), continuous associations
    (Pearson linear regression), and age-adjusted OLS (group p-value, age p-value).
    Saves nothing; returns the DataFrame (caller saves to CSV).

    Returns:
        DataFrame with columns: analysis, test, statistic, p_value, n, n_control, n_diabetic
        (n_control/n_diabetic only for group comparisons).
    """
    from src.analysis.stiffness_coeff import age_adjusted_analysis

    rows = []
    df = _ensure_diabetes_bool(results_df)

    def _mw_group(col_name: str, var_col: str, label: str) -> None:
        control = df[(df['Diabetes'] == False) & (df[var_col].notna())][var_col]
        diabetic = df[(df['Diabetes'] == True) & (df[var_col].notna())][var_col]
        if len(control) > 0 and len(diabetic) > 0:
            stat, p = stats.mannwhitneyu(control, diabetic, alternative='two-sided')
            rows.append({
                'analysis': label,
                'test': 'Mann-Whitney U',
                'statistic': stat,
                'p_value': p,
                'n': len(control) + len(diabetic),
                'n_control': len(control),
                'n_diabetic': len(diabetic),
            })

    def _linreg(y_col: str, x_col: str, label: str) -> None:
        valid = df[[y_col, x_col]].dropna()
        if len(valid) < 3:
            return
        x = valid[x_col].values
        y = valid[y_col].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        rows.append({
            'analysis': label,
            'test': 'Pearson regression',
            'statistic': r_value,
            'p_value': p_value,
            'n': len(valid),
            'n_control': None,
            'n_diabetic': None,
        })

    # Group comparisons
    if 'MAP' in df.columns:
        _mw_group('MAP', 'MAP', 'MAP by group (Control vs Diabetic)')
    if 'SYS_BP' in df.columns:
        _mw_group('SYS_BP', 'SYS_BP', 'SBP by group (Control vs Diabetic)')

    for col in ['stiffness_coeff_up_04_12', 'stiffness_coeff_up_02_12',
                'stiffness_coeff_averaged_04_12', 'stiffness_coeff_averaged_02_12']:
        if col in df.columns:
            _mw_group(col, col, f'{col} by group (Control vs Diabetic)')

    if 'P50' in df.columns:
        _mw_group('P50', 'P50', 'P50 by group (Control vs Diabetic)')
    if 'EV_lin' in df.columns:
        _mw_group('EV_lin', 'EV_lin', 'EV_lin by group (Control vs Diabetic)')
    if 'composite_stiffness' in df.columns:
        _mw_group('composite_stiffness', 'composite_stiffness',
                  'composite_stiffness by group (Control vs Diabetic)')

    for log_col in ['log_stiffness_coeff_up_04_12', 'log_stiffness_coeff_up_02_12',
                    'log_stiffness_coeff_averaged_04_12', 'log_stiffness_coeff_averaged_02_12',
                    'log_composite_stiffness']:
        if log_col in df.columns:
            _mw_group(log_col, log_col, f'{log_col} by group (Control vs Diabetic)')

    # Continuous associations (SI vs Age, MAP, SBP)
    for col in ['stiffness_coeff_up_04_12', 'stiffness_coeff_up_02_12',
                'stiffness_coeff_averaged_04_12', 'stiffness_coeff_averaged_02_12']:
        if col not in df.columns:
            continue
        if 'Age' in df.columns:
            _linreg(col, 'Age', f'{col} vs Age')
        if 'MAP' in df.columns:
            _linreg(col, 'MAP', f'{col} vs MAP')
        if 'SYS_BP' in df.columns:
            _linreg(col, 'SYS_BP', f'{col} vs SBP')

    # Age-adjusted analysis (OLS: SI ~ Group + Age)
    for stiffness_col in ['stiffness_coeff_averaged_04_12', 'stiffness_coeff_averaged_02_12',
                          'composite_stiffness']:
        if stiffness_col not in df.columns or 'Age' not in df.columns or 'Diabetes' not in df.columns:
            continue
        res = age_adjusted_analysis(df, stiffness_col, group_col='Diabetes')
        if 'error' in res:
            continue
        rows.append({
            'analysis': f'{stiffness_col} age-adjusted (group effect)',
            'test': 'OLS SI ~ Group + Age',
            'statistic': res.get('group_coef'),
            'p_value': res['group_pvalue'],
            'n': res['n'],
            'n_control': None,
            'n_diabetic': None,
        })
        rows.append({
            'analysis': f'{stiffness_col} age-adjusted (age effect)',
            'test': 'OLS SI ~ Group + Age',
            'statistic': res.get('age_coef'),
            'p_value': res['age_pvalue'],
            'n': res['n'],
            'n_control': None,
            'n_diabetic': None,
        })

    return pd.DataFrame(rows)


def plot_age_adjusted_analysis(results_df: pd.DataFrame, output_dir: str,
                               stiffness_col: str = 'stiffness_coeff_averaged_04_12',
                               figsize: Tuple[float, float] = (2.4, 2.0)):
    """Plot age-adjusted analysis showing group differences after controlling for age."""
    from src.analysis.stiffness_coeff import age_adjusted_analysis
    
    # Check if required columns exist
    required_cols = [stiffness_col, 'Diabetes', 'Age']
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns {missing_cols} for age-adjusted analysis. Skipping plot for {stiffness_col}.")
        return
    
    df_plot = results_df[[stiffness_col, 'Diabetes', 'Age']].dropna().copy()
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    # Perform age-adjusted analysis
    age_results = age_adjusted_analysis(results_df, stiffness_col, group_col='Diabetes')
    
    if 'error' in age_results:
        print(f"Warning: Could not perform age-adjusted analysis for {stiffness_col}")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points by group
    for group, color in [('Control', CONTROL_COLOR), ('Diabetic', DIABETES_COLOR)]:
        group_data = df_plot[df_plot['Group'] == group]
        ax.scatter(group_data['Age'], group_data[stiffness_col], 
                  alpha=0.6, s=20, color=color, label=group, zorder=3)
    
    # Fit separate regression lines for each group
    for group, color in [('Control', CONTROL_COLOR), ('Diabetic', DIABETES_COLOR)]:
        group_data = df_plot[df_plot['Group'] == group]
        if len(group_data) > 2:
            x = group_data['Age'].values
            y = group_data[stiffness_col].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, '--', linewidth=1, alpha=0.7, color=color)
    
    # Add text with age-adjusted p-value
    ax.text(0.05, 0.95, 
            f'Age-adjusted p = {age_results["group_pvalue"]:.4f}\n'
            f'R² = {age_results["r_squared"]:.3f}',
            transform=ax.transAxes, ha='left', va='top', fontsize=6,
            fontproperties=source_sans if source_sans else None,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Determine range label and whether it's log velocity
    is_log = '_log' in stiffness_col or 'log_' in stiffness_col
    if '02_12' in stiffness_col:
        range_label = '0.2-1.2'
    else:
        range_label = '0.4-1.2'
    
    # Set labels based on whether it's log velocity
    if is_log:
        ylabel = f'Log Stiffness Index (log SI_AUC)'
        title = f'Age-Adjusted Group Comparison (Log Velocity, {range_label} psi)'
    else:
        ylabel = 'Stiffness Index (SI_AUC)'
        title = f'Age-Adjusted Group Comparison ({range_label} psi)'
    
    ax.set_xlabel('Age (years)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel(ylabel, fontproperties=source_sans if source_sans else None)
    ax.set_title(title, fontproperties=source_sans if source_sans else None)
    ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)
    
    apply_font(ax, source_sans)
    plt.tight_layout()
    
    filename = f'stiffness_fig_age_adjusted_{stiffness_col}'
    save_figure(fig, output_dir, filename)
    plt.close()


def main():
    """Main function to generate all stiffness plots."""
    print("\nGenerating stiffness coefficient plots...")
    
    # Load stiffness results
    stiffness_file = os.path.join(cap_flow_path, 'results', 'Stiffness', 'stiffness_coefficients.csv')
    if not os.path.exists(stiffness_file):
        print(f"Error: Stiffness results file not found at {stiffness_file}")
        print("Please run stiffness_coeff.py first to generate the results.")
        return 1
    
    results_df = pd.read_csv(stiffness_file)
    print(f"Loaded stiffness results for {len(results_df)} participants")
    
    # Create output directory
    output_dir = os.path.join(cap_flow_path, 'results', 'Stiffness', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate main figures
    print("\nGenerating main figures...")
    plot_map_by_group(results_df, output_dir)
    plot_sbp_by_group(results_df, output_dir)
    
    # Generate stiffness by group plots for different methods/ranges
    for method in ['up', 'averaged']:
        for range_label in ['04_12', '02_12']:
            col = f'stiffness_coeff_{method}_{range_label}'
            if col in results_df.columns:
                plot_stiffness_by_group(results_df, output_dir, stiffness_col=col)
    
    # Generate regression plots
    print("\nGenerating regression plots...")
    for method in ['up', 'averaged']:
        for range_label in ['04_12', '02_12']:
            col = f'stiffness_coeff_{method}_{range_label}'
            if col in results_df.columns:
                plot_stiffness_vs_age(results_df, output_dir, stiffness_col=col)
                plot_stiffness_vs_map(results_df, output_dir, stiffness_col=col)
                plot_stiffness_vs_sbp(results_df, output_dir, stiffness_col=col)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    for range_label in ['04_12', '02_12']:
        compare_up_vs_averaged(results_df, output_dir, range_label=range_label)
    
    for method in ['up', 'averaged']:
        compare_ranges(results_df, output_dir, method=method)
    
    # Generate supplement figures for secondary metrics
    print("\nGenerating supplement figures...")
    if 'P50' in results_df.columns:
        plot_p50_vs_stiffness(results_df, output_dir)
        plot_p50_by_group(results_df, output_dir)
    if 'EV_lin' in results_df.columns:
        plot_ev_lin_vs_stiffness(results_df, output_dir)
        plot_ev_lin_by_group(results_df, output_dir)
    
    # Generate composite stiffness plots
    print("\nGenerating composite stiffness plots...")
    if 'composite_stiffness' in results_df.columns:
        plot_composite_stiffness_by_group(results_df, output_dir)
    
    # Generate log-transformed stiffness plots
    print("\nGenerating log-transformed stiffness plots...")
    for col in ['log_stiffness_coeff_up_04_12', 'log_stiffness_coeff_up_02_12',
                'log_stiffness_coeff_averaged_04_12', 'log_stiffness_coeff_averaged_02_12',
                'log_composite_stiffness']:
        if col in results_df.columns:
            plot_log_stiffness_by_group(results_df, output_dir, stiffness_col=col)
    
    # Generate age-adjusted analysis plots
    print("\nGenerating age-adjusted analysis plots...")
    for stiffness_col in ['stiffness_coeff_averaged_04_12', 'stiffness_coeff_averaged_02_12', 'composite_stiffness']:
        if stiffness_col in results_df.columns:
            plot_age_adjusted_analysis(results_df, output_dir, stiffness_col=stiffness_col)
    
    # Generate age-adjusted analysis plots for log velocity data
    print("\nGenerating age-adjusted analysis plots for log velocity...")
    log_stiffness_file = os.path.join(cap_flow_path, 'results', 'Stiffness', 'stiffness_coefficients_log.csv')
    if os.path.exists(log_stiffness_file):
        results_df_log = pd.read_csv(log_stiffness_file)
        print(f"Loaded log velocity stiffness results for {len(results_df_log)} participants")
        
        # Check if Age and Diabetes are already in the log results (they should be from calculate_stiffness_metrics)
        # If not, merge with age and diabetes data from main results
        if 'Age' not in results_df_log.columns or 'Diabetes' not in results_df_log.columns:
            if 'Age' in results_df.columns and 'Diabetes' in results_df.columns:
                age_diabetes = results_df[['Participant', 'Age', 'Diabetes']].copy()
                results_df_log = results_df_log.merge(age_diabetes, on='Participant', how='left')
                print("Merged Age and Diabetes data from main results")
            else:
                print("Warning: Age and Diabetes columns not found. Skipping log velocity age-adjusted plots.")
                results_df_log = None
        
        if results_df_log is not None and 'Age' in results_df_log.columns and 'Diabetes' in results_df_log.columns:
            # Use the log velocity stiffness metrics (not the log-transformed ones)
            for stiffness_col in ['stiffness_coeff_averaged_04_12_log', 'stiffness_coeff_averaged_02_12_log']:
                if stiffness_col in results_df_log.columns:
                    plot_age_adjusted_analysis(results_df_log, output_dir, stiffness_col=stiffness_col)
        else:
            print("Warning: Age and Diabetes data not available for log velocity results. Skipping age-adjusted plots.")
    
    # Calculate and save correlation table
    print("\nCalculating correlations...")
    corr_df = calculate_correlations(results_df)
    corr_file = os.path.join(output_dir, 'stiffness_correlations.csv')
    corr_df.to_csv(corr_file, index=False)
    print(f"Saved correlations to: {corr_file}")
    print("\nCorrelation Summary:")
    print(corr_df.to_string())

    # Collect and save all significance test results
    print("\nCollecting significance results...")
    sig_df = collect_significance_results(results_df)
    sig_file = os.path.join(output_dir, 'stiffness_significance.csv')
    sig_df.to_csv(sig_file, index=False)
    print(f"Saved significance results to: {sig_file}")

    print("\nAll plots generated successfully!")
    return 0


if __name__ == "__main__":
    main()

