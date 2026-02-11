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
from src.tools.plotting_utils import create_monochromatic_palette, adjust_brightness_of_colors
from matplotlib.colors import rgb2hex
from src.analysis.hysteresis import calculate_velocity_hysteresis

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

# Base colors for monochromatic palettes (matching plotting_utils / hysteresis conventions)
AGE_BASE_COLOR = '#1f77b4'       # Blue for age
BP_BASE_COLOR = '#2ca02c'        # Green for blood pressure
DISEASE_BASE_COLOR = '#00CED1'   # Teal for control-vs-any-disease


def _make_binary_palette(base_color: str):
    """Create a two-colour palette (darker, lighter) from a base hex colour.

    Uses the same monochromatic approach as hysteresis.py for grayscale
    readability: palette[4] (darker) for the first group and palette[1]
    (lighter) for the second group.

    Args:
        base_color: Hex colour string, e.g. '#1f77b4'.

    Returns:
        List of two hex colour strings [darker, lighter].
    """
    palette = create_monochromatic_palette(base_color)
    palette = adjust_brightness_of_colors(palette, brightness_scale=0.1)
    return [rgb2hex(palette[4]), rgb2hex(palette[1])]


def _ensure_healthy_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure *is_healthy* column exists (True for controls, False otherwise).

    Derives from SET column when available; otherwise falls back to Diabetes
    and Hypertension booleans.
    """
    out = df.copy()
    if 'is_healthy' not in out.columns:
        if 'SET' in out.columns:
            out['is_healthy'] = out['SET'].astype(str).str.startswith('set01')
        elif 'Diabetes' in out.columns and 'Hypertension' in out.columns:
            out['is_healthy'] = ~(out['Diabetes'].astype(bool) | out['Hypertension'].astype(bool))
        elif 'Diabetes' in out.columns:
            out['is_healthy'] = ~out['Diabetes'].astype(bool)
        else:
            warnings.warn("Cannot derive is_healthy flag – no SET, Diabetes, or Hypertension column.")
    return out


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


def save_figure(fig, output_dir: str, filename: str, dpi: int = 300,
                minimal_dir: Optional[str] = None):
    """Save figure as both PDF and PNG.
    
    If minimal_dir is set, also saves a copy with title and legend removed
    to minimal_dir using the same filename (for figure assembly).
    
    Args:
        fig: matplotlib figure object
        output_dir: Directory to save files
        filename: Base filename (without extension)
        dpi: Resolution for PNG (default: 300)
        minimal_dir: If set, save a no-title/no-legend copy here with same filename.
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

    if minimal_dir is not None:
        for ax in fig.get_axes():
            ax.set_title('')
            leg = ax.get_legend()
            if leg is not None:
                leg.set_visible(False)
            for txt in ax.texts:
                txt.set_visible(False)
        os.makedirs(minimal_dir, exist_ok=True)
        minimal_pdf = os.path.join(minimal_dir, f'{filename}.pdf')
        fig.savefig(minimal_pdf, dpi=dpi, bbox_inches='tight')
        minimal_png_dir = os.path.join(minimal_dir, 'png')
        os.makedirs(minimal_png_dir, exist_ok=True)
        minimal_png = os.path.join(minimal_png_dir, f'{filename}.png')
        fig.savefig(minimal_png, dpi=dpi, bbox_inches='tight')
        print(f"Saved (no annotations): {minimal_pdf}")
        print(f"Saved (no annotations): {minimal_png}")


def plot_map_by_group(results_df: pd.DataFrame, output_dir: str, figsize: Tuple[float, float] = (2.4, 2.0),
                      minimal_dir: Optional[str] = None):
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
    
    save_figure(fig, output_dir, 'stiffness_fig_MAP_by_group', minimal_dir=minimal_dir)
    plt.close()


def plot_sbp_by_group(results_df: pd.DataFrame, output_dir: str, figsize: Tuple[float, float] = (2.4, 2.0),
                      minimal_dir: Optional[str] = None):
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
    
    save_figure(fig, output_dir, 'stiffness_fig_SBP_by_group', minimal_dir=minimal_dir)
    plt.close()


def plot_stiffness_by_group(results_df: pd.DataFrame, output_dir: str, 
                           stiffness_col: str = 'stiffness_coeff_averaged_04_12',
                           figsize: Tuple[float, float] = (2.4, 2.0),
                           minimal_dir: Optional[str] = None):
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
                          figsize: Tuple[float, float] = (2.4, 2.0),
                          minimal_dir: Optional[str] = None):
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
    save_figure(fig, output_dir, filename, minimal_dir=minimal_dir)
    plt.close()


def plot_stiffness_vs_map(results_df: pd.DataFrame, output_dir: str,
                          stiffness_col: str = 'stiffness_coeff_averaged_04_12',
                          figsize: Tuple[float, float] = (2.4, 2.0),
                          minimal_dir: Optional[str] = None):
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
    save_figure(fig, output_dir, filename, minimal_dir=minimal_dir)
    plt.close()


def plot_stiffness_vs_sbp(results_df: pd.DataFrame, output_dir: str,
                          stiffness_col: str = 'stiffness_coeff_averaged_04_12',
                          figsize: Tuple[float, float] = (2.4, 2.0),
                          minimal_dir: Optional[str] = None):
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
    save_figure(fig, output_dir, filename, minimal_dir=minimal_dir)
    plt.close()


# ---------------------------------------------------------------------------
# Hysteresis vs Age scatterplots
# ---------------------------------------------------------------------------

def plot_hysteresis_vs_age_control(hysteresis_df: pd.DataFrame, output_dir: str,
                                   figsize: Tuple[float, float] = (2.4, 2.0),
                                   minimal_dir: Optional[str] = None):
    """Plot hysteresis vs age for control group only with regression line.

    Args:
        hysteresis_df: DataFrame with 'up_down_diff', 'Age', and 'is_healthy' columns.
        output_dir: Directory to save figures.
        figsize: Figure size.
        minimal_dir: If set, save a no-title/no-legend copy here.

    Returns:
        List of dicts with significance results.
    """
    sig_rows = []

    if 'up_down_diff' not in hysteresis_df.columns:
        print("Warning: up_down_diff not found. Skipping hysteresis vs age control plot.")
        return sig_rows
    if 'Age' not in hysteresis_df.columns:
        print("Warning: Age not found. Skipping hysteresis vs age control plot.")
        return sig_rows

    # Ensure is_healthy flag
    df = _ensure_healthy_flag(hysteresis_df)

    # Filter to controls with valid data
    df_plot = df[(df['is_healthy'] == True) &
                 df['up_down_diff'].notna() &
                 df['Age'].notna()].copy()

    if len(df_plot) < 3:
        print("Warning: Too few control participants for hysteresis vs age plot.")
        return sig_rows

    fig, ax = plt.subplots(figsize=figsize)

    # Plot points
    ax.scatter(df_plot['Age'], df_plot['up_down_diff'],
               alpha=0.6, s=20, color=CONTROL_COLOR, label='Control', zorder=3)

    # Fit regression line
    x = df_plot['Age'].values
    y = df_plot['up_down_diff'].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7)

        # Add correlation text
        ax.text(0.05, 0.95, f'R = {r_value:.3f}\np = {p_value:.4f}\nn = {len(df_plot)}',
                transform=ax.transAxes, ha='left', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        sig_rows.append({
            'analysis': 'Hysteresis vs Age (controls)',
            'test': 'Pearson correlation',
            'group1': 'Control', 'group2': 'N/A',
            'n1': len(df_plot), 'n2': np.nan,
            'statistic': r_value, 'p_value': p_value,
            'cohens_d': np.nan,
        })

    ax.set_xlabel('Age (years)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Velocity Hysteresis (up-down)', fontproperties=source_sans if source_sans else None)
    ax.set_title('Hysteresis vs Age (Controls)', fontproperties=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)

    apply_font(ax, source_sans)
    plt.tight_layout()

    save_figure(fig, output_dir, 'hysteresis_vs_age_control', minimal_dir=minimal_dir)
    plt.close()

    return sig_rows


def plot_hysteresis_vs_age_all(hysteresis_df: pd.DataFrame, output_dir: str,
                               figsize: Tuple[float, float] = (2.4, 2.0),
                               minimal_dir: Optional[str] = None):
    """Plot hysteresis vs age for control, diabetic, and hypertensive groups.

    Args:
        hysteresis_df: DataFrame with 'up_down_diff', 'Age', 'Diabetes', 'Hypertension' columns.
        output_dir: Directory to save figures.
        figsize: Figure size.
        minimal_dir: If set, save a no-title/no-legend copy here.

    Returns:
        List of dicts with significance results.
    """
    sig_rows = []

    if 'up_down_diff' not in hysteresis_df.columns:
        print("Warning: up_down_diff not found. Skipping hysteresis vs age all plot.")
        return sig_rows
    if 'Age' not in hysteresis_df.columns:
        print("Warning: Age not found. Skipping hysteresis vs age all plot.")
        return sig_rows

    # Ensure boolean columns
    df = _ensure_diabetes_bool(hysteresis_df.copy())
    if 'Hypertension' in df.columns:
        if df['Hypertension'].dtype == object or df['Hypertension'].dtype.name == 'str':
            df['Hypertension'] = df['Hypertension'].astype(str).str.upper() == 'TRUE'

    # Filter to valid data
    df_plot = df[df['up_down_diff'].notna() & df['Age'].notna()].copy()

    if len(df_plot) < 3:
        print("Warning: Too few participants for hysteresis vs age all plot.")
        return sig_rows

    # Create group labels: prioritize Diabetes > Hypertension > Control
    def assign_group(row):
        if row.get('Diabetes', False):
            return 'Diabetic'
        elif row.get('Hypertension', False):
            return 'Hypertensive'
        else:
            return 'Control'

    df_plot['Group'] = df_plot.apply(assign_group, axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for each group
    group_colors = {
        'Control': CONTROL_COLOR,
        'Diabetic': DIABETES_COLOR,
        'Hypertensive': '#d62728',  # Red for hypertension
    }

    # Plot points by group
    for group in ['Control', 'Diabetic', 'Hypertensive']:
        group_data = df_plot[df_plot['Group'] == group]
        if len(group_data) > 0:
            ax.scatter(group_data['Age'], group_data['up_down_diff'],
                       alpha=0.6, s=20, color=group_colors[group], label=group, zorder=3)

    # Fit overall regression line
    x = df_plot['Age'].values
    y = df_plot['up_down_diff'].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7)

        # Add correlation text
        ax.text(0.05, 0.95, f'R = {r_value:.3f}\np = {p_value:.4f}\nn = {len(df_plot)}',
                transform=ax.transAxes, ha='left', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        sig_rows.append({
            'analysis': 'Hysteresis vs Age (all participants)',
            'test': 'Pearson correlation',
            'group1': 'All', 'group2': 'N/A',
            'n1': len(df_plot), 'n2': np.nan,
            'statistic': r_value, 'p_value': p_value,
            'cohens_d': np.nan,
        })

    ax.set_xlabel('Age (years)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Velocity Hysteresis (up-down)', fontproperties=source_sans if source_sans else None)
    ax.set_title('Hysteresis vs Age', fontproperties=source_sans if source_sans else None)
    ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)

    apply_font(ax, source_sans)
    plt.tight_layout()

    save_figure(fig, output_dir, 'hysteresis_vs_age_all', minimal_dir=minimal_dir)
    plt.close()

    return sig_rows


def plot_abs_hysteresis_vs_age_control(hysteresis_df: pd.DataFrame, output_dir: str,
                                      figsize: Tuple[float, float] = (2.4, 2.0),
                                      minimal_dir: Optional[str] = None):
    """Plot absolute value of hysteresis vs age for control group only with regression line.

    Args:
        hysteresis_df: DataFrame with 'up_down_diff', 'Age', and 'is_healthy' columns.
        output_dir: Directory to save figures.
        figsize: Figure size.
        minimal_dir: If set, save a no-title/no-legend copy here.

    Returns:
        List of dicts with significance results.
    """
    sig_rows = []

    if 'up_down_diff' not in hysteresis_df.columns:
        print("Warning: up_down_diff not found. Skipping |hysteresis| vs age control plot.")
        return sig_rows
    if 'Age' not in hysteresis_df.columns:
        print("Warning: Age not found. Skipping |hysteresis| vs age control plot.")
        return sig_rows

    df = _ensure_healthy_flag(hysteresis_df)
    df_plot = df[(df['is_healthy'] == True) &
                 df['up_down_diff'].notna() &
                 df['Age'].notna()].copy()
    df_plot['abs_up_down_diff'] = df_plot['up_down_diff'].abs()

    if len(df_plot) < 3:
        print("Warning: Too few control participants for |hysteresis| vs age plot.")
        return sig_rows

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(df_plot['Age'], df_plot['abs_up_down_diff'],
               alpha=0.6, s=20, color=CONTROL_COLOR, label='Control', zorder=3)

    x = df_plot['Age'].values
    y = df_plot['abs_up_down_diff'].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7)

        ax.text(0.05, 0.95, f'R = {r_value:.3f}\np = {p_value:.4f}\nn = {len(df_plot)}',
                transform=ax.transAxes, ha='left', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        sig_rows.append({
            'analysis': '|Hysteresis| vs Age (controls)',
            'test': 'Pearson correlation',
            'group1': 'Control', 'group2': 'N/A',
            'n1': len(df_plot), 'n2': np.nan,
            'statistic': r_value, 'p_value': p_value,
            'cohens_d': np.nan,
        })

    ax.set_xlabel('Age (years)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('|Velocity Hysteresis| (up-down)', fontproperties=source_sans if source_sans else None)
    ax.set_title('|Hysteresis| vs Age (Controls)', fontproperties=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)

    apply_font(ax, source_sans)
    plt.tight_layout()

    save_figure(fig, output_dir, 'abs_hysteresis_vs_age_control', minimal_dir=minimal_dir)
    plt.close()

    return sig_rows


def plot_abs_hysteresis_vs_age_all(hysteresis_df: pd.DataFrame, output_dir: str,
                                  figsize: Tuple[float, float] = (2.4, 2.0),
                                  minimal_dir: Optional[str] = None):
    """Plot absolute value of hysteresis vs age for control, diabetic, and hypertensive groups.

    Args:
        hysteresis_df: DataFrame with 'up_down_diff', 'Age', 'Diabetes', 'Hypertension' columns.
        output_dir: Directory to save figures.
        figsize: Figure size.
        minimal_dir: If set, save a no-title/no-legend copy here.

    Returns:
        List of dicts with significance results.
    """
    sig_rows = []

    if 'up_down_diff' not in hysteresis_df.columns:
        print("Warning: up_down_diff not found. Skipping |hysteresis| vs age all plot.")
        return sig_rows
    if 'Age' not in hysteresis_df.columns:
        print("Warning: Age not found. Skipping |hysteresis| vs age all plot.")
        return sig_rows

    df = _ensure_diabetes_bool(hysteresis_df.copy())
    if 'Hypertension' in df.columns:
        if df['Hypertension'].dtype == object or df['Hypertension'].dtype.name == 'str':
            df['Hypertension'] = df['Hypertension'].astype(str).str.upper() == 'TRUE'

    df_plot = df[df['up_down_diff'].notna() & df['Age'].notna()].copy()
    df_plot['abs_up_down_diff'] = df_plot['up_down_diff'].abs()

    if len(df_plot) < 3:
        print("Warning: Too few participants for |hysteresis| vs age all plot.")
        return sig_rows

    def assign_group(row):
        if row.get('Diabetes', False):
            return 'Diabetic'
        elif row.get('Hypertension', False):
            return 'Hypertensive'
        else:
            return 'Control'

    df_plot['Group'] = df_plot.apply(assign_group, axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    group_colors = {
        'Control': CONTROL_COLOR,
        'Diabetic': DIABETES_COLOR,
        'Hypertensive': '#d62728',
    }

    for group in ['Control', 'Diabetic', 'Hypertensive']:
        group_data = df_plot[df_plot['Group'] == group]
        if len(group_data) > 0:
            ax.scatter(group_data['Age'], group_data['abs_up_down_diff'],
                       alpha=0.6, s=20, color=group_colors[group], label=group, zorder=3)

    x = df_plot['Age'].values
    y = df_plot['abs_up_down_diff'].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7)

        ax.text(0.05, 0.95, f'R = {r_value:.3f}\np = {p_value:.4f}\nn = {len(df_plot)}',
                transform=ax.transAxes, ha='left', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        sig_rows.append({
            'analysis': '|Hysteresis| vs Age (all participants)',
            'test': 'Pearson correlation',
            'group1': 'All', 'group2': 'N/A',
            'n1': len(df_plot), 'n2': np.nan,
            'statistic': r_value, 'p_value': p_value,
            'cohens_d': np.nan,
        })

    ax.set_xlabel('Age (years)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('|Velocity Hysteresis| (up-down)', fontproperties=source_sans if source_sans else None)
    ax.set_title('|Hysteresis| vs Age', fontproperties=source_sans if source_sans else None)
    ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)

    apply_font(ax, source_sans)
    plt.tight_layout()

    save_figure(fig, output_dir, 'abs_hysteresis_vs_age_all', minimal_dir=minimal_dir)
    plt.close()

    return sig_rows


def plot_hysteresis_vs_age_hypertension(hysteresis_df: pd.DataFrame, output_dir: str,
                                        figsize: Tuple[float, float] = (2.4, 2.0),
                                        minimal_dir: Optional[str] = None):
    """Plot hysteresis vs age for Control vs Hypertensive groups.

    Args:
        hysteresis_df: DataFrame with 'up_down_diff', 'Age', 'Hypertension' columns.
        output_dir: Directory to save figures.
        figsize: Figure size.
        minimal_dir: If set, save a no-title/no-legend copy here.

    Returns:
        List of dicts with significance results.
    """
    sig_rows = []
    HYPERTENSION_COLOR = '#d62728'  # Red

    if 'up_down_diff' not in hysteresis_df.columns:
        print("Warning: up_down_diff not found. Skipping hysteresis vs age hypertension plot.")
        return sig_rows
    if 'Age' not in hysteresis_df.columns:
        print("Warning: Age not found. Skipping hysteresis vs age hypertension plot.")
        return sig_rows
    if 'Hypertension' not in hysteresis_df.columns:
        print("Warning: Hypertension not found. Skipping hysteresis vs age hypertension plot.")
        return sig_rows

    # Ensure boolean column
    df = hysteresis_df.copy()
    if df['Hypertension'].dtype == object or df['Hypertension'].dtype.name == 'str':
        df['Hypertension'] = df['Hypertension'].astype(str).str.upper() == 'TRUE'

    # Filter to valid data (exclude diabetics to isolate hypertension effect)
    df = _ensure_diabetes_bool(df)
    df_plot = df[df['up_down_diff'].notna() & df['Age'].notna()].copy()
    # Exclude diabetics
    if 'Diabetes' in df_plot.columns:
        df_plot = df_plot[df_plot['Diabetes'] == False]

    if len(df_plot) < 3:
        print("Warning: Too few participants for hysteresis vs age hypertension plot.")
        return sig_rows

    # Create group labels
    df_plot['Group'] = df_plot['Hypertension'].map({True: 'Hypertensive', False: 'Control'})

    fig, ax = plt.subplots(figsize=figsize)

    # Plot points by group
    for group, color in [('Control', CONTROL_COLOR), ('Hypertensive', HYPERTENSION_COLOR)]:
        group_data = df_plot[df_plot['Group'] == group]
        if len(group_data) > 0:
            ax.scatter(group_data['Age'], group_data['up_down_diff'],
                       alpha=0.6, s=20, color=color, label=group, zorder=3)

    # Fit overall regression line
    x = df_plot['Age'].values
    y = df_plot['up_down_diff'].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7)

        ax.text(0.05, 0.95, f'R = {r_value:.3f}\np = {p_value:.4f}\nn = {len(df_plot)}',
                transform=ax.transAxes, ha='left', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        sig_rows.append({
            'analysis': 'Hysteresis vs Age (Control vs Hypertensive)',
            'test': 'Pearson correlation',
            'group1': 'All (excl. diabetics)', 'group2': 'N/A',
            'n1': len(df_plot), 'n2': np.nan,
            'statistic': r_value, 'p_value': p_value,
            'cohens_d': np.nan,
        })

    ax.set_xlabel('Age (years)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Velocity Hysteresis (up-down)', fontproperties=source_sans if source_sans else None)
    ax.set_title('Hysteresis vs Age (Hypertension)', fontproperties=source_sans if source_sans else None)
    ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)

    apply_font(ax, source_sans)
    plt.tight_layout()

    save_figure(fig, output_dir, 'hysteresis_vs_age_hypertension', minimal_dir=minimal_dir)
    plt.close()

    return sig_rows


def plot_hysteresis_vs_age_diabetes(hysteresis_df: pd.DataFrame, output_dir: str,
                                    figsize: Tuple[float, float] = (2.4, 2.0),
                                    minimal_dir: Optional[str] = None):
    """Plot hysteresis vs age for Control vs Diabetic groups.

    Args:
        hysteresis_df: DataFrame with 'up_down_diff', 'Age', 'Diabetes' columns.
        output_dir: Directory to save figures.
        figsize: Figure size.
        minimal_dir: If set, save a no-title/no-legend copy here.

    Returns:
        List of dicts with significance results.
    """
    sig_rows = []

    if 'up_down_diff' not in hysteresis_df.columns:
        print("Warning: up_down_diff not found. Skipping hysteresis vs age diabetes plot.")
        return sig_rows
    if 'Age' not in hysteresis_df.columns:
        print("Warning: Age not found. Skipping hysteresis vs age diabetes plot.")
        return sig_rows

    # Ensure boolean column
    df = _ensure_diabetes_bool(hysteresis_df.copy())

    # Filter to valid data
    df_plot = df[df['up_down_diff'].notna() & df['Age'].notna()].copy()

    if len(df_plot) < 3:
        print("Warning: Too few participants for hysteresis vs age diabetes plot.")
        return sig_rows

    # Create group labels
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})

    fig, ax = plt.subplots(figsize=figsize)

    # Plot points by group
    for group, color in [('Control', CONTROL_COLOR), ('Diabetic', DIABETES_COLOR)]:
        group_data = df_plot[df_plot['Group'] == group]
        if len(group_data) > 0:
            ax.scatter(group_data['Age'], group_data['up_down_diff'],
                       alpha=0.6, s=20, color=color, label=group, zorder=3)

    # Fit overall regression line
    x = df_plot['Age'].values
    y = df_plot['up_down_diff'].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7)

        ax.text(0.05, 0.95, f'R = {r_value:.3f}\np = {p_value:.4f}\nn = {len(df_plot)}',
                transform=ax.transAxes, ha='left', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        sig_rows.append({
            'analysis': 'Hysteresis vs Age (Control vs Diabetic)',
            'test': 'Pearson correlation',
            'group1': 'All', 'group2': 'N/A',
            'n1': len(df_plot), 'n2': np.nan,
            'statistic': r_value, 'p_value': p_value,
            'cohens_d': np.nan,
        })

    ax.set_xlabel('Age (years)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Velocity Hysteresis (up-down)', fontproperties=source_sans if source_sans else None)
    ax.set_title('Hysteresis vs Age (Diabetes)', fontproperties=source_sans if source_sans else None)
    ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)

    apply_font(ax, source_sans)
    plt.tight_layout()

    save_figure(fig, output_dir, 'hysteresis_vs_age_diabetes', minimal_dir=minimal_dir)
    plt.close()

    return sig_rows


def plot_hysteresis_boxplot_hypertension(hysteresis_df: pd.DataFrame, output_dir: str,
                                         figsize: Tuple[float, float] = (2.4, 2.0),
                                         minimal_dir: Optional[str] = None):
    """Boxplot of hysteresis comparing Control vs Hypertensive groups.

    Args:
        hysteresis_df: DataFrame with 'up_down_diff' and 'Hypertension' columns.
        output_dir: Directory to save figures.
        figsize: Figure size.
        minimal_dir: If set, save a no-title/no-legend copy here.

    Returns:
        List of dicts with significance results.
    """
    sig_rows = []
    HYPERTENSION_COLOR = '#d62728'  # Red

    if 'up_down_diff' not in hysteresis_df.columns:
        print("Warning: up_down_diff not found. Skipping hysteresis boxplot hypertension.")
        return sig_rows
    if 'Hypertension' not in hysteresis_df.columns:
        print("Warning: Hypertension not found. Skipping hysteresis boxplot hypertension.")
        return sig_rows

    # Ensure boolean column
    df = hysteresis_df.copy()
    if df['Hypertension'].dtype == object or df['Hypertension'].dtype.name == 'str':
        df['Hypertension'] = df['Hypertension'].astype(str).str.upper() == 'TRUE'

    # Filter to valid data (exclude diabetics to isolate hypertension effect)
    df = _ensure_diabetes_bool(df)
    df_plot = df[df['up_down_diff'].notna()].copy()
    if 'Diabetes' in df_plot.columns:
        df_plot = df_plot[df_plot['Diabetes'] == False]

    if len(df_plot) < 3:
        print("Warning: Too few participants for hysteresis boxplot hypertension.")
        return sig_rows

    # Create group labels
    df_plot['Group'] = df_plot['Hypertension'].map({True: 'Hypertensive', False: 'Control'})
    group_labels = ['Control', 'Hypertensive']
    palette = [CONTROL_COLOR, HYPERTENSION_COLOR]

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(data=df_plot, x='Group', y='up_down_diff', ax=ax,
                order=group_labels, palette=palette, width=0.6)

    # Add jittered individual points
    for i, grp in enumerate(group_labels):
        grp_vals = df_plot[df_plot['Group'] == grp]['up_down_diff']
        x_pos = i + np.random.normal(0, 0.04, len(grp_vals))
        ax.scatter(x_pos, grp_vals, alpha=0.4, s=15, color='black', zorder=3)

    # Statistical test
    ctrl_vals = df_plot[df_plot['Group'] == 'Control']['up_down_diff']
    hyp_vals = df_plot[df_plot['Group'] == 'Hypertensive']['up_down_diff']
    stat_text_parts = [f'n = {len(ctrl_vals)}, {len(hyp_vals)}']

    if len(ctrl_vals) > 0 and len(hyp_vals) > 0:
        stat_val, pval = stats.mannwhitneyu(ctrl_vals, hyp_vals, alternative='two-sided')
        d = _cohens_d(ctrl_vals, hyp_vals)
        stat_text_parts.append(f'p = {pval:.4f}')
        if d is not None and not np.isnan(d):
            stat_text_parts.append(f'd = {d:.2f}')
        sig_rows.append({
            'analysis': 'Hysteresis by Hypertension',
            'test': 'Mann-Whitney U',
            'group1': 'Control', 'group2': 'Hypertensive',
            'n1': len(ctrl_vals), 'n2': len(hyp_vals),
            'statistic': stat_val, 'p_value': pval,
            'cohens_d': d if d is not None else np.nan,
        })

    ax.text(0.5, 0.95, '\n'.join(stat_text_parts), transform=ax.transAxes,
            ha='center', va='top', fontsize=6,
            fontproperties=source_sans if source_sans else None,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                      edgecolor='black', linewidth=0.5))

    ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Velocity Hysteresis (up-down)', fontproperties=source_sans if source_sans else None)
    ax.set_title('Hysteresis by Hypertension', fontproperties=source_sans if source_sans else None)

    apply_font(ax, source_sans)
    plt.tight_layout()

    save_figure(fig, output_dir, 'hysteresis_boxplot_hypertension', minimal_dir=minimal_dir)
    plt.close()

    return sig_rows


def plot_hysteresis_boxplot_diabetes(hysteresis_df: pd.DataFrame, output_dir: str,
                                     figsize: Tuple[float, float] = (2.4, 2.0),
                                     minimal_dir: Optional[str] = None):
    """Boxplot of hysteresis comparing Control vs Diabetic groups.

    Args:
        hysteresis_df: DataFrame with 'up_down_diff' and 'Diabetes' columns.
        output_dir: Directory to save figures.
        figsize: Figure size.
        minimal_dir: If set, save a no-title/no-legend copy here.

    Returns:
        List of dicts with significance results.
    """
    sig_rows = []

    if 'up_down_diff' not in hysteresis_df.columns:
        print("Warning: up_down_diff not found. Skipping hysteresis boxplot diabetes.")
        return sig_rows

    # Ensure boolean column
    df = _ensure_diabetes_bool(hysteresis_df.copy())

    # Filter to valid data
    df_plot = df[df['up_down_diff'].notna()].copy()

    if len(df_plot) < 3:
        print("Warning: Too few participants for hysteresis boxplot diabetes.")
        return sig_rows

    # Create group labels
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    group_labels = ['Control', 'Diabetic']
    palette = [CONTROL_COLOR, DIABETES_COLOR]

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(data=df_plot, x='Group', y='up_down_diff', ax=ax,
                order=group_labels, palette=palette, width=0.6)

    # Add jittered individual points
    for i, grp in enumerate(group_labels):
        grp_vals = df_plot[df_plot['Group'] == grp]['up_down_diff']
        x_pos = i + np.random.normal(0, 0.04, len(grp_vals))
        ax.scatter(x_pos, grp_vals, alpha=0.4, s=15, color='black', zorder=3)

    # Statistical test
    ctrl_vals = df_plot[df_plot['Group'] == 'Control']['up_down_diff']
    diab_vals = df_plot[df_plot['Group'] == 'Diabetic']['up_down_diff']
    stat_text_parts = [f'n = {len(ctrl_vals)}, {len(diab_vals)}']

    if len(ctrl_vals) > 0 and len(diab_vals) > 0:
        stat_val, pval = stats.mannwhitneyu(ctrl_vals, diab_vals, alternative='two-sided')
        d = _cohens_d(ctrl_vals, diab_vals)
        stat_text_parts.append(f'p = {pval:.4f}')
        if d is not None and not np.isnan(d):
            stat_text_parts.append(f'd = {d:.2f}')
        sig_rows.append({
            'analysis': 'Hysteresis by Diabetes',
            'test': 'Mann-Whitney U',
            'group1': 'Control', 'group2': 'Diabetic',
            'n1': len(ctrl_vals), 'n2': len(diab_vals),
            'statistic': stat_val, 'p_value': pval,
            'cohens_d': d if d is not None else np.nan,
        })

    ax.text(0.5, 0.95, '\n'.join(stat_text_parts), transform=ax.transAxes,
            ha='center', va='top', fontsize=6,
            fontproperties=source_sans if source_sans else None,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                      edgecolor='black', linewidth=0.5))

    ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('Velocity Hysteresis (up-down)', fontproperties=source_sans if source_sans else None)
    ax.set_title('Hysteresis by Diabetes', fontproperties=source_sans if source_sans else None)

    apply_font(ax, source_sans)
    plt.tight_layout()

    save_figure(fig, output_dir, 'hysteresis_boxplot_diabetes', minimal_dir=minimal_dir)
    plt.close()

    return sig_rows


def compare_up_vs_averaged(results_df: pd.DataFrame, output_dir: str,
                           range_label: str = '04_12',
                           figsize: Tuple[float, float] = (2.4, 2.0),
                           minimal_dir: Optional[str] = None):
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
                  figsize: Tuple[float, float] = (2.4, 2.0),
                  minimal_dir: Optional[str] = None):
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
                          figsize: Tuple[float, float] = (2.4, 2.0),
                          minimal_dir: Optional[str] = None):
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
                             figsize: Tuple[float, float] = (2.4, 2.0),
                             minimal_dir: Optional[str] = None):
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
                      figsize: Tuple[float, float] = (2.4, 2.0),
                      minimal_dir: Optional[str] = None):
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
                         figsize: Tuple[float, float] = (2.4, 2.0),
                         minimal_dir: Optional[str] = None):
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
                                     figsize: Tuple[float, float] = (2.4, 2.0),
                                     minimal_dir: Optional[str] = None):
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
                                stiffness_col: str = 'SI_log1p_averaged_04_12',
                                figsize: Tuple[float, float] = (2.4, 2.0),
                                minimal_dir: Optional[str] = None):
    """Plot boxplot of log-transformed stiffness by group.

    Handles both SI_log1p (log of AUC of raw velocity) and SI_logvel
    (AUC of log velocity) columns, setting axis labels and filenames
    appropriately based on the column prefix.

    Args:
        results_df: DataFrame containing the stiffness column and Diabetes flag.
        output_dir: Directory to save figures.
        stiffness_col: Column name to plot (SI_log1p_* or SI_logvel_*).
        figsize: Figure size.
        minimal_dir: If set, save a no-title/no-legend copy here.
    """
    if stiffness_col not in results_df.columns:
        return
    
    df_plot = _ensure_diabetes_bool(results_df)
    df_plot = df_plot[df_plot[stiffness_col].notna()].copy()
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    box_colors = [CONTROL_COLOR, DIABETES_COLOR]
    sns.boxplot(data=df_plot, x='Group', y=stiffness_col, ax=ax, palette=box_colors, width=0.6)
    
    for i, group in enumerate(['Control', 'Diabetic']):
        group_data = df_plot[df_plot['Group'] == group][stiffness_col]
        x_pos = i + np.random.normal(0, 0.05, len(group_data))
        ax.scatter(x_pos, group_data, alpha=0.4, s=10, color='gray', zorder=3)
    
    # Determine labels based on metric type
    is_logvel = stiffness_col.startswith('SI_logvel_')
    if is_logvel:
        ylabel = 'SI_logvel (AUC of log velocity)'
        title = 'Log-Velocity Stiffness by Group'
    else:
        ylabel = 'Log(AUC + 1)'
        title = 'Log-Transformed Stiffness by Group'
    
    ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel(ylabel, fontproperties=source_sans if source_sans else None)
    ax.set_title(title, fontproperties=source_sans if source_sans else None)
    
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
    
    # Use distinct filename prefix for logvel vs log1p
    if is_logvel:
        filename = f'stiffness_fig_logvel_SI_by_group_{method_label}_{range_label}'
    else:
        filename = f'stiffness_fig_log_SI_by_group_{method_label}_{range_label}'
    save_figure(fig, output_dir, filename, minimal_dir=minimal_dir)
    plt.close()


def _plot_group_boxplot(ax, df_plot: pd.DataFrame, y_col: str, ylabel: str, title: str) -> None:
    """Shared helper for group boxplots with scatter and p-value."""
    box_colors = [CONTROL_COLOR, DIABETES_COLOR]
    sns.boxplot(data=df_plot, x='Group', y=y_col, ax=ax, palette=box_colors, width=0.6)
    for i, group in enumerate(['Control', 'Diabetic']):
        group_data = df_plot[df_plot['Group'] == group][y_col]
        x_pos = i + np.random.normal(0, 0.05, len(group_data))
        ax.scatter(x_pos, group_data, alpha=0.4, s=10, color='gray', zorder=3)
    control_vals = df_plot[df_plot['Group'] == 'Control'][y_col]
    diabetic_vals = df_plot[df_plot['Group'] == 'Diabetic'][y_col]
    if len(control_vals) > 0 and len(diabetic_vals) > 0:
        stat, pval = stats.mannwhitneyu(control_vals, diabetic_vals, alternative='two-sided')
        ax.text(0.5, 0.95, f'p = {pval:.4f}', transform=ax.transAxes,
                ha='center', va='top', fontsize=6, fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel(ylabel, fontproperties=source_sans if source_sans else None)
    ax.set_title(title, fontproperties=source_sans if source_sans else None)
    apply_font(ax, source_sans)


def plot_log_metric_comparison_by_group(results_df: pd.DataFrame,
                                        results_df_log: pd.DataFrame,
                                        output_dir: str,
                                        range_label: str = '04_12',
                                        figsize: Tuple[float, float] = (4.8, 2.0),
                                        minimal_dir: Optional[str] = None):
    """Compare SI_log1p (log of AUC of raw velocity) vs SI_logvel (AUC of log velocity) by group.

    Args:
        results_df: Main stiffness results (contains SI_log1p_* columns).
        results_df_log: Log-velocity stiffness results (contains SI_logvel_* columns).
        output_dir: Directory to save figures.
        range_label: Pressure range label ('04_12' or '02_12').
        figsize: Figure size for the side-by-side panels.
        minimal_dir: If set, save a no-title/no-legend copy here.
    """
    metrics = []
    log1p_col = f'SI_log1p_averaged_{range_label}'
    if log1p_col in results_df.columns:
        df_raw = _ensure_diabetes_bool(results_df)
        df_plot = df_raw[[log1p_col, 'Diabetes']].dropna().copy()
        df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
        metrics.append((df_plot, log1p_col, 'SI_log1p', 'log1p(AUC raw vel)'))
    logvel_col = f'SI_logvel_averaged_{range_label}'
    if results_df_log is not None and logvel_col in results_df_log.columns:
        df_log = _ensure_diabetes_bool(results_df_log)
        df_plot = df_log[[logvel_col, 'Diabetes']].dropna().copy()
        df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})
        metrics.append((df_plot, logvel_col, 'SI_logvel', 'AUC of log velocity'))

    if len(metrics) == 0:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
    for idx, (df_plot, col, ylabel, title) in enumerate(metrics):
        _plot_group_boxplot(axes[0, idx], df_plot, col, ylabel, title)

    plt.tight_layout()
    filename = f'stiffness_fig_log_metric_compare_by_group_{range_label}'
    save_figure(fig, output_dir, filename, minimal_dir=minimal_dir)
    plt.close()


def plot_bp_si_by_condition(results_df: pd.DataFrame,
                            results_df_log: pd.DataFrame,
                            output_dir: str,
                            stiffness_col: str = 'SI_logvel_averaged_02_12',
                            bp_col: str = 'SYS_BP',
                            figsize: Tuple[float, float] = (4.8, 4.0),
                            minimal_dir: Optional[str] = None):
    """Create 2x2 boxplot comparing BP and SI_logvel for Diabetes and Hypertension.

    Panel layout:
        (0,0) BP for Diabetes (Control vs Diabetic)
        (0,1) BP for Hypertension (Control vs Hypertensive)
        (1,0) SI_logvel for Diabetes (Control vs Diabetic)
        (1,1) SI_logvel for Hypertension (Control vs Hypertensive)

    Args:
        results_df: Main stiffness results with BP and Diabetes/Hypertension columns.
        results_df_log: Log-velocity stiffness results with SI_logvel columns.
        output_dir: Directory to save figures.
        stiffness_col: SI_logvel column to plot (default SI_logvel_averaged_02_12).
        bp_col: Blood pressure column (default SYS_BP).
        figsize: Figure size for the 2x2 grid.
        minimal_dir: If set, save a no-title/no-legend copy here.

    Returns:
        List of dicts with significance results for each panel.
    """
    sig_rows = []

    # Ensure boolean columns
    df_main = _ensure_diabetes_bool(results_df.copy())
    if 'Hypertension' in df_main.columns:
        if df_main['Hypertension'].dtype == object or df_main['Hypertension'].dtype.name == 'str':
            df_main['Hypertension'] = df_main['Hypertension'].astype(str).str.upper() == 'TRUE'

    # Use log results for SI_logvel if available
    if results_df_log is not None and stiffness_col in results_df_log.columns:
        df_si = _ensure_diabetes_bool(results_df_log.copy())
        if 'Hypertension' in df_si.columns:
            if df_si['Hypertension'].dtype == object or df_si['Hypertension'].dtype.name == 'str':
                df_si['Hypertension'] = df_si['Hypertension'].astype(str).str.upper() == 'TRUE'
    else:
        df_si = df_main

    # Check required columns
    if bp_col not in df_main.columns:
        print(f"Warning: {bp_col} not found. Skipping BP vs SI condition comparison.")
        return sig_rows
    if stiffness_col not in df_si.columns:
        print(f"Warning: {stiffness_col} not found. Skipping BP vs SI condition comparison.")
        return sig_rows
    if 'Diabetes' not in df_main.columns or 'Hypertension' not in df_main.columns:
        print("Warning: Diabetes or Hypertension column not found. Skipping.")
        return sig_rows

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Define the 4 panels: (row, col, df, y_col, condition_col, ylabel, title_suffix)
    bp_label = 'Systolic BP (mmHg)' if bp_col == 'SYS_BP' else 'Diastolic BP (mmHg)'
    si_label = 'SI_logvel (AUC of log velocity)'

    panels = [
        (0, 0, df_main, bp_col, 'Diabetes', bp_label, 'Diabetes'),
        (0, 1, df_main, bp_col, 'Hypertension', bp_label, 'Hypertension'),
        (1, 0, df_si, stiffness_col, 'Diabetes', si_label, 'Diabetes'),
        (1, 1, df_si, stiffness_col, 'Hypertension', si_label, 'Hypertension'),
    ]

    for row, col, df, y_col, cond_col, ylabel, cond_name in panels:
        ax = axes[row, col]

        # Filter to valid data
        df_plot = df[df[y_col].notna() & df[cond_col].notna()].copy()

        # Create group labels
        if cond_col == 'Diabetes':
            df_plot['Group'] = df_plot[cond_col].map({True: 'Diabetic', False: 'Control'})
            group_labels = ['Control', 'Diabetic']
            palette = [CONTROL_COLOR, DIABETES_COLOR]
        else:  # Hypertension
            df_plot['Group'] = df_plot[cond_col].map({True: 'Hypertensive', False: 'Control'})
            group_labels = ['Control', 'Hypertensive']
            # Use a distinct color for hypertension (purple)
            palette = [CONTROL_COLOR, '#d62728']  # Red for hypertension

        # Create boxplot
        sns.boxplot(data=df_plot, x='Group', y=y_col, ax=ax,
                    order=group_labels, palette=palette, width=0.6)

        # Add jittered individual points
        for i, grp in enumerate(group_labels):
            grp_vals = df_plot[df_plot['Group'] == grp][y_col]
            x_pos = i + np.random.normal(0, 0.04, len(grp_vals))
            ax.scatter(x_pos, grp_vals, alpha=0.4, s=15, color='black', zorder=3)

        # Statistical test
        ctrl_vals = df_plot[df_plot['Group'] == group_labels[0]][y_col]
        cond_vals = df_plot[df_plot['Group'] == group_labels[1]][y_col]
        stat_text_parts = [f'n = {len(ctrl_vals)}, {len(cond_vals)}']

        if len(ctrl_vals) > 0 and len(cond_vals) > 0:
            stat_val, pval = stats.mannwhitneyu(ctrl_vals, cond_vals, alternative='two-sided')
            d = _cohens_d(ctrl_vals, cond_vals)
            stat_text_parts.append(f'p = {pval:.4f}')
            if d is not None and not np.isnan(d):
                stat_text_parts.append(f'd = {d:.2f}')
            sig_rows.append({
                'analysis': f'{y_col} by {cond_name}',
                'test': 'Mann-Whitney U',
                'group1': group_labels[0], 'group2': group_labels[1],
                'n1': len(ctrl_vals), 'n2': len(cond_vals),
                'statistic': stat_val, 'p_value': pval,
                'cohens_d': d if d is not None else np.nan,
            })

        ax.text(0.5, 0.95, '\n'.join(stat_text_parts), transform=ax.transAxes,
                ha='center', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                          edgecolor='black', linewidth=0.5))

        ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
        ax.set_ylabel(ylabel, fontproperties=source_sans if source_sans else None)
        ax.set_title(f'{ylabel.split(" (")[0]} by {cond_name}',
                     fontproperties=source_sans if source_sans else None)
        apply_font(ax, source_sans)

    plt.tight_layout()
    filename = 'stiffness_fig_BP_SI_by_condition_comparison'
    save_figure(fig, output_dir, filename, minimal_dir=minimal_dir)
    plt.close()

    return sig_rows


# ---------------------------------------------------------------------------
# Age-group and blood-pressure comparison functions
# ---------------------------------------------------------------------------

def plot_si_by_age_group_control(results_df: pd.DataFrame, output_dir: str,
                                  stiffness_col: str = 'SI_logvel_averaged_02_12',
                                  age_thresholds: list = None,
                                  figsize: Tuple[float, float] = (2.4, 2.0),
                                  minimal_dir: Optional[str] = None):
    """Box plots of SI_logvel comparing age groups within the control cohort.

    For each age threshold a binary split is created (<threshold vs
    >=threshold) and a box plot is produced with Mann-Whitney U p-value
    and Cohen's d.

    Args:
        results_df: DataFrame containing SI_logvel column, Age, and is_healthy.
        output_dir: Directory to save figures.
        stiffness_col: SI column to plot (default SI_logvel_averaged_02_12).
        age_thresholds: List of integer thresholds (default [30, 50, 60]).
        figsize: Figure size.
        minimal_dir: If set, save minimal copies.

    Returns:
        List of dicts with significance results for each threshold.
    """
    sig_rows = []
    if stiffness_col not in results_df.columns:
        print(f"Warning: {stiffness_col} not found. Skipping SI age-group control plots.")
        return sig_rows

    if age_thresholds is None:
        age_thresholds = [30, 50, 60]

    df = _ensure_healthy_flag(results_df)
    controls = df[(df['is_healthy'] == True) & df[stiffness_col].notna() & df['Age'].notna()].copy()
    if len(controls) < 4:
        print("Warning: Too few control participants for age-group SI analysis.")
        return sig_rows

    palette_hex = _make_binary_palette(AGE_BASE_COLOR)

    for threshold in age_thresholds:
        controls['Age_Group'] = np.where(controls['Age'] < threshold,
                                         f'<{threshold}', f'\u2265{threshold}')
        group_labels = [f'<{threshold}', f'\u2265{threshold}']

        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=controls, x='Age_Group', y=stiffness_col, ax=ax,
                    order=group_labels, palette=palette_hex, width=0.6)

        # Jittered individual points
        for i, grp in enumerate(group_labels):
            grp_data = controls[controls['Age_Group'] == grp][stiffness_col]
            x_pos = i + np.random.normal(0, 0.04, len(grp_data))
            ax.scatter(x_pos, grp_data, alpha=0.4, s=20, color='black', zorder=3)

        # Stats
        young = controls[controls['Age_Group'] == group_labels[0]][stiffness_col]
        old = controls[controls['Age_Group'] == group_labels[1]][stiffness_col]
        stat_text_parts = [f'n = {len(young)}, {len(old)}']
        if len(young) > 0 and len(old) > 0:
            stat_val, pval = stats.mannwhitneyu(young, old, alternative='two-sided')
            d = _cohens_d(young, old)
            stat_text_parts.append(f'p = {pval:.4f}')
            if d is not None and not np.isnan(d):
                stat_text_parts.append(f'd = {d:.2f}')
            sig_rows.append({
                'analysis': f'SI_logvel by age (controls, threshold {threshold})',
                'test': 'Mann-Whitney U',
                'group1': group_labels[0], 'group2': group_labels[1],
                'n1': len(young), 'n2': len(old),
                'statistic': stat_val, 'p_value': pval,
                'cohens_d': d if d is not None else np.nan,
            })
        ax.text(0.5, 0.95, '\n'.join(stat_text_parts), transform=ax.transAxes,
                ha='center', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5))

        ax.set_xlabel('Age Group', fontproperties=source_sans if source_sans else None)
        ax.set_ylabel('SI_logvel (AUC of log velocity)', fontproperties=source_sans if source_sans else None)
        ax.set_title(f'SI_logvel by Age (Controls, threshold {threshold})',
                     fontproperties=source_sans if source_sans else None)
        apply_font(ax, source_sans)
        plt.tight_layout()

        save_figure(fig, output_dir,
                    f'stiffness_fig_SI_logvel_by_age_{threshold}_control',
                    minimal_dir=minimal_dir)
        plt.close()

    return sig_rows


def plot_si_by_age_brackets_control(results_df: pd.DataFrame, output_dir: str,
                                     stiffness_col: str = 'SI_logvel_averaged_02_12',
                                     figsize: Tuple[float, float] = (2.4, 2.0),
                                     minimal_dir: Optional[str] = None):
    """Box plot of SI_logvel across multiple age brackets within controls.

    Mirrors the hysteresis multi-age-group box plot: bins
    [0, 30, 50, 60, 70, 100] with labels ['<30', '30-49', '50-59', '60-69', '70+'].
    Pairwise Mann-Whitney U tests are performed for each bracket vs the <30
    reference group, with significance brackets drawn for p < 0.05.

    Args:
        results_df: DataFrame with SI_logvel, Age, and is_healthy.
        output_dir: Directory to save figures.
        stiffness_col: SI column (default SI_logvel_averaged_02_12).
        figsize: Figure size.
        minimal_dir: If set, save minimal copies.

    Returns:
        List of dicts with significance results for each pairwise comparison.
    """
    sig_rows = []
    if stiffness_col not in results_df.columns:
        print(f"Warning: {stiffness_col} not found. Skipping SI age-brackets control plot.")
        return sig_rows

    df = _ensure_healthy_flag(results_df)
    controls = df[(df['is_healthy'] == True) & df[stiffness_col].notna() & df['Age'].notna()].copy()
    if len(controls) < 4:
        print("Warning: Too few control participants for age-bracket SI analysis.")
        return sig_rows

    # Age bins matching hysteresis.py
    bins = [0, 30, 50, 60, 70, 100]
    labels = ['<30', '30-49', '50-59', '60-69', '70+']
    controls['Age_Bracket'] = pd.cut(controls['Age'], bins=bins, labels=labels,
                                     include_lowest=True)

    # Drop brackets with no data and determine order
    present_labels = [lbl for lbl in labels if (controls['Age_Bracket'] == lbl).sum() > 0]
    if len(present_labels) < 2:
        print("Warning: Fewer than 2 age brackets with data. Skipping.")
        return sig_rows

    # 5-shade blue monochromatic palette
    full_palette = create_monochromatic_palette(AGE_BASE_COLOR, n_colors=len(labels))
    full_palette = adjust_brightness_of_colors(full_palette, brightness_scale=0.1)
    palette_hex = [rgb2hex(c) for c in full_palette]
    # Keep only colours for present labels
    color_map = {lbl: palette_hex[i] for i, lbl in enumerate(labels) if lbl in present_labels}
    ordered_colors = [color_map[lbl] for lbl in present_labels]

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=controls, x='Age_Bracket', y=stiffness_col, ax=ax,
                order=present_labels, palette=ordered_colors, width=0.6, fliersize=3)

    # Jittered individual points
    for i, lbl in enumerate(present_labels):
        grp_data = controls[controls['Age_Bracket'] == lbl][stiffness_col]
        x_pos = i + np.random.normal(0, 0.04, len(grp_data))
        ax.scatter(x_pos, grp_data, alpha=0.4, s=20, color='black', zorder=3)

    # Sample-size tick labels
    counts = controls['Age_Bracket'].value_counts()
    tick_labels = [f'{lbl}\n(n={counts.get(lbl, 0)})' for lbl in present_labels]
    ax.set_xticklabels(tick_labels)

    # Pairwise tests vs <30 reference group (matching hysteresis.py pattern)
    reference = '<30'
    if reference in present_labels:
        ref_vals = controls[controls['Age_Bracket'] == reference][stiffness_col]
        ref_pos = present_labels.index(reference)

        significant_comparisons = []
        for lbl in present_labels:
            if lbl == reference:
                continue
            other_vals = controls[controls['Age_Bracket'] == lbl][stiffness_col]
            if len(ref_vals) < 3 or len(other_vals) < 3:
                continue
            stat_val, pval = stats.mannwhitneyu(ref_vals, other_vals, alternative='two-sided')
            d = _cohens_d(ref_vals, other_vals)

            # Significance marker
            if pval < 0.001:
                sig_marker = '***'
            elif pval < 0.01:
                sig_marker = '**'
            elif pval < 0.05:
                sig_marker = '*'
            else:
                sig_marker = None

            sig_rows.append({
                'analysis': f'SI_logvel age brackets (controls, {lbl} vs {reference})',
                'test': 'Mann-Whitney U',
                'group1': reference, 'group2': lbl,
                'n1': len(ref_vals), 'n2': len(other_vals),
                'statistic': stat_val, 'p_value': pval,
                'cohens_d': d if d is not None else np.nan,
            })

            if sig_marker is not None:
                significant_comparisons.append({
                    'group': lbl,
                    'pos_ref': ref_pos,
                    'pos_other': present_labels.index(lbl),
                    'p_value': pval,
                    'sig_marker': sig_marker,
                })

        # Draw significance brackets (matching hysteresis.py lines 1696-1717)
        if significant_comparisons:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            significant_comparisons = sorted(significant_comparisons,
                                             key=lambda c: c['pos_other'])
            for idx, comp in enumerate(significant_comparisons):
                bracket_height = y_max - (y_range * (0.05 + idx * 0.10))
                pos_ref = comp['pos_ref']
                pos_other = comp['pos_other']
                ax.plot([pos_ref, pos_ref, pos_other, pos_other],
                        [bracket_height - (y_range * 0.02), bracket_height,
                         bracket_height, bracket_height - (y_range * 0.02)],
                        'k-', linewidth=0.8)
                ax.text((pos_ref + pos_other) / 2,
                        bracket_height + (y_range * 0.01),
                        comp['sig_marker'],
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Age Group', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel('SI_logvel (AUC of log velocity)',
                  fontproperties=source_sans if source_sans else None)
    ax.set_title('SI_logvel by Age Brackets (Controls)',
                 fontproperties=source_sans if source_sans else None)
    apply_font(ax, source_sans)
    plt.tight_layout()

    save_figure(fig, output_dir,
                'stiffness_fig_SI_logvel_by_age_brackets_control',
                minimal_dir=minimal_dir)
    plt.close()

    return sig_rows


def plot_bp_by_age_group_control(results_df: pd.DataFrame, output_dir: str,
                                  bp_cols: list = None,
                                  age_thresholds: list = None,
                                  figsize: Tuple[float, float] = (2.4, 2.0),
                                  minimal_dir: Optional[str] = None):
    """Box plots of blood pressure comparing age groups within the control cohort.

    Args:
        results_df: DataFrame with SYS_BP / DIA_BP, Age, and is_healthy.
        output_dir: Directory to save figures.
        bp_cols: List of BP columns to plot (default ['SYS_BP', 'DIA_BP']).
        age_thresholds: List of integer thresholds (default [30, 50, 60]).
        figsize: Figure size.
        minimal_dir: If set, save minimal copies.

    Returns:
        List of dicts with significance results for each threshold/BP column.
    """
    sig_rows = []
    if bp_cols is None:
        bp_cols = ['SYS_BP', 'DIA_BP']
    if age_thresholds is None:
        age_thresholds = [30, 50, 60]

    df = _ensure_healthy_flag(results_df)
    controls = df[(df['is_healthy'] == True) & df['Age'].notna()].copy()
    if len(controls) < 4:
        print("Warning: Too few control participants for age-group BP analysis.")
        return sig_rows

    palette_hex = _make_binary_palette(BP_BASE_COLOR)

    for bp_col in bp_cols:
        if bp_col not in controls.columns:
            print(f"Warning: {bp_col} not found. Skipping.")
            continue
        bp_data = controls[controls[bp_col].notna()].copy()

        bp_label = 'Systolic BP (mmHg)' if bp_col == 'SYS_BP' else 'Diastolic BP (mmHg)'
        bp_short = 'SBP' if bp_col == 'SYS_BP' else 'DBP'

        for threshold in age_thresholds:
            bp_data['Age_Group'] = np.where(bp_data['Age'] < threshold,
                                            f'<{threshold}', f'\u2265{threshold}')
            group_labels = [f'<{threshold}', f'\u2265{threshold}']

            fig, ax = plt.subplots(figsize=figsize)
            sns.boxplot(data=bp_data, x='Age_Group', y=bp_col, ax=ax,
                        order=group_labels, palette=palette_hex, width=0.6)

            for i, grp in enumerate(group_labels):
                grp_vals = bp_data[bp_data['Age_Group'] == grp][bp_col]
                x_pos = i + np.random.normal(0, 0.04, len(grp_vals))
                ax.scatter(x_pos, grp_vals, alpha=0.4, s=20, color='black', zorder=3)

            young = bp_data[bp_data['Age_Group'] == group_labels[0]][bp_col]
            old = bp_data[bp_data['Age_Group'] == group_labels[1]][bp_col]
            stat_text_parts = [f'n = {len(young)}, {len(old)}']
            if len(young) > 0 and len(old) > 0:
                stat_val, pval = stats.mannwhitneyu(young, old, alternative='two-sided')
                d = _cohens_d(young, old)
                stat_text_parts.append(f'p = {pval:.4f}')
                if d is not None and not np.isnan(d):
                    stat_text_parts.append(f'd = {d:.2f}')
                sig_rows.append({
                    'analysis': f'{bp_short} by age (controls, threshold {threshold})',
                    'test': 'Mann-Whitney U',
                    'group1': group_labels[0], 'group2': group_labels[1],
                    'n1': len(young), 'n2': len(old),
                    'statistic': stat_val, 'p_value': pval,
                    'cohens_d': d if d is not None else np.nan,
                })
            ax.text(0.5, 0.95, '\n'.join(stat_text_parts), transform=ax.transAxes,
                    ha='center', va='top', fontsize=6,
                    fontproperties=source_sans if source_sans else None,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5))

            ax.set_xlabel('Age Group', fontproperties=source_sans if source_sans else None)
            ax.set_ylabel(bp_label, fontproperties=source_sans if source_sans else None)
            ax.set_title(f'{bp_short} by Age (Controls, threshold {threshold})',
                         fontproperties=source_sans if source_sans else None)
            apply_font(ax, source_sans)
            plt.tight_layout()

            save_figure(fig, output_dir,
                        f'stiffness_fig_{bp_short}_by_age_{threshold}_control',
                        minimal_dir=minimal_dir)
            plt.close()

    return sig_rows


def plot_bp_by_disease_group(results_df: pd.DataFrame, output_dir: str,
                              bp_cols: list = None,
                              figsize: Tuple[float, float] = (2.4, 2.0),
                              minimal_dir: Optional[str] = None):
    """Box plots of blood pressure comparing control vs any-disease group.

    Generates two versions for each BP column:
    * All ages
    * Age >= 50 only

    Args:
        results_df: DataFrame with SYS_BP / DIA_BP, Age, and is_healthy.
        output_dir: Directory to save figures.
        bp_cols: List of BP columns (default ['SYS_BP', 'DIA_BP']).
        figsize: Figure size.
        minimal_dir: If set, save minimal copies.

    Returns:
        List of dicts with significance results for each BP column / subset.
    """
    sig_rows = []
    if bp_cols is None:
        bp_cols = ['SYS_BP', 'DIA_BP']

    df = _ensure_healthy_flag(results_df)
    palette_hex = _make_binary_palette(DISEASE_BASE_COLOR)
    group_labels = ['Control', 'Disease']

    subsets = [
        ('all', df.copy()),
    ]
    if 'Age' in df.columns:
        subsets.append(('50plus', df[df['Age'] >= 50].copy()))

    for bp_col in bp_cols:
        if bp_col not in df.columns:
            print(f"Warning: {bp_col} not found. Skipping disease-group BP plot.")
            continue
        bp_label = 'Systolic BP (mmHg)' if bp_col == 'SYS_BP' else 'Diastolic BP (mmHg)'
        bp_short = 'SBP' if bp_col == 'SYS_BP' else 'DBP'

        for tag, subset in subsets:
            sub = subset[subset[bp_col].notna()].copy()
            sub['Group'] = np.where(sub['is_healthy'] == True, 'Control', 'Disease')

            if sub['Group'].nunique() < 2:
                print(f"Warning: fewer than 2 groups for {bp_short} ({tag}). Skipping.")
                continue

            fig, ax = plt.subplots(figsize=figsize)
            sns.boxplot(data=sub, x='Group', y=bp_col, ax=ax,
                        order=group_labels, palette=palette_hex, width=0.6)

            for i, grp in enumerate(group_labels):
                grp_vals = sub[sub['Group'] == grp][bp_col]
                x_pos = i + np.random.normal(0, 0.04, len(grp_vals))
                ax.scatter(x_pos, grp_vals, alpha=0.4, s=20, color='black', zorder=3)

            ctrl = sub[sub['Group'] == 'Control'][bp_col]
            dis = sub[sub['Group'] == 'Disease'][bp_col]
            stat_text_parts = [f'n = {len(ctrl)}, {len(dis)}']
            if len(ctrl) > 0 and len(dis) > 0:
                stat_val, pval = stats.mannwhitneyu(ctrl, dis, alternative='two-sided')
                d = _cohens_d(ctrl, dis)
                stat_text_parts.append(f'p = {pval:.4f}')
                if d is not None and not np.isnan(d):
                    stat_text_parts.append(f'd = {d:.2f}')
                tag_label = 'all ages' if tag == 'all' else 'age>=50'
                sig_rows.append({
                    'analysis': f'{bp_short} control vs disease ({tag_label})',
                    'test': 'Mann-Whitney U',
                    'group1': 'Control', 'group2': 'Disease',
                    'n1': len(ctrl), 'n2': len(dis),
                    'statistic': stat_val, 'p_value': pval,
                    'cohens_d': d if d is not None else np.nan,
                })
            ax.text(0.5, 0.95, '\n'.join(stat_text_parts), transform=ax.transAxes,
                    ha='center', va='top', fontsize=6,
                    fontproperties=source_sans if source_sans else None,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5))

            suffix = '' if tag == 'all' else f'_{tag}'
            title_suffix = '' if tag == 'all' else ' (Age \u226550)'
            ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
            ax.set_ylabel(bp_label, fontproperties=source_sans if source_sans else None)
            ax.set_title(f'{bp_short}: Control vs Disease{title_suffix}',
                         fontproperties=source_sans if source_sans else None)
            apply_font(ax, source_sans)
            plt.tight_layout()

            save_figure(fig, output_dir,
                        f'stiffness_fig_{bp_short}_control_vs_disease{suffix}',
                        minimal_dir=minimal_dir)
            plt.close()

    return sig_rows


def plot_si_by_disease_group(results_df: pd.DataFrame, output_dir: str,
                             stiffness_col: str = 'SI_logvel_averaged_02_12',
                             figsize: Tuple[float, float] = (2.4, 2.0),
                             minimal_dir: Optional[str] = None):
    """Box plots of SI_logvel comparing control vs any-disease group.

    Generates two versions:
    * All ages
    * Age >= 50 only (if Age is available)

    Args:
        results_df: DataFrame with SI_logvel, Age, and is_healthy.
        output_dir: Directory to save figures.
        stiffness_col: SI column to plot (default SI_logvel_averaged_02_12).
        figsize: Figure size.
        minimal_dir: If set, save minimal copies.

    Returns:
        List of dicts with significance results for each subset.
    """
    sig_rows = []
    if stiffness_col not in results_df.columns:
        print(f"Warning: {stiffness_col} not found. Skipping SI_logvel disease-group plot.")
        return sig_rows

    df = _ensure_healthy_flag(results_df)
    palette_hex = _make_binary_palette(DISEASE_BASE_COLOR)
    group_labels = ['Control', 'Disease']

    subsets = [
        ('all', df.copy()),
    ]
    if 'Age' in df.columns:
        subsets.append(('50plus', df[df['Age'] >= 50].copy()))

    for tag, subset in subsets:
        sub = subset[subset[stiffness_col].notna()].copy()
        sub['Group'] = np.where(sub['is_healthy'] == True, 'Control', 'Disease')

        if sub['Group'].nunique() < 2:
            print(f"Warning: fewer than 2 groups for SI_logvel ({tag}). Skipping.")
            continue

        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(
            data=sub,
            x='Group',
            y=stiffness_col,
            ax=ax,
            order=group_labels,
            palette=palette_hex,
            width=0.6,
        )

        for i, grp in enumerate(group_labels):
            grp_vals = sub[sub['Group'] == grp][stiffness_col]
            x_pos = i + np.random.normal(0, 0.04, len(grp_vals))
            ax.scatter(x_pos, grp_vals, alpha=0.4, s=20, color='black', zorder=3)

        ctrl = sub[sub['Group'] == 'Control'][stiffness_col]
        dis = sub[sub['Group'] == 'Disease'][stiffness_col]
        stat_text_parts = [f'n = {len(ctrl)}, {len(dis)}']
        if len(ctrl) > 0 and len(dis) > 0:
            stat_val, pval = stats.mannwhitneyu(ctrl, dis, alternative='two-sided')
            d = _cohens_d(ctrl, dis)
            stat_text_parts.append(f'p = {pval:.4f}')
            if d is not None and not np.isnan(d):
                stat_text_parts.append(f'd = {d:.2f}')
            tag_label = 'all ages' if tag == 'all' else 'age>=50'
            sig_rows.append({
                'analysis': f'SI_logvel control vs disease ({tag_label})',
                'test': 'Mann-Whitney U',
                'group1': 'Control', 'group2': 'Disease',
                'n1': len(ctrl), 'n2': len(dis),
                'statistic': stat_val, 'p_value': pval,
                'cohens_d': d if d is not None else np.nan,
            })

        ax.text(0.5, 0.95, '\n'.join(stat_text_parts), transform=ax.transAxes,
                ha='center', va='top', fontsize=6,
                fontproperties=source_sans if source_sans else None,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                          edgecolor='black', linewidth=0.5))

        suffix = '' if tag == 'all' else f'_{tag}'
        title_suffix = '' if tag == 'all' else ' (Age \u226550)'
        ax.set_xlabel('Group', fontproperties=source_sans if source_sans else None)
        ax.set_ylabel('SI_logvel (AUC of log velocity)',
                      fontproperties=source_sans if source_sans else None)
        ax.set_title(f'SI_logvel: Control vs Disease{title_suffix}',
                     fontproperties=source_sans if source_sans else None)
        apply_font(ax, source_sans)
        plt.tight_layout()

        save_figure(fig, output_dir,
                    f'stiffness_fig_SI_logvel_control_vs_disease{suffix}',
                    minimal_dir=minimal_dir)
        plt.close()

    return sig_rows


def plot_si_correlation_control(results_df: pd.DataFrame, output_dir: str,
                                 stiffness_col: str = 'SI_logvel_averaged_02_12',
                                 figsize: Tuple[float, float] = (2.4, 2.0),
                                 minimal_dir: Optional[str] = None):
    """Scatter/regression of SI_logvel vs Age, SYS_BP, DIA_BP within controls.

    Args:
        results_df: DataFrame with SI_logvel, Age, SYS_BP, DIA_BP, is_healthy.
        output_dir: Directory to save figures.
        stiffness_col: SI column (default SI_logvel_averaged_02_12).
        figsize: Figure size.
        minimal_dir: If set, save minimal copies.

    Returns:
        List of dicts with correlation/regression results.
    """
    sig_rows = []
    if stiffness_col not in results_df.columns:
        print(f"Warning: {stiffness_col} not found. Skipping SI correlation control plots.")
        return sig_rows

    df = _ensure_healthy_flag(results_df)
    controls = df[(df['is_healthy'] == True) & df[stiffness_col].notna()].copy()

    x_vars = [
        ('Age', 'Age (years)'),
        ('SYS_BP', 'Systolic BP (mmHg)'),
        ('DIA_BP', 'Diastolic BP (mmHg)'),
    ]

    for x_col, x_label in x_vars:
        if x_col not in controls.columns:
            continue
        plot_data = controls[controls[x_col].notna()].copy()
        if len(plot_data) < 4:
            continue

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(plot_data[x_col], plot_data[stiffness_col],
                   alpha=0.6, s=20, color=CONTROL_COLOR, zorder=3)

        x = plot_data[x_col].values
        y = plot_data[stiffness_col].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
            x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7,
                    label=f'R\u00b2 = {r_value**2:.3f}')
            ax.text(0.05, 0.95,
                    f'n = {int(np.sum(valid))}\nR = {r_value:.3f}\np = {p_value:.4f}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=6,
                    fontproperties=source_sans if source_sans else None,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5))
            sig_rows.append({
                'analysis': f'SI_logvel vs {x_col} (controls)',
                'test': 'Linear regression',
                'group1': 'SI_logvel', 'group2': x_col,
                'n1': int(np.sum(valid)), 'n2': int(np.sum(valid)),
                'statistic': r_value,
                'p_value': p_value,
                'cohens_d': np.nan,
                'R': r_value,
                'R_squared': r_value ** 2,
                'slope': slope,
            })

        ax.set_xlabel(x_label, fontproperties=source_sans if source_sans else None)
        ax.set_ylabel('SI_logvel (AUC of log velocity)',
                      fontproperties=source_sans if source_sans else None)
        ax.set_title(f'SI_logvel vs {x_col} (Controls)',
                     fontproperties=source_sans if source_sans else None)
        ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
        ax.grid(True, alpha=0.3)
        apply_font(ax, source_sans)
        plt.tight_layout()

        save_figure(fig, output_dir,
                    f'stiffness_fig_SI_logvel_vs_{x_col}_control',
                    minimal_dir=minimal_dir)
        plt.close()

    return sig_rows


def plot_bp_si_correlation_all(results_df: pd.DataFrame, output_dir: str,
                                stiffness_col: str = 'SI_logvel_averaged_02_12',
                                figsize: Tuple[float, float] = (2.4, 2.0),
                                minimal_dir: Optional[str] = None):
    """Scatter/regression plots coloured by health group for all participants.

    Generates:
    * SI_logvel vs SYS_BP
    * SI_logvel vs Age
    * SYS_BP vs Age

    Args:
        results_df: DataFrame with SI_logvel, Age, SYS_BP, is_healthy.
        output_dir: Directory to save figures.
        stiffness_col: SI column (default SI_logvel_averaged_02_12).
        figsize: Figure size.
        minimal_dir: If set, save minimal copies.

    Returns:
        List of dicts with correlation/regression results.
    """
    sig_rows = []
    df = _ensure_healthy_flag(results_df)
    palette_hex = _make_binary_palette(DISEASE_BASE_COLOR)
    group_colors = {'Control': palette_hex[0], 'Disease': palette_hex[1]}
    df['HealthGroup'] = np.where(df['is_healthy'] == True, 'Control', 'Disease')

    pairs = [
        (stiffness_col, 'SYS_BP', 'SI_logvel (AUC of log velocity)', 'Systolic BP (mmHg)'),
        (stiffness_col, 'Age', 'SI_logvel (AUC of log velocity)', 'Age (years)'),
        ('SYS_BP', 'Age', 'Systolic BP (mmHg)', 'Age (years)'),
    ]

    for y_col, x_col, y_label, x_label in pairs:
        if y_col not in df.columns or x_col not in df.columns:
            continue
        plot_data = df[df[y_col].notna() & df[x_col].notna()].copy()
        if len(plot_data) < 4:
            continue

        fig, ax = plt.subplots(figsize=figsize)

        for grp in ['Control', 'Disease']:
            gd = plot_data[plot_data['HealthGroup'] == grp]
            ax.scatter(gd[x_col], gd[y_col], alpha=0.6, s=20,
                       color=group_colors[grp], label=grp, zorder=3)

        # Overall regression line
        x = plot_data[x_col].values
        y = plot_data[y_col].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
            x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'k--', linewidth=1, alpha=0.7,
                    label=f'R\u00b2 = {r_value**2:.3f}')
            ax.text(0.05, 0.95,
                    f'n = {int(np.sum(valid))}\nR = {r_value:.3f}\np = {p_value:.4f}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=6,
                    fontproperties=source_sans if source_sans else None,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5))
            y_short = y_col.replace('SI_logvel_averaged_02_12', 'SI_logvel')
            sig_rows.append({
                'analysis': f'{y_short} vs {x_col} (all participants)',
                'test': 'Linear regression',
                'group1': y_short, 'group2': x_col,
                'n1': int(np.sum(valid)), 'n2': int(np.sum(valid)),
                'statistic': r_value,
                'p_value': p_value,
                'cohens_d': np.nan,
                'R': r_value,
                'R_squared': r_value ** 2,
                'slope': slope,
            })

        y_col_short = y_col.replace('SI_logvel_averaged_02_12', 'SI_logvel')
        ax.set_xlabel(x_label, fontproperties=source_sans if source_sans else None)
        ax.set_ylabel(y_label, fontproperties=source_sans if source_sans else None)
        ax.set_title(f'{y_col_short} vs {x_col} (All)',
                     fontproperties=source_sans if source_sans else None)
        ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
        ax.grid(True, alpha=0.3)
        apply_font(ax, source_sans)
        plt.tight_layout()

        fname_y = 'SI_logvel' if stiffness_col in y_col else y_col
        fname_x = x_col
        save_figure(fig, output_dir,
                    f'stiffness_fig_{fname_y}_vs_{fname_x}_all',
                    minimal_dir=minimal_dir)
        plt.close()

    return sig_rows


def _ensure_diabetes_bool(results_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Diabetes column is boolean for group comparisons."""
    df = results_df.copy()
    if 'Diabetes' not in df.columns:
        return df
    if df['Diabetes'].dtype == object or df['Diabetes'].dtype.name == 'str':
        df['Diabetes'] = df['Diabetes'].astype(str).str.upper() == 'TRUE'
    return df


def collect_significance_results(results_df: pd.DataFrame,
                                 results_df_log: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Collect all significance test results into a single DataFrame for export.

    Includes: group comparisons (Mann-Whitney U), continuous associations
    (Pearson linear regression), and age-adjusted OLS (group p-value, age p-value).
    If results_df_log is provided and has Age/Diabetes, also adds age-adjusted
    results for SI_logvel_* columns.
    Saves nothing; returns the DataFrame (caller saves to CSV).

    Args:
        results_df: Main stiffness results (raw velocity, contains SI_log1p_* columns).
        results_df_log: Optional log-velocity stiffness results (contains SI_logvel_* columns).

    Returns:
        DataFrame with columns: analysis, test, statistic, p_value, n, n_control, n_diabetic
        (n_control/n_diabetic only for group comparisons).
    """
    from src.analysis.stiffness_coeff import age_adjusted_analysis

    rows = []
    df = _ensure_diabetes_bool(results_df)

    def _mw_group(df_context: pd.DataFrame, var_col: str, label: str) -> None:
        control = df_context[(df_context['Diabetes'] == False) & (df_context[var_col].notna())][var_col]
        diabetic = df_context[(df_context['Diabetes'] == True) & (df_context[var_col].notna())][var_col]
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
        _mw_group(df, 'MAP', 'MAP by group (Control vs Diabetic)')
    if 'SYS_BP' in df.columns:
        _mw_group(df, 'SYS_BP', 'SBP by group (Control vs Diabetic)')

    for col in ['stiffness_coeff_up_04_12', 'stiffness_coeff_up_02_12',
                'stiffness_coeff_averaged_04_12', 'stiffness_coeff_averaged_02_12']:
        if col in df.columns:
            _mw_group(df, col, f'{col} by group (Control vs Diabetic)')

    if 'P50' in df.columns:
        _mw_group(df, 'P50', 'P50 by group (Control vs Diabetic)')
    if 'EV_lin' in df.columns:
        _mw_group(df, 'EV_lin', 'EV_lin by group (Control vs Diabetic)')
    if 'composite_stiffness' in df.columns:
        _mw_group(df, 'composite_stiffness',
                  'composite_stiffness by group (Control vs Diabetic)')

    for log_col in ['SI_log1p_up_04_12', 'SI_log1p_up_02_12',
                    'SI_log1p_averaged_04_12', 'SI_log1p_averaged_02_12',
                    'SI_log1p_composite']:
        if log_col in df.columns:
            _mw_group(df, log_col, f'{log_col} by group (Control vs Diabetic)')

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

    # Age-adjusted results for log-velocity AUC data (SI_logvel_*)
    if results_df_log is not None and 'Age' in results_df_log.columns and 'Diabetes' in results_df_log.columns:
        df_log = _ensure_diabetes_bool(results_df_log)
        # Group comparisons for SI_logvel columns
        for logvel_col in ['SI_logvel_up_04_12', 'SI_logvel_up_02_12',
                           'SI_logvel_averaged_04_12', 'SI_logvel_averaged_02_12']:
            if logvel_col in df_log.columns:
                _mw_group(df_log, logvel_col, f'{logvel_col} by group (Control vs Diabetic)')

        # Age-adjusted analysis for SI_logvel columns
        for stiffness_col in ['SI_logvel_averaged_04_12', 'SI_logvel_averaged_02_12']:
            if stiffness_col not in df_log.columns:
                continue
            res = age_adjusted_analysis(df_log, stiffness_col, group_col='Diabetes')
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
                               figsize: Tuple[float, float] = (2.4, 2.0),
                               minimal_dir: Optional[str] = None):
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
    
    # Determine range label and metric type
    is_logvel = stiffness_col.startswith('SI_logvel_')
    is_log1p = stiffness_col.startswith('SI_log1p_')
    if '02_12' in stiffness_col:
        range_label = '0.2-1.2'
    else:
        range_label = '0.4-1.2'
    
    # Set labels based on metric type
    if is_logvel:
        ylabel = 'SI_logvel (AUC of log velocity)'
        title = f'Age-Adjusted (SI_logvel, {range_label} psi)'
    elif is_log1p:
        ylabel = 'SI_log1p (log(AUC + 1))'
        title = f'Age-Adjusted (SI_log1p, {range_label} psi)'
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
    save_figure(fig, output_dir, filename, minimal_dir=minimal_dir)
    plt.close()


def _plot_age_adjusted_panel(ax, df_plot: pd.DataFrame, stiffness_col: str,
                             title: str, ylabel: str) -> None:
    """Plot age-adjusted scatter with group lines and p-value annotation."""
    from src.analysis.stiffness_coeff import age_adjusted_analysis

    df_plot = df_plot[[stiffness_col, 'Diabetes', 'Age']].dropna().copy()
    if df_plot.empty:
        return
    df_plot['Group'] = df_plot['Diabetes'].map({True: 'Diabetic', False: 'Control'})

    age_results = age_adjusted_analysis(df_plot, stiffness_col, group_col='Diabetes')
    if 'error' in age_results:
        return

    for group, color in [('Control', CONTROL_COLOR), ('Diabetic', DIABETES_COLOR)]:
        group_data = df_plot[df_plot['Group'] == group]
        ax.scatter(group_data['Age'], group_data[stiffness_col],
                   alpha=0.6, s=20, color=color, label=group, zorder=3)
        if len(group_data) > 2:
            x = group_data['Age'].values
            y = group_data[stiffness_col].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, '--', linewidth=1, alpha=0.7, color=color)

    ax.text(0.05, 0.95,
            f'Age-adjusted p = {age_results["group_pvalue"]:.4f}\n'
            f'R² = {age_results["r_squared"]:.3f}',
            transform=ax.transAxes, ha='left', va='top', fontsize=6,
            fontproperties=source_sans if source_sans else None,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Age (years)', fontproperties=source_sans if source_sans else None)
    ax.set_ylabel(ylabel, fontproperties=source_sans if source_sans else None)
    ax.set_title(title, fontproperties=source_sans if source_sans else None)
    ax.legend(loc='best', fontsize=5, prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)
    apply_font(ax, source_sans)


def plot_log_metric_comparison_age_adjusted(results_df: pd.DataFrame,
                                            results_df_log: pd.DataFrame,
                                            output_dir: str,
                                            range_label: str = '04_12',
                                            figsize: Tuple[float, float] = (4.8, 2.0),
                                            minimal_dir: Optional[str] = None):
    """Compare age-adjusted trends for SI_log1p vs SI_logvel.

    Args:
        results_df: Main stiffness results (contains SI_log1p_* columns).
        results_df_log: Log-velocity stiffness results (contains SI_logvel_* columns).
        output_dir: Directory to save figures.
        range_label: Pressure range label ('04_12' or '02_12').
        figsize: Figure size for the side-by-side panels.
        minimal_dir: If set, save a no-title/no-legend copy here.
    """
    if results_df_log is None:
        return

    # Build a merged frame that has both metrics + Age/Diabetes
    keep_cols = ['Participant', 'Age', 'Diabetes']
    log1p_col = f'SI_log1p_averaged_{range_label}'
    if log1p_col in results_df.columns:
        keep_cols.append(log1p_col)
    df_merge = results_df[keep_cols].merge(
        results_df_log, on='Participant', how='inner', suffixes=('', '_logdf')
    )
    # Resolve any duplicate Age/Diabetes columns
    for c in ['Age', 'Diabetes']:
        dup = f'{c}_logdf'
        if dup in df_merge.columns:
            df_merge.drop(columns=[dup], inplace=True)

    metrics = []
    if log1p_col in df_merge.columns:
        metrics.append((log1p_col, 'SI_log1p', 'log1p(AUC raw vel)'))
    logvel_col = f'SI_logvel_averaged_{range_label}'
    if logvel_col in df_merge.columns:
        metrics.append((logvel_col, 'SI_logvel', 'AUC of log velocity'))

    if len(metrics) == 0:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
    for idx, (col, ylabel, title) in enumerate(metrics):
        _plot_age_adjusted_panel(axes[0, idx], df_merge, col, title, ylabel)

    plt.tight_layout()
    filename = f'stiffness_fig_log_metric_compare_age_adjusted_{range_label}'
    save_figure(fig, output_dir, filename, minimal_dir=minimal_dir)
    plt.close()


def _cohens_d(control: pd.Series, diabetic: pd.Series) -> Optional[float]:
    """Compute Cohen's d for two groups."""
    if len(control) < 2 or len(diabetic) < 2:
        return np.nan
    n1, n2 = len(control), len(diabetic)
    s1, s2 = np.var(control, ddof=1), np.var(diabetic, ddof=1)
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    if pooled <= 0:
        return np.nan
    return float((np.mean(diabetic) - np.mean(control)) / np.sqrt(pooled))


def _safe_corr(x: pd.Series, y: pd.Series) -> Optional[float]:
    """Compute Pearson correlation if data are sufficient."""
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3:
        return np.nan
    return float(valid.corr().iloc[0, 1])


def build_log_metric_comparison_table(results_df: pd.DataFrame,
                                      results_df_log: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Build a comparison table: SI_log1p vs SI_logvel for each pressure range.

    Args:
        results_df: Main stiffness results (contains SI_log1p_* columns).
        results_df_log: Log-velocity stiffness results (contains SI_logvel_* columns).

    Returns:
        DataFrame with effect sizes, p-values, and velocity-bias flags.
    """
    from src.analysis.stiffness_coeff import age_adjusted_analysis

    rows = []
    for range_label in ['04_12', '02_12']:
        metric_specs = [
            ('SI_log1p', results_df, f'SI_log1p_averaged_{range_label}', 'raw'),
        ]
        if results_df_log is not None:
            metric_specs.extend([
                ('SI_logvel', results_df_log, f'SI_logvel_averaged_{range_label}', 'log'),
            ])

        for metric_name, df, col, source in metric_specs:
            if col not in df.columns or 'Diabetes' not in df.columns:
                continue
            df_metric = _ensure_diabetes_bool(df)
            control = df_metric[(df_metric['Diabetes'] == False) & (df_metric[col].notna())][col]
            diabetic = df_metric[(df_metric['Diabetes'] == True) & (df_metric[col].notna())][col]

            stat, pval = (np.nan, np.nan)
            if len(control) > 0 and len(diabetic) > 0:
                stat, pval = stats.mannwhitneyu(control, diabetic, alternative='two-sided')

            age_coef = age_p = r2 = n = np.nan
            if 'Age' in df_metric.columns:
                res = age_adjusted_analysis(df_metric, col, group_col='Diabetes')
                if 'error' not in res:
                    age_coef = res.get('group_coef')
                    age_p = res.get('group_pvalue')
                    r2 = res.get('r_squared')
                    n = res.get('n')

            corr_04 = corr_12 = np.nan
            if 'velocity_04' in df_metric.columns and 'velocity_12' in df_metric.columns:
                corr_04 = _safe_corr(df_metric[col], df_metric['velocity_04'])
                corr_12 = _safe_corr(df_metric[col], df_metric['velocity_12'])
            delta = corr_04 - corr_12 if pd.notna(corr_04) and pd.notna(corr_12) else np.nan
            if pd.isna(delta):
                bias = None
            elif delta > 0.05:
                bias = 'low'
            elif delta < -0.05:
                bias = 'high'
            else:
                bias = 'balanced'

            rows.append({
                'metric': metric_name,
                'range': range_label,
                'source': source,
                'n_control': len(control),
                'n_diabetic': len(diabetic),
                'mean_control': float(np.mean(control)) if len(control) else np.nan,
                'mean_diabetic': float(np.mean(diabetic)) if len(diabetic) else np.nan,
                'median_control': float(np.median(control)) if len(control) else np.nan,
                'median_diabetic': float(np.median(diabetic)) if len(diabetic) else np.nan,
                'cohens_d': _cohens_d(control, diabetic),
                'mwu_stat': stat,
                'mwu_p': pval,
                'age_adj_group_coef': age_coef,
                'age_adj_group_p': age_p,
                'age_adj_r2': r2,
                'corr_velocity_04': corr_04,
                'corr_velocity_12': corr_12,
                'velocity_bias': bias,
            })

    return pd.DataFrame(rows)


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
    
    # Create output directory and minimal (no annotations) subfolder
    output_dir = os.path.join(cap_flow_path, 'results', 'Stiffness', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    minimal_dir = os.path.join(output_dir, 'no_annotations')
    os.makedirs(minimal_dir, exist_ok=True)

    # Generate main figures
    print("\nGenerating main figures...")
    plot_map_by_group(results_df, output_dir, minimal_dir=minimal_dir)
    plot_sbp_by_group(results_df, output_dir, minimal_dir=minimal_dir)
    
    # Generate stiffness by group plots for different methods/ranges
    for method in ['up', 'averaged']:
        for range_label in ['04_12', '02_12']:
            col = f'stiffness_coeff_{method}_{range_label}'
            if col in results_df.columns:
                plot_stiffness_by_group(results_df, output_dir, stiffness_col=col, minimal_dir=minimal_dir)
    
    # Generate regression plots
    print("\nGenerating regression plots...")
    for method in ['up', 'averaged']:
        for range_label in ['04_12', '02_12']:
            col = f'stiffness_coeff_{method}_{range_label}'
            if col in results_df.columns:
                plot_stiffness_vs_age(results_df, output_dir, stiffness_col=col, minimal_dir=minimal_dir)
                plot_stiffness_vs_map(results_df, output_dir, stiffness_col=col, minimal_dir=minimal_dir)
                plot_stiffness_vs_sbp(results_df, output_dir, stiffness_col=col, minimal_dir=minimal_dir)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    for range_label in ['04_12', '02_12']:
        compare_up_vs_averaged(results_df, output_dir, range_label=range_label, minimal_dir=minimal_dir)
    
    for method in ['up', 'averaged']:
        compare_ranges(results_df, output_dir, method=method, minimal_dir=minimal_dir)
    
    # Generate supplement figures for secondary metrics
    print("\nGenerating supplement figures...")
    if 'P50' in results_df.columns:
        plot_p50_vs_stiffness(results_df, output_dir, minimal_dir=minimal_dir)
        plot_p50_by_group(results_df, output_dir, minimal_dir=minimal_dir)
    if 'EV_lin' in results_df.columns:
        plot_ev_lin_vs_stiffness(results_df, output_dir, minimal_dir=minimal_dir)
        plot_ev_lin_by_group(results_df, output_dir, minimal_dir=minimal_dir)
    
    # Generate composite stiffness plots
    print("\nGenerating composite stiffness plots...")
    if 'composite_stiffness' in results_df.columns:
        plot_composite_stiffness_by_group(results_df, output_dir, minimal_dir=minimal_dir)
    
    # Generate log-transformed stiffness plots (SI_log1p)
    print("\nGenerating log-transformed stiffness plots...")
    for col in ['SI_log1p_up_04_12', 'SI_log1p_up_02_12',
                'SI_log1p_averaged_04_12', 'SI_log1p_averaged_02_12',
                'SI_log1p_composite']:
        if col in results_df.columns:
            plot_log_stiffness_by_group(results_df, output_dir, stiffness_col=col, minimal_dir=minimal_dir)
    
    # Generate age-adjusted analysis plots
    print("\nGenerating age-adjusted analysis plots...")
    for stiffness_col in ['stiffness_coeff_averaged_04_12', 'stiffness_coeff_averaged_02_12', 'composite_stiffness']:
        if stiffness_col in results_df.columns:
            plot_age_adjusted_analysis(results_df, output_dir, stiffness_col=stiffness_col, minimal_dir=minimal_dir)
    
    # Load log velocity results if present (for plots and for significance CSV)
    results_df_log = None
    log_stiffness_file = os.path.join(cap_flow_path, 'results', 'Stiffness', 'stiffness_coefficients_log.csv')
    if os.path.exists(log_stiffness_file):
        results_df_log = pd.read_csv(log_stiffness_file)
        print(f"Loaded log velocity stiffness results for {len(results_df_log)} participants")
        # Merge all demographic columns needed for downstream analyses
        merge_cols = ['Participant', 'Age', 'Diabetes', 'Hypertension', 'SET',
                      'is_healthy', 'SYS_BP', 'DIA_BP', 'MAP']
        available = [c for c in merge_cols if c in results_df.columns]
        # Only merge columns not already present in the log DataFrame
        cols_to_merge = [c for c in available if c not in results_df_log.columns or c == 'Participant']
        if len(cols_to_merge) > 1:  # more than just 'Participant'
            results_df_log = results_df_log.merge(
                results_df[cols_to_merge].drop_duplicates(subset='Participant'),
                on='Participant', how='left'
            )
            print(f"Merged demographic columns into log results: {[c for c in cols_to_merge if c != 'Participant']}")
        # Ensure is_healthy flag exists
        results_df_log = _ensure_healthy_flag(results_df_log)
        if 'Age' not in results_df_log.columns or 'Diabetes' not in results_df_log.columns:
            print("Warning: Age and Diabetes columns not found. Skipping log velocity age-adjusted plots.")
            results_df_log = None

    # Generate side-by-side comparisons for log-related metrics
    print("\nGenerating log metric comparison plots...")
    if results_df_log is not None:
        for range_label in ['04_12', '02_12']:
            plot_log_metric_comparison_by_group(
                results_df, results_df_log, output_dir, range_label=range_label, minimal_dir=minimal_dir
            )
            plot_log_metric_comparison_age_adjusted(
                results_df, results_df_log, output_dir, range_label=range_label, minimal_dir=minimal_dir
            )

    # Generate BP vs SI comparison by condition (Diabetes and Hypertension)
    print("\nGenerating BP vs SI condition comparison plot...")
    if results_df_log is not None:
        bp_si_sig = plot_bp_si_by_condition(
            results_df, results_df_log, output_dir,
            stiffness_col='SI_logvel_averaged_02_12',
            bp_col='SYS_BP',
            minimal_dir=minimal_dir
        )
        if bp_si_sig:
            print(f"  Generated 2x2 comparison: {len(bp_si_sig)} statistical tests")

    # Generate standalone boxplots for log velocity AUC (SI_logvel)
    print("\nGenerating standalone SI_logvel boxplots...")
    if results_df_log is not None and 'Diabetes' in results_df_log.columns:
        for col in ['SI_logvel_up_04_12', 'SI_logvel_up_02_12',
                     'SI_logvel_averaged_04_12', 'SI_logvel_averaged_02_12']:
            if col in results_df_log.columns:
                plot_log_stiffness_by_group(results_df_log, output_dir, stiffness_col=col, minimal_dir=minimal_dir)
    else:
        print("Warning: Diabetes data not available for log velocity results. Skipping SI_logvel boxplots.")

    # Generate age-adjusted analysis plots for log velocity AUC (SI_logvel)
    print("\nGenerating age-adjusted analysis plots for SI_logvel...")
    if results_df_log is not None and 'Age' in results_df_log.columns and 'Diabetes' in results_df_log.columns:
        for stiffness_col in ['SI_logvel_averaged_04_12', 'SI_logvel_averaged_02_12']:
            if stiffness_col in results_df_log.columns:
                plot_age_adjusted_analysis(results_df_log, output_dir, stiffness_col=stiffness_col, minimal_dir=minimal_dir)
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

    # Collect and save all significance test results (main + log velocity age-adjusted)
    print("\nCollecting significance results...")
    sig_df = collect_significance_results(results_df, results_df_log=results_df_log)
    sig_file = os.path.join(output_dir, 'stiffness_significance.csv')
    sig_df.to_csv(sig_file, index=False)
    print(f"Saved significance results to: {sig_file}")

    # Save log metric comparison table
    print("\nSaving log metric comparison table...")
    comparison_df = build_log_metric_comparison_table(results_df, results_df_log)
    comparison_file = os.path.join(cap_flow_path, 'results', 'Stiffness', 'stiffness_metric_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Saved log metric comparison to: {comparison_file}")

    # ------------------------------------------------------------------
    # Age-group and blood-pressure comparisons (SI_logvel)
    # ------------------------------------------------------------------

    # Ensure is_healthy flag on both DataFrames
    results_df = _ensure_healthy_flag(results_df)
    all_sig_rows = []

    print("\nGenerating SI_logvel age-group box plots (controls only)...")
    si_source = results_df_log if results_df_log is not None else results_df
    all_sig_rows.extend(plot_si_by_age_group_control(si_source, output_dir, minimal_dir=minimal_dir) or [])

    print("\nGenerating SI_logvel age-bracket box plot (controls only)...")
    all_sig_rows.extend(plot_si_by_age_brackets_control(si_source, output_dir, minimal_dir=minimal_dir) or [])

    print("\nGenerating BP age-group box plots (controls only)...")
    # Use main results_df which always has SYS_BP / DIA_BP
    all_sig_rows.extend(plot_bp_by_age_group_control(results_df, output_dir, minimal_dir=minimal_dir) or [])

    print("\nGenerating BP control-vs-disease box plots...")
    all_sig_rows.extend(plot_bp_by_disease_group(results_df, output_dir, minimal_dir=minimal_dir) or [])

    print("\nGenerating SI_logvel control-vs-disease box plots...")
    all_sig_rows.extend(plot_si_by_disease_group(si_source, output_dir, minimal_dir=minimal_dir) or [])

    print("\nGenerating SI_logvel correlation plots (controls only)...")
    all_sig_rows.extend(plot_si_correlation_control(si_source, output_dir, minimal_dir=minimal_dir) or [])

    print("\nGenerating correlation plots (all participants)...")
    all_sig_rows.extend(plot_bp_si_correlation_all(si_source, output_dir, minimal_dir=minimal_dir) or [])

    # Generate hysteresis vs age scatterplots
    print("\nGenerating hysteresis vs age scatterplots...")
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    if os.path.exists(data_filepath):
        raw_df = pd.read_csv(data_filepath)
        hysteresis_df = calculate_velocity_hysteresis(raw_df, use_log_velocity=False)
        # Scatterplots
        all_sig_rows.extend(plot_hysteresis_vs_age_control(hysteresis_df, output_dir, minimal_dir=minimal_dir) or [])
        all_sig_rows.extend(plot_hysteresis_vs_age_all(hysteresis_df, output_dir, minimal_dir=minimal_dir) or [])
        all_sig_rows.extend(plot_hysteresis_vs_age_diabetes(hysteresis_df, output_dir, minimal_dir=minimal_dir) or [])
        all_sig_rows.extend(plot_hysteresis_vs_age_hypertension(hysteresis_df, output_dir, minimal_dir=minimal_dir) or [])
        # Scatterplots of absolute value of hysteresis
        all_sig_rows.extend(plot_abs_hysteresis_vs_age_control(hysteresis_df, output_dir, minimal_dir=minimal_dir) or [])
        all_sig_rows.extend(plot_abs_hysteresis_vs_age_all(hysteresis_df, output_dir, minimal_dir=minimal_dir) or [])
        # Boxplots
        all_sig_rows.extend(plot_hysteresis_boxplot_diabetes(hysteresis_df, output_dir, minimal_dir=minimal_dir) or [])
        all_sig_rows.extend(plot_hysteresis_boxplot_hypertension(hysteresis_df, output_dir, minimal_dir=minimal_dir) or [])
    else:
        print(f"Warning: {data_filepath} not found. Skipping hysteresis plots.")

    # Save significance results from all age-group/BP analyses
    if all_sig_rows:
        age_sig_df = pd.DataFrame(all_sig_rows)
        age_sig_file = os.path.join(output_dir, 'age_group_analysis_significance.csv')
        age_sig_df.to_csv(age_sig_file, index=False)
        print(f"\nSaved age-group analysis significance to: {age_sig_file}")
        print(age_sig_df.to_string())

    print("\nAll plots generated successfully!")
    return 0


if __name__ == "__main__":
    main()

