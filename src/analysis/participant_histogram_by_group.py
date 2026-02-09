"""
Filename: src/analysis/participant_histogram_by_group.py

File for creating histograms of participant age colored by group:
Control, Hypertension, and Diabetes. Uses the same colors as figs_ci.py
(plotting_utils): Control = blue (#1f77b4), Hypertension = red (#d62728),
Diabetes = orange (#ff7f0e). Group assignment is by SET (set01=control,
set02=hypertension, set03=diabetes).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import argparse

from src.config import PATHS, load_source_sans
from src.tools.plotting_utils import (
    create_monochromatic_palette,
    adjust_brightness_of_colors,
)

cap_flow_path = PATHS['cap_flow']
source_sans = load_source_sans()

# Title font: copy of source_sans with a larger size so titles stand out
TITLE_FONTSIZE = 9
if source_sans is not None:
    source_sans_title = source_sans.copy()
    source_sans_title.set_size(TITLE_FONTSIZE)
else:
    source_sans_title = None

# Base colors matching src/analysis/figs_ci.py (via plotting_utils)
COLOR_CONTROL = '#1f77b4'
COLOR_HYPERTENSION = '#d62728'
COLOR_DIABETES = '#ff7f0e'

# Alpha for violin fill to mirror the lighter fill (semi-transparent like the light shade)
VIOLIN_FILL_ALPHA = 0.6

# Palette indices for fill (light) and outline (dark).
# Control uses a darker fill so it is distinguishable in grayscale.
_LIGHT_IDX_CONTROL = 2      # mid-tone  (was 4 = lightest)
_DARK_IDX_CONTROL = 0       # darkest   (was 1)
_LIGHT_IDX_DEFAULT = 4      # lightest
_DARK_IDX_DEFAULT = 1       # dark


def _get_group_colors():
    """Return (light_colors, dark_colors) lists in Control / Hypertension / Diabetes order.

    Control uses a darker shade so the group is visible in grayscale print.
    """
    base_colors = [COLOR_CONTROL, COLOR_HYPERTENSION, COLOR_DIABETES]
    light_idx = [_LIGHT_IDX_CONTROL, _LIGHT_IDX_DEFAULT, _LIGHT_IDX_DEFAULT]
    dark_idx = [_DARK_IDX_CONTROL, _DARK_IDX_DEFAULT, _DARK_IDX_DEFAULT]
    light_colors = []
    dark_colors = []
    for i, base in enumerate(base_colors):
        pal = create_monochromatic_palette(base, n_colors=5)
        pal = adjust_brightness_of_colors(pal, brightness_scale=0.1)
        light_colors.append(pal[light_idx[i]])
        dark_colors.append(pal[dark_idx[i]])
    return light_colors, dark_colors


def _assign_group(row: pd.Series) -> str:
    """Assign group label from SET. set01=Control, set02=Hypertension, set03=Diabetes."""
    s = row.get('SET', None)
    if pd.isna(s):
        return 'Unknown'
    s = str(s).strip().lower()
    if s == 'set01':
        return 'Control'
    if s == 'set02':
        return 'Hypertension'
    if s == 'set03':
        return 'Diabetes'
    return 'Other'


def _get_plot_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """One row per participant with Group from SET; only Control, Hypertension, Diabetes."""
    plot_df = df.dropna(subset=['Age']).copy()
    plot_df = plot_df.drop_duplicates(subset=['Participant'])
    plot_df['Group'] = plot_df.apply(_assign_group, axis=1)
    plot_df = plot_df[plot_df['Group'].isin(['Control', 'Hypertension', 'Diabetes'])]
    return plot_df if len(plot_df) > 0 else None


def create_participant_histogram_by_group(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> None:
    """
    Creates a histogram of participant age colored by Control, Hypertension,
    and Diabetes using the same colors as figs_ci.py.

    Args:
        df: DataFrame with Age and SET columns.
        output_dir: Directory to save the plot (optional).
    """
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'ParticipantHistograms')
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating participant histogram by group...")

    # Standard plot configuration (match age_histogram.py)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 4,
        'lines.linewidth': 0.5,
    })

    plot_df = _get_plot_df(df)
    if plot_df is None:
        print("Error: No valid age data with SET in set01/set02/set03 found.")
        return

    # Use 5-year bins (same as age_histogram.py)
    age_values = plot_df['Age']
    age_min = int(age_values.min())
    age_max = int(age_values.max())
    bin_start = (age_min // 5) * 5
    bin_end = ((age_max // 5) + 1) * 5
    bins = np.arange(bin_start, bin_end + 5, 5)

    # Transparent fills + opaque colored outlines so overlap is readable
    # (semi-transparent fill blends where they overlap; outline stays visible)
    group_order = ['Control', 'Hypertension', 'Diabetes']
    colors = [COLOR_CONTROL, COLOR_HYPERTENSION, COLOR_DIABETES]

    plt.figure(figsize=(4, 2.5))
    for grp, color in zip(group_order, colors):
        subset = plot_df[plot_df['Group'] == grp]['Age']
        if len(subset) == 0:
            continue
        # Draw filled bars with transparent fill, no edge (so overlaps blend)
        plt.hist(
            subset,
            bins=bins,
            alpha=0.3,
            color=color,
            label=f'{grp}',
            edgecolor='none',
        )
    # Redraw each group as step outline (opaque) so each distribution stays visible
    for grp, color in zip(group_order, colors):
        subset = plot_df[plot_df['Group'] == grp]['Age']
        if len(subset) > 0:
            plt.hist(
                subset,
                bins=bins,
                histtype='step',
                color=color,
                linewidth=1.2,
                fill=False,
            )

    # Set title and labels with font handling (match age_histogram.py)
    if source_sans:
        plt.title('Age Distribution by Group', fontproperties=source_sans_title)
        plt.xlabel('Age (years)', fontproperties=source_sans)
        plt.ylabel('Frequency', fontproperties=source_sans)
        plt.legend(prop=source_sans)
    else:
        plt.title('Age Distribution by Group', fontsize=TITLE_FONTSIZE)
        plt.xlabel('Age (years)')
        plt.ylabel('Frequency')
        plt.legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = 'participant_histogram_by_group.png'
    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Histogram saved to: {outpath}")
    for grp in group_order:
        n = len(plot_df[plot_df['Group'] == grp])
        if n > 0:
            ages = plot_df[plot_df['Group'] == grp]['Age']
            print(f"  {grp}: n={n}, age {ages.min():.0f}-{ages.max():.0f}, mean={ages.mean():.1f}")


def create_participant_violin_by_group(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> None:
    """
    Creates violin plots of participant age by group (Control, Hypertension,
    Diabetes) using the same colors as figs_ci.py.

    Args:
        df: DataFrame with Age and SET columns.
        output_dir: Directory to save the plot (optional).
    """
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'ParticipantHistograms')
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating participant violin plot by group...")

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 4,
        'lines.linewidth': 0.5,
    })

    plot_df = _get_plot_df(df)
    if plot_df is None:
        print("Error: No valid age data with SET in set01/set02/set03 found.")
        return

    group_order = ['Control', 'Hypertension', 'Diabetes']
    light_colors, dark_colors = _get_group_colors()

    # Build RGBA tuples for fill (with alpha) and edge (opaque)
    fill_rgba = [(r, g, b, VIOLIN_FILL_ALPHA) for r, g, b in light_colors]
    edge_rgba = [(r, g, b, 1.0) for r, g, b in dark_colors]

    fig, ax = plt.subplots(figsize=(4, 2.5))
    vp = sns.violinplot(
        data=plot_df,
        x='Group',
        y='Age',
        order=group_order,
        palette=fill_rgba,
        linewidth=1.2,
        inner=None,
    )
    # Override edge colors on every PolyCollection that seaborn created
    for i, pc in enumerate(ax.collections):
        if i >= len(group_order):
            break
        pc.set_edgecolor(edge_rgba[i])
        pc.set_linewidth(1.2)
    # Strip plot for individual points (light, small)
    sns.stripplot(
        data=plot_df,
        x='Group',
        y='Age',
        order=group_order,
        color='black',
        size=2.5,
        alpha=0.45,
        jitter=0.1,
        ax=ax,
    )

    if source_sans:
        ax.set_title('Age by Group', fontproperties=source_sans_title)
        ax.set_xlabel('', fontproperties=source_sans)
        ax.set_ylabel('Age (years)', fontproperties=source_sans)
    else:
        ax.set_title('Age by Group', fontsize=TITLE_FONTSIZE)
        ax.set_xlabel('')
        ax.set_ylabel('Age (years)')

    ax.grid(True, alpha=0.3, axis='both')
    plt.tight_layout()

    filename = 'participant_violin_by_group.png'
    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Violin plot saved to: {outpath}")
    for grp in group_order:
        n = len(plot_df[plot_df['Group'] == grp])
        if n > 0:
            ages = plot_df[plot_df['Group'] == grp]['Age']
            print(f"  {grp}: n={n}, age {ages.min():.0f}-{ages.max():.0f}, mean={ages.mean():.1f}")


def create_participant_kde_by_group(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> None:
    """
    Creates smoothed kernel density estimates of participant age by group
    (Control, Hypertension, Diabetes), like the histogram but smooth.
    Uses the same light fill + dark outline colors as the violin plot.

    Args:
        df: DataFrame with Age and SET columns.
        output_dir: Directory to save the plot (optional).
    """
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'ParticipantHistograms')
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating participant KDE (distribution) plot by group...")

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 4,
        'lines.linewidth': 0.5,
    })

    plot_df = _get_plot_df(df)
    if plot_df is None:
        print("Error: No valid age data with SET in set01/set02/set03 found.")
        return

    group_order = ['Control', 'Hypertension', 'Diabetes']
    light_colors, dark_colors = _get_group_colors()

    # Use 5-year bins (same as histogram)
    age_values = plot_df['Age']
    age_min = int(age_values.min())
    age_max = int(age_values.max())
    bin_start = (age_min // 5) * 5
    bin_end = ((age_max // 5) + 1) * 5
    bins = np.arange(bin_start, bin_end + 5, 5)

    fig, ax = plt.subplots(figsize=(4, 2.5))
    for i, grp in enumerate(group_order):
        subset = plot_df[plot_df['Group'] == grp]['Age']
        if len(subset) < 2:
            continue
        fill_color = (*light_colors[i], VIOLIN_FILL_ALPHA)
        sns.histplot(
            data=subset,
            ax=ax,
            bins=bins,
            stat='count',
            kde=True,
            color=fill_color,
            edgecolor=dark_colors[i],
            linewidth=0.5,
            line_kws=dict(color=dark_colors[i], linewidth=1.2),
            label=f'{grp}',
        )

    if source_sans:
        ax.set_title('Age distribution by group', fontproperties=source_sans_title)
        ax.set_xlabel('Age (years)', fontproperties=source_sans)
        ax.set_ylabel('Frequency', fontproperties=source_sans)
        ax.legend(prop=source_sans)
    else:
        ax.set_title('Age distribution by group', fontsize=TITLE_FONTSIZE)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Frequency')
        ax.legend()

    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    plt.tight_layout()

    filename = 'participant_kde_by_group.png'
    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"KDE plot saved to: {outpath}")
    for grp in group_order:
        n = len(plot_df[plot_df['Group'] == grp])
        if n > 0:
            ages = plot_df[plot_df['Group'] == grp]['Age']
            print(f"  {grp}: n={n}, age {ages.min():.0f}-{ages.max():.0f}, mean={ages.mean():.1f}")


def create_participant_venn(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> None:
    """
    Creates a Venn diagram showing cohort sizes for Control, Hypertension,
    and Diabetes. Uses the boolean Hypertension / Diabetes columns so that
    participants with both conditions appear in the overlap.

    Args:
        df: DataFrame with Participant, Hypertension, and Diabetes columns.
        output_dir: Directory to save the plot (optional).
    """
    from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles

    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'ParticipantHistograms')
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating participant Venn diagram...")

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 4,
        'lines.linewidth': 0.5,
    })

    # One row per participant
    plot_df = df.drop_duplicates(subset=['Participant']).copy()

    # Build boolean masks from the actual Hypertension / Diabetes columns
    # (these can overlap, unlike SET which is mutually exclusive)
    hyp_mask = plot_df['Hypertension'].isin([True, 'True', 'TRUE', 1, 1.0])
    dia_mask = plot_df['Diabetes'].isin([True, 'True', 'TRUE', 1, 1.0])
    control_mask = ~hyp_mask & ~dia_mask

    set_control = set(plot_df.loc[control_mask, 'Participant'])
    set_hyp = set(plot_df.loc[hyp_mask, 'Participant'])
    set_dia = set(plot_df.loc[dia_mask, 'Participant'])

    # Counts per region
    only_ctrl = len(set_control - set_hyp - set_dia)
    only_hyp = len(set_hyp - set_control - set_dia)
    only_dia = len(set_dia - set_control - set_hyp)
    ctrl_hyp = len((set_control & set_hyp) - set_dia)
    ctrl_dia = len((set_control & set_dia) - set_hyp)
    hyp_dia = len((set_hyp & set_dia) - set_control)
    all_three = len(set_control & set_hyp & set_dia)

    print(f"  Control only: {only_ctrl}")
    print(f"  Hypertension only: {only_hyp}")
    print(f"  Diabetes only: {only_dia}")
    print(f"  Hypertension & Diabetes: {hyp_dia}")
    print(f"  Control & Hypertension: {ctrl_hyp}")
    print(f"  Control & Diabetes: {ctrl_dia}")
    print(f"  All three: {all_three}")

    # Build light / dark palettes (same as violin)
    # Order: Control, Hypertension, Diabetes
    light_colors, dark_colors = _get_group_colors()

    # venn3 layout: A=upper-left, B=upper-right, C=bottom-center
    # We want: Control=left (A), Diabetes=right (B), Hypertension=bottom (C)
    # Subset tuple: (Abc, aBc, ABc, abC, AbC, aBC, ABC)
    #   A=Control, B=Diabetes, C=Hypertension
    subsets = (
        only_ctrl,   # Abc  = Control only
        only_dia,    # aBc  = Diabetes only
        ctrl_dia,    # ABc  = Control & Diabetes
        only_hyp,    # abC  = Hypertension only
        ctrl_hyp,    # AbC  = Control & Hypertension
        hyp_dia,     # aBC  = Diabetes & Hypertension
        all_three,   # ABC  = all three
    )

    fig, ax = plt.subplots(figsize=(4, 3.5))
    v = venn3(
        subsets=subsets,
        set_labels=('Control', 'Diabetes', 'Hypertension'),
        ax=ax,
    )

    # Color each region
    # A=Control (idx 0), B=Diabetes (idx 2), C=Hypertension (idx 1)
    region_color_map = {
        '100': light_colors[0],        # Control only
        '010': light_colors[2],        # Diabetes only
        '110': light_colors[0],        # Control & Diabetes
        '001': light_colors[1],        # Hypertension only
        '101': light_colors[0],        # Control & Hypertension
        '011': (0.85, 0.55, 0.2),      # Diabetes & Hypertension blend
        '111': (0.7, 0.7, 0.7),        # All three
    }
    for region_id, color in region_color_map.items():
        patch = v.get_patch_by_id(region_id)
        if patch is not None:
            patch.set_color((*color, VIOLIN_FILL_ALPHA))
            patch.set_edgecolor('none')

    # Dark outlines on the circles: A=Control, B=Diabetes, C=Hypertension
    circle_dark = [dark_colors[0], dark_colors[2], dark_colors[1]]
    circles = venn3_circles(subsets=subsets, ax=ax)
    for i, circ in enumerate(circles):
        circ.set_edgecolor(circle_dark[i])
        circ.set_linewidth(1.2)
        circ.set_alpha(1.0)

    # Style set labels (group names around the circles)
    # Align all labels on the same horizontal line below the diagram
    label_y_positions = []
    for text in v.set_labels:
        if text is not None:
            label_y_positions.append(text.get_position()[1])
    # Use the lowest label position (most negative y) for all labels
    common_y = min(label_y_positions) if label_y_positions else -0.35
    # set_labels order: 0=Control (A), 1=Diabetes (B), 2=Hypertension (C)
    # x-axis is inverted, so subtracting from x moves text visually right
    label_x_nudge = {1: -0.08}  # nudge Diabetes slightly right
    for i, text in enumerate(v.set_labels):
        if text is not None:
            x, _ = text.get_position()
            x += label_x_nudge.get(i, 0)
            text.set_position((x, common_y))
            text.set_ha('center')
            if source_sans:
                text.set_fontproperties(source_sans)
            text.set_fontsize(10)

    # Style subset labels (counts inside the circles)
    for text in v.subset_labels:
        if text is not None:
            if source_sans:
                text.set_fontproperties(source_sans)
            text.set_fontsize(14)

    if source_sans:
        ax.set_title('Participant cohort overlap', fontproperties=source_sans_title)
    else:
        ax.set_title('Participant cohort overlap', fontsize=TITLE_FONTSIZE)

    # Flip left-right so Control is on the left and disease pair on the right
    ax.invert_xaxis()
    ax.set_axis_off()
    plt.tight_layout()

    filename = 'participant_venn_diagram.png'
    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Venn diagram saved to: {outpath}")


def create_split_histogram_by_group(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> None:
    """
    Creates a back-to-back population-pyramid histogram on a single axes:
    Control bars extend upward (positive y), Hypertension and Diabetes bars
    extend downward (negative y, overlapping with transparency).
    y-tick labels are shown as absolute values.

    Args:
        df: DataFrame with Age and SET columns.
        output_dir: Directory to save the plot (optional).
    """
    from matplotlib.ticker import FuncFormatter

    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'ParticipantHistograms')
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating back-to-back pyramid histogram (Control vs Disease)...")

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 4,
        'lines.linewidth': 0.5,
    })

    plot_df = _get_plot_df(df)
    if plot_df is None:
        print("Error: No valid age data with SET in set01/set02/set03 found.")
        return

    # Shared 5-year bins
    age_values = plot_df['Age']
    age_min = int(age_values.min())
    age_max = int(age_values.max())
    bin_start = (age_min // 5) * 5
    bin_end = ((age_max // 5) + 1) * 5
    bins = np.arange(bin_start, bin_end + 5, 5)
    bin_width = bins[1] - bins[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Light / dark palettes (same as violin)
    light_colors, dark_colors = _get_group_colors()

    control_ages = plot_df[plot_df['Group'] == 'Control']['Age']
    hyp_ages = plot_df[plot_df['Group'] == 'Hypertension']['Age']
    dia_ages = plot_df[plot_df['Group'] == 'Diabetes']['Age']

    # Histogram counts
    ctrl_counts, _ = np.histogram(control_ages, bins=bins)
    hyp_counts, _ = np.histogram(hyp_ages, bins=bins)
    dia_counts, _ = np.histogram(dia_ages, bins=bins)

    # Symmetric y-limit
    y_abs_max = max(ctrl_counts.max(), hyp_counts.max(), dia_counts.max()) + 1

    fig, ax = plt.subplots(figsize=(4, 4))

    # --- Control: bars going UP (positive) ---
    fill_ctrl = (*light_colors[0], VIOLIN_FILL_ALPHA)
    ax.bar(
        bin_centers, ctrl_counts, width=bin_width * 0.9,
        color=fill_ctrl, edgecolor=dark_colors[0], linewidth=0.8,
        label='Control',
    )

    # --- Disease bars going DOWN (negative), semi-transparent with dark edges ---
    # Draw diabetes first (behind), then hypertension (in front);
    # both are semi-transparent so the back group still shows through.
    fill_dia = (*light_colors[2], VIOLIN_FILL_ALPHA)
    ax.bar(
        bin_centers, -dia_counts, width=bin_width * 0.9,
        color=fill_dia, edgecolor=dark_colors[2], linewidth=0.8,
        alpha=0.7, label=f'Diabetes',
    )

    fill_hyp = (*light_colors[1], VIOLIN_FILL_ALPHA)
    ax.bar(
        bin_centers, -hyp_counts, width=bin_width * 0.9,
        color=fill_hyp, edgecolor=dark_colors[1], linewidth=0.8,
        alpha=0.7, label=f'Hypertension',
    )

    # --- Dividing line at y=0 ---
    ax.axhline(0, color='black', linewidth=0.6)

    # --- Symmetric y-limits ---
    ax.set_ylim(-y_abs_max, y_abs_max)

    # --- Replace negative tick labels with absolute values ---
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{abs(int(v))}'))

    # --- Labels and legend ---
    ax.grid(True, alpha=0.3)
    if source_sans:
        ax.set_xlabel('Age (years)', fontproperties=source_sans)
        ax.set_ylabel('Frequency', fontproperties=source_sans)
        ax.legend(prop=source_sans, loc='upper right')
        ax.set_title('Age distribution: Control vs Disease',
                     fontproperties=source_sans_title)
    else:
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right')
        ax.set_title('Age distribution: Control vs Disease', fontsize=TITLE_FONTSIZE)

    plt.tight_layout()

    filename = 'participant_split_histogram.png'
    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Split histogram saved to: {outpath}")
    for grp in ['Control', 'Hypertension', 'Diabetes']:
        subset = plot_df[plot_df['Group'] == grp]
        if len(subset) > 0:
            ages = subset['Age']
            print(f"  {grp}: n={len(subset)}, age {ages.min():.0f}-{ages.max():.0f}, mean={ages.mean():.1f}")


def create_participant_density_by_group(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> None:
    """
    Creates overlapping density (KDE) curves for participant age by group.
    Light transparent fill + dark opaque outline for each group.
    No histogram bars -- just smooth curves.

    Args:
        df: DataFrame with Age and SET columns.
        output_dir: Directory to save the plot (optional).
    """
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'ParticipantHistograms')
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating participant density plot by group...")

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 4,
        'lines.linewidth': 0.5,
    })

    plot_df = _get_plot_df(df)
    if plot_df is None:
        print("Error: No valid age data with SET in set01/set02/set03 found.")
        return

    group_order = ['Control', 'Hypertension', 'Diabetes']
    light_colors, dark_colors = _get_group_colors()

    fig, ax = plt.subplots(figsize=(4, 2.5))
    for i, grp in enumerate(group_order):
        subset = plot_df[plot_df['Group'] == grp]['Age']
        if len(subset) < 2:
            continue
        fill_color = (*light_colors[i], VIOLIN_FILL_ALPHA)
        # Pass 1: transparent fill, no visible border
        sns.kdeplot(
            data=subset,
            ax=ax,
            fill=True,
            color=fill_color,
            linewidth=0,
        )
        # Pass 2: dark opaque outline
        sns.kdeplot(
            data=subset,
            ax=ax,
            fill=False,
            color=dark_colors[i],
            linewidth=1.2,
            label=f'{grp}',
        )

    if source_sans:
        ax.set_title('Age density by group', fontproperties=source_sans_title)
        ax.set_xlabel('Age (years)', fontproperties=source_sans)
        ax.set_ylabel('Density', fontproperties=source_sans)
        ax.legend(prop=source_sans)
    else:
        ax.set_title('Age density by group', fontsize=TITLE_FONTSIZE)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Density')
        ax.legend()

    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    plt.tight_layout()

    filename = 'participant_density_by_group.png'
    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Density plot saved to: {outpath}")
    for grp in group_order:
        n = len(plot_df[plot_df['Group'] == grp])
        if n > 0:
            ages = plot_df[plot_df['Group'] == grp]['Age']
            print(f"  {grp}: n={n}, age {ages.min():.0f}-{ages.max():.0f}, mean={ages.mean():.1f}")


def create_participant_count_density_by_group(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> None:
    """
    Creates overlapping KDE curves scaled to participant counts (y-axis =
    number of participants, not probability density). The area under each
    curve equals that group's n, making the curves directly comparable to
    histogram bar heights.

    Args:
        df: DataFrame with Age and SET columns.
        output_dir: Directory to save the plot (optional).
    """
    from scipy.stats import gaussian_kde

    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'ParticipantHistograms')
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating participant count-density plot by group...")

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 4,
        'lines.linewidth': 0.5,
    })

    plot_df = _get_plot_df(df)
    if plot_df is None:
        print("Error: No valid age data with SET in set01/set02/set03 found.")
        return

    group_order = ['Control', 'Hypertension', 'Diabetes']
    light_colors, dark_colors = _get_group_colors()

    # Evaluation grid spanning all ages
    age_values = plot_df['Age']
    x_grid = np.linspace(age_values.min() - 5, age_values.max() + 5, 300)

    fig, ax = plt.subplots(figsize=(4, 2.5))
    for i, grp in enumerate(group_order):
        subset = plot_df[plot_df['Group'] == grp]['Age'].values
        if len(subset) < 2:
            continue
        n = len(subset)
        kde = gaussian_kde(subset)
        # density * n  scales so area under curve = n
        y = kde(x_grid) * n

        fill_color = (*light_colors[i], VIOLIN_FILL_ALPHA)
        ax.fill_between(x_grid, y, color=fill_color)
        ax.plot(x_grid, y, color=dark_colors[i], linewidth=1.2,
                label=f'{grp}')

    if source_sans:
        ax.set_title('Age count-density by group', fontproperties=source_sans_title)
        ax.set_xlabel('Age (years)', fontproperties=source_sans)
        ax.set_ylabel('Count density', fontproperties=source_sans)
        ax.legend(prop=source_sans)
    else:
        ax.set_title('Age count-density by group', fontsize=TITLE_FONTSIZE)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Count density')
        ax.legend()

    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    plt.tight_layout()

    filename = 'participant_count_density_by_group.png'
    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Count-density plot saved to: {outpath}")
    for grp in group_order:
        n = len(plot_df[plot_df['Group'] == grp])
        if n > 0:
            ages = plot_df[plot_df['Group'] == grp]['Age']
            print(f"  {grp}: n={n}, age {ages.min():.0f}-{ages.max():.0f}, mean={ages.mean():.1f}")


def create_disease_histogram(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> None:
    """
    Creates an overlapping histogram of just Hypertension and Diabetes
    participants (no Control). Same light fill + dark edge style.

    Args:
        df: DataFrame with Age and SET columns.
        output_dir: Directory to save the plot (optional).
    """
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'ParticipantHistograms')
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating disease-only histogram (Hypertension + Diabetes)...")

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 4,
        'lines.linewidth': 0.5,
    })

    plot_df = _get_plot_df(df)
    if plot_df is None:
        print("Error: No valid age data with SET in set01/set02/set03 found.")
        return

    # Only disease groups
    plot_df = plot_df[plot_df['Group'].isin(['Hypertension', 'Diabetes'])]
    if len(plot_df) == 0:
        print("Error: No Hypertension or Diabetes participants found.")
        return

    # 5-year bins
    age_values = plot_df['Age']
    age_min = int(age_values.min())
    age_max = int(age_values.max())
    bin_start = (age_min // 5) * 5
    bin_end = ((age_max // 5) + 1) * 5
    bins = np.arange(bin_start, bin_end + 5, 5)

    # Light / dark palettes (indices 1=hyp, 2=dia)
    light_colors, dark_colors = _get_group_colors()

    hyp_ages = plot_df[plot_df['Group'] == 'Hypertension']['Age']
    dia_ages = plot_df[plot_df['Group'] == 'Diabetes']['Age']

    fig, ax = plt.subplots(figsize=(4, 2.5))

    # Draw diabetes first (behind), then hypertension (in front)
    fill_dia = (*light_colors[2], VIOLIN_FILL_ALPHA)
    ax.hist(
        dia_ages, bins=bins, alpha=0.7,
        color=fill_dia, edgecolor=dark_colors[2], linewidth=0.8,
        label=f'Diabetes',
    )
    fill_hyp = (*light_colors[1], VIOLIN_FILL_ALPHA)
    ax.hist(
        hyp_ages, bins=bins, alpha=0.7,
        color=fill_hyp, edgecolor=dark_colors[1], linewidth=0.8,
        label=f'Hypertension',
    )

    if source_sans:
        ax.set_title('Age distribution: Hypertension & Diabetes',
                     fontproperties=source_sans_title)
        ax.set_xlabel('Age (years)', fontproperties=source_sans)
        ax.set_ylabel('Frequency', fontproperties=source_sans)
        ax.legend(prop=source_sans)
    else:
        ax.set_title('Age distribution: Hypertension & Diabetes', fontsize=TITLE_FONTSIZE)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Frequency')
        ax.legend()

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = 'participant_disease_histogram.png'
    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Disease histogram saved to: {outpath}")
    for grp in ['Hypertension', 'Diabetes']:
        subset = plot_df[plot_df['Group'] == grp]
        if len(subset) > 0:
            ages = subset['Age']
            print(f"  {grp}: n={len(subset)}, age {ages.min():.0f}-{ages.max():.0f}, mean={ages.mean():.1f}")


def create_disease_histogram_alt(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> None:
    """
    Same as create_disease_histogram but with Diabetes (orange) drawn on top
    of Hypertension (red).

    Args:
        df: DataFrame with Age and SET columns.
        output_dir: Directory to save the plot (optional).
    """
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'ParticipantHistograms')
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating disease histogram (Diabetes on top)...")

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 4,
        'lines.linewidth': 0.5,
    })

    plot_df = _get_plot_df(df)
    if plot_df is None:
        print("Error: No valid age data with SET in set01/set02/set03 found.")
        return

    plot_df = plot_df[plot_df['Group'].isin(['Hypertension', 'Diabetes'])]
    if len(plot_df) == 0:
        print("Error: No Hypertension or Diabetes participants found.")
        return

    age_values = plot_df['Age']
    age_min = int(age_values.min())
    age_max = int(age_values.max())
    bin_start = (age_min // 5) * 5
    bin_end = ((age_max // 5) + 1) * 5
    bins = np.arange(bin_start, bin_end + 5, 5)

    light_colors, dark_colors = _get_group_colors()

    hyp_ages = plot_df[plot_df['Group'] == 'Hypertension']['Age']
    dia_ages = plot_df[plot_df['Group'] == 'Diabetes']['Age']

    fig, ax = plt.subplots(figsize=(4, 2.5))

    # Draw hypertension first (behind), then diabetes on top
    fill_hyp = (*light_colors[1], VIOLIN_FILL_ALPHA)
    ax.hist(
        hyp_ages, bins=bins, alpha=0.7,
        color=fill_hyp, edgecolor=dark_colors[1], linewidth=0.8,
        label=f'Hypertension',
    )
    fill_dia = (*light_colors[2], VIOLIN_FILL_ALPHA)
    ax.hist(
        dia_ages, bins=bins, alpha=0.7,
        color=fill_dia, edgecolor=dark_colors[2], linewidth=0.8,
        label=f'Diabetes',
    )

    if source_sans:
        ax.set_title('Age distribution: Hypertension & Diabetes',
                     fontproperties=source_sans_title)
        ax.set_xlabel('Age (years)', fontproperties=source_sans)
        ax.set_ylabel('Frequency', fontproperties=source_sans)
        ax.legend(prop=source_sans)
    else:
        ax.set_title('Age distribution: Hypertension & Diabetes', fontsize=TITLE_FONTSIZE)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Frequency')
        ax.legend()

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = 'participant_disease_histogram_alt.png'
    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Disease histogram (alt) saved to: {outpath}")
    for grp in ['Hypertension', 'Diabetes']:
        subset = plot_df[plot_df['Group'] == grp]
        if len(subset) > 0:
            ages = subset['Age']
            print(f"  {grp}: n={len(subset)}, age {ages.min():.0f}-{ages.max():.0f}, mean={ages.mean():.1f}")


def create_control_histogram(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> None:
    """
    Creates a histogram of just Control participants.
    Same light fill + dark edge style as the other histograms.

    Args:
        df: DataFrame with Age and SET columns.
        output_dir: Directory to save the plot (optional).
    """
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'ParticipantHistograms')
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating control-only histogram...")

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 4,
        'lines.linewidth': 0.5,
    })

    plot_df = _get_plot_df(df)
    if plot_df is None:
        print("Error: No valid age data with SET in set01/set02/set03 found.")
        return

    control_df = plot_df[plot_df['Group'] == 'Control']
    if len(control_df) == 0:
        print("Error: No Control participants found.")
        return

    # 5-year bins
    age_values = control_df['Age']
    age_min = int(age_values.min())
    age_max = int(age_values.max())
    bin_start = (age_min // 5) * 5
    bin_end = ((age_max // 5) + 1) * 5
    bins = np.arange(bin_start, bin_end + 5, 5)

    # Light / dark palette for control (index 0)
    light_colors, dark_colors = _get_group_colors()
    light = light_colors[0]
    dark = dark_colors[0]

    fig, ax = plt.subplots(figsize=(4, 2.5))
    fill_color = (*light, VIOLIN_FILL_ALPHA)
    ax.hist(
        age_values, bins=bins,
        color=fill_color, edgecolor=dark, linewidth=0.8,
        label=f'Control',
    )

    if source_sans:
        ax.set_title('Age distribution: Control',
                     fontproperties=source_sans_title)
        ax.set_xlabel('Age (years)', fontproperties=source_sans)
        ax.set_ylabel('Frequency', fontproperties=source_sans)
        ax.legend(prop=source_sans)
    else:
        ax.set_title('Age distribution: Control', fontsize=TITLE_FONTSIZE)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Frequency')
        ax.legend()

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = 'participant_control_histogram.png'
    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Control histogram saved to: {outpath}")
    print(f"  Control: n={len(control_df)}, age {age_values.min():.0f}-{age_values.max():.0f}, mean={age_values.mean():.1f}")


def create_single_group_histogram(
    df: pd.DataFrame,
    group: str,
    output_dir: Optional[str] = None,
    shared_bins: Optional[np.ndarray] = None,
    shared_ylim: Optional[float] = None,
) -> None:
    """
    Creates a histogram for a single group (e.g. 'Hypertension' or 'Diabetes').

    Args:
        df: DataFrame with Age and SET columns.
        group: Group name ('Control', 'Hypertension', or 'Diabetes').
        output_dir: Directory to save the plot (optional).
        shared_bins: Bin edges to use (optional; computed from this group if None).
        shared_ylim: Maximum y value (optional; auto-scaled if None).
    """
    color_map = {
        'Control': COLOR_CONTROL,
        'Hypertension': COLOR_HYPERTENSION,
        'Diabetes': COLOR_DIABETES,
    }
    if group not in color_map:
        print(f"Error: Unknown group '{group}'.")
        return

    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'ParticipantHistograms')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCreating {group}-only histogram...")

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 4,
        'lines.linewidth': 0.5,
    })

    plot_df = _get_plot_df(df)
    if plot_df is None:
        print("Error: No valid age data found.")
        return

    group_df = plot_df[plot_df['Group'] == group]
    if len(group_df) == 0:
        print(f"Error: No {group} participants found.")
        return

    # Use shared bins if provided, otherwise compute from this group
    if shared_bins is not None:
        bins = shared_bins
    else:
        age_values = group_df['Age']
        age_min = int(age_values.min())
        age_max = int(age_values.max())
        bin_start = (age_min // 5) * 5
        bin_end = ((age_max // 5) + 1) * 5
        bins = np.arange(bin_start, bin_end + 5, 5)

    age_values = group_df['Age']

    group_idx = {'Control': 0, 'Hypertension': 1, 'Diabetes': 2}[group]
    light_colors, dark_colors = _get_group_colors()
    light = light_colors[group_idx]
    dark = dark_colors[group_idx]

    fig, ax = plt.subplots(figsize=(4, 2.5))
    fill_color = (*light, VIOLIN_FILL_ALPHA)
    ax.hist(
        age_values, bins=bins,
        color=fill_color, edgecolor=dark, linewidth=0.8,
        label=f'{group}',
    )

    if shared_ylim is not None:
        ax.set_ylim(0, shared_ylim)

    if source_sans:
        ax.set_title(f'Age distribution: {group}',
                     fontproperties=source_sans_title)
        ax.set_xlabel('Age (years)', fontproperties=source_sans)
        ax.set_ylabel('Frequency', fontproperties=source_sans)
        ax.legend(prop=source_sans)
    else:
        ax.set_title(f'Age distribution: {group}', fontsize=TITLE_FONTSIZE)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Frequency')
        ax.legend()

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f'participant_{group.lower()}_histogram.png'
    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"{group} histogram saved to: {outpath}")
    print(f"  {group}: n={len(group_df)}, age {age_values.min():.0f}-{age_values.max():.0f}, mean={age_values.mean():.1f}")


def main() -> int:
    """Load summary data and create all participant plots by group."""
    print("\nCreating participant plots by group...")
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)

    if 'Age' not in df.columns:
        print("Error: Age column not found in the data.")
        return 1
    if 'SET' not in df.columns:
        print("Error: SET column not found in the data.")
        return 1

    output_dir = os.path.join(cap_flow_path, 'results', 'ParticipantHistograms')
    create_participant_histogram_by_group(df, output_dir)
    create_participant_violin_by_group(df, output_dir)
    create_participant_kde_by_group(df, output_dir)
    create_participant_venn(df, output_dir)
    create_split_histogram_by_group(df, output_dir)
    create_participant_density_by_group(df, output_dir)
    create_participant_count_density_by_group(df, output_dir)
    create_disease_histogram(df, output_dir)
    create_disease_histogram_alt(df, output_dir)
    create_control_histogram(df, output_dir)

    # Compute shared bins and y-limit across all three groups for individual histograms
    plot_df = _get_plot_df(df)
    if plot_df is not None:
        all_ages = plot_df['Age']
        age_min = int(all_ages.min())
        age_max = int(all_ages.max())
        bin_start = (age_min // 5) * 5
        bin_end = ((age_max // 5) + 1) * 5
        shared_bins = np.arange(bin_start, bin_end + 5, 5)
        # Find max count across all groups
        max_count = 0
        for grp in ['Control', 'Hypertension', 'Diabetes']:
            grp_ages = plot_df[plot_df['Group'] == grp]['Age']
            if len(grp_ages) > 0:
                counts, _ = np.histogram(grp_ages, bins=shared_bins)
                max_count = max(max_count, counts.max())
        shared_ylim = max_count + 1

        for grp in ['Control', 'Hypertension', 'Diabetes']:
            create_single_group_histogram(df, grp, output_dir,
                                          shared_bins=shared_bins,
                                          shared_ylim=shared_ylim)

    print("\nParticipant plots by group complete.")
    print(f"Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create participant age histogram colored by Control / Hypertension / Diabetes.',
    )
    args = parser.parse_args()
    raise SystemExit(main())
