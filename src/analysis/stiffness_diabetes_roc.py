"""
Filename: src/analysis/stiffness_diabetes_roc.py
--------------------------------------------------

ROC analysis testing whether the Stiffness Index (SI) can discriminate
diabetic from non-diabetic participants.

Biological hypothesis: diabetic capillaries are stiffer (higher SI)
because they resist collapse under external pressure.  Higher SI → stiffer
→ less healthy.

Analyses:
  1. Single-variable ROC: SI alone predicting diabetes
  2. Multi-variable ROC: Logistic regression with LOO cross-validation
  3. Overlay ROC plots comparing models
  4. SI boxplot by diabetes status
  5. Summary CSV table of all AUCs and CIs

By: Marcus Forst
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc
from typing import List, Tuple, Dict
import warnings

from src.config import PATHS, load_source_sans
from src.analysis.plot_stiffness import (
    apply_font, save_figure, _ensure_healthy_flag,
    CONTROL_COLOR, DIABETES_COLOR,
)
from src.analysis.stiffness_age_roc import (
    load_stiffness_data,
    calculate_auc_ci_delong,
    bootstrap_roc,
    loo_cv_predict_proba,
    _font_kw,
)

cap_flow_path = PATHS['cap_flow']
source_sans = load_source_sans()

# Publication-quality plotting style
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
    'grid.linewidth': 0.3,
})
sns.set_style("whitegrid")

# Feature used throughout
SI_COL = 'SI_logvel_averaged_02_12'

# Overlay model colours
MODEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


# ---------------------------------------------------------------------------
# Single-variable ROC pipeline
# ---------------------------------------------------------------------------

def run_single_variable_roc(df: pd.DataFrame, feature: str,
                            subset_label: str) -> Dict:
    """Compute ROC for a single feature predicting diabetes.

    Args:
        df: DataFrame with ``Diabetes`` boolean and *feature* column.
        feature: Column name for the predictor (e.g. SI_COL).
        subset_label: 'Diab vs Controls' or 'Diab vs All Non-diabetic'.

    Returns:
        Dict with keys: fpr, tpr, auc, ci_lower, ci_upper,
        boot_ci_lower, boot_ci_upper, tprs_boot, n_pos, n_neg,
        subset, model_label.
    """
    y = df['Diabetes'].astype(int).values
    scores = df[feature].values

    if len(np.unique(y)) < 2:
        warnings.warn(f"Only one class in {subset_label} — skipping.")
        return {}

    fpr, tpr, _ = roc_curve(y, scores)
    auc_val = auc(fpr, tpr)
    _, _, ci_lo, ci_hi = calculate_auc_ci_delong(y, scores)
    tprs_boot, _, boot_lo, boot_hi = bootstrap_roc(y, scores)

    return dict(
        fpr=fpr, tpr=tpr, auc=auc_val,
        ci_lower=ci_lo, ci_upper=ci_hi,
        boot_ci_lower=boot_lo, boot_ci_upper=boot_hi,
        tprs_boot=tprs_boot,
        n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
        subset=subset_label,
        model_label='SI alone',
    )


# ---------------------------------------------------------------------------
# Multi-variable ROC pipeline (LOO CV)
# ---------------------------------------------------------------------------

def run_multivariable_roc(df: pd.DataFrame, feature_cols: List[str],
                          subset_label: str,
                          model_label: str) -> Dict:
    """Logistic regression ROC with LOO CV for diabetes prediction.

    Args:
        df: DataFrame (already subset to controls or all non-diabetic + diabetic).
        feature_cols: Column names to include as predictors.
        subset_label: 'Diab vs Controls' or 'Diab vs All Non-diabetic'.
        model_label: Human-readable model name for legends.

    Returns:
        Dict matching run_single_variable_roc output schema.
    """
    sub = df.dropna(subset=feature_cols).copy()
    y = sub['Diabetes'].astype(int).values
    X = sub[feature_cols].values

    if len(np.unique(y)) < 2 or len(sub) < 10:
        warnings.warn(f"Insufficient data for {model_label} "
                      f"in {subset_label} — skipping.")
        return {}

    proba = loo_cv_predict_proba(X, y)
    valid = ~np.isnan(proba)
    y_v, p_v = y[valid], proba[valid]

    if len(np.unique(y_v)) < 2:
        return {}

    fpr, tpr, _ = roc_curve(y_v, p_v)
    auc_val = auc(fpr, tpr)
    _, _, ci_lo, ci_hi = calculate_auc_ci_delong(y_v, p_v)
    tprs_boot, _, boot_lo, boot_hi = bootstrap_roc(y_v, p_v)

    return dict(
        fpr=fpr, tpr=tpr, auc=auc_val,
        ci_lower=ci_lo, ci_upper=ci_hi,
        boot_ci_lower=boot_lo, boot_ci_upper=boot_hi,
        tprs_boot=tprs_boot,
        n_pos=int(y_v.sum()), n_neg=int((1 - y_v).sum()),
        subset=subset_label,
        model_label=model_label,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single_roc(result: Dict, output_dir: str, minimal_dir: str):
    """Plot an individual ROC curve with bootstrap CI shading."""
    if not result:
        return

    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    base_color = '#1f77b4'

    ax.plot(result['fpr'], result['tpr'], color=base_color,
            label=f"AUC = {result['auc']:.2f} "
                  f"[{result['ci_lower']:.2f}\u2013{result['ci_upper']:.2f}]")

    if len(result['tprs_boot']) > 1:
        lo = np.percentile(result['tprs_boot'], 2.5, axis=0)
        hi = np.percentile(result['tprs_boot'], 97.5, axis=0)
        ax.fill_between(np.linspace(0, 1, 100), lo, hi,
                        color=base_color, alpha=0.2, label='95% CI')

    ax.plot([0, 1], [0, 1], 'grey', linestyle='--', label='Random')

    ax.set_xlabel('False Positive Rate', **_font_kw())
    ax.set_ylabel('True Positive Rate', **_font_kw())
    ax.set_title(f"{result['subset']}  "
                 f"(n+={result['n_pos']}, n\u2013={result['n_neg']})",
                 fontsize=7, **_font_kw())
    ax.legend(loc='lower right', fontsize=4,
              prop=source_sans if source_sans else None)
    ax.grid(True, linewidth=0.3)
    apply_font(ax, source_sans)
    plt.tight_layout()

    fname = (f"roc_SI_diabetes_"
             f"{result['subset'].lower().replace(' ', '_')}")
    save_figure(fig, output_dir, fname, minimal_dir=minimal_dir)
    plt.close()


def plot_roc_overlay(results_list: List[Dict], output_dir: str,
                     minimal_dir: str):
    """Overlay multiple ROC curves on one axes."""
    valid = [r for r in results_list if r]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(2.4, 2.0))

    for i, res in enumerate(valid):
        c = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(res['fpr'], res['tpr'], color=c,
                label=f"{res['model_label']}  AUC={res['auc']:.2f}")

        if len(res['tprs_boot']) > 1:
            lo = np.percentile(res['tprs_boot'], 2.5, axis=0)
            hi = np.percentile(res['tprs_boot'], 97.5, axis=0)
            ax.fill_between(np.linspace(0, 1, 100), lo, hi,
                            color=c, alpha=0.1)

    ax.plot([0, 1], [0, 1], 'grey', linestyle='--', label='Random')

    ref = valid[0]
    ax.set_xlabel('False Positive Rate', **_font_kw())
    ax.set_ylabel('True Positive Rate', **_font_kw())
    ax.set_title(f"{ref['subset']}", fontsize=7, **_font_kw())
    ax.legend(loc='lower right', fontsize=4,
              prop=source_sans if source_sans else None)
    ax.grid(True, linewidth=0.3)
    apply_font(ax, source_sans)
    plt.tight_layout()

    fname = (f"roc_diabetes_overlay_"
             f"{ref['subset'].lower().replace(' ', '_')}")
    save_figure(fig, output_dir, fname, minimal_dir=minimal_dir)
    plt.close()


def plot_si_boxplot(df: pd.DataFrame, output_dir: str, minimal_dir: str,
                    subset_label: str):
    """Boxplot of SI by diabetes status with jittered points and p-value.

    Args:
        df: DataFrame with ``Diabetes`` column and SI_COL.
        output_dir: Directory for output files.
        minimal_dir: Directory for no-annotation copies.
        subset_label: 'Diab vs Controls' or 'Diab vs All Non-diabetic'.
    """
    sub = df.dropna(subset=[SI_COL]).copy()
    if len(sub) < 5:
        return

    diab = sub.loc[sub['Diabetes'] == True, SI_COL].values
    non_diab = sub.loc[sub['Diabetes'] == False, SI_COL].values

    # Mann-Whitney U test
    stat, p_val = stats.mannwhitneyu(diab, non_diab, alternative='two-sided')

    # Build tidy dataframe for seaborn
    sub['Group'] = sub['Diabetes'].map({True: 'Diabetic', False: 'Non-diabetic'})

    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    palette = [CONTROL_COLOR, DIABETES_COLOR]
    order = ['Non-diabetic', 'Diabetic']

    sns.boxplot(data=sub, x='Group', y=SI_COL, ax=ax,
                order=order, palette=palette, width=0.6)

    # Jittered individual points
    rng = np.random.RandomState(42)
    for i, group in enumerate(order):
        group_data = sub.loc[sub['Group'] == group, SI_COL]
        x_pos = i + rng.normal(0, 0.05, len(group_data))
        ax.scatter(x_pos, group_data, alpha=0.4, s=10, color='gray', zorder=3)

    # p-value annotation
    y_max = sub[SI_COL].max()
    y_range = sub[SI_COL].max() - sub[SI_COL].min()
    bar_y = y_max + 0.05 * y_range
    ax.plot([0, 0, 1, 1], [bar_y, bar_y + 0.02 * y_range,
                            bar_y + 0.02 * y_range, bar_y],
            color='black', linewidth=0.5)
    p_str = f'p = {p_val:.3f}' if p_val >= 0.001 else f'p = {p_val:.1e}'
    ax.text(0.5, bar_y + 0.03 * y_range, p_str, ha='center', va='bottom',
            fontsize=5, **_font_kw())

    ax.set_xlabel('', **_font_kw())
    ax.set_ylabel('SI (log-velocity AUC, 0.2\u20131.2 psi)', **_font_kw())
    ax.set_title(f'SI by Diabetes Status \u2014 {subset_label}',
                 fontsize=7, **_font_kw())
    ax.grid(True, linewidth=0.3)
    apply_font(ax, source_sans)
    plt.tight_layout()

    fname = (f"boxplot_SI_diabetes_"
             f"{subset_label.lower().replace(' ', '_')}")
    save_figure(fig, output_dir, fname, minimal_dir=minimal_dir)
    plt.close()

    print(f"  Mann-Whitney U: stat={stat:.1f}, p={p_val:.4f}, "
          f"n_diab={len(diab)}, n_non={len(non_diab)}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_table(all_results: List[Dict], output_dir: str):
    """Write a CSV summarising all ROC results."""
    rows = []
    for r in all_results:
        if not r:
            continue
        rows.append({
            'Subset': r['subset'],
            'Model': r['model_label'],
            'AUC': round(r['auc'], 3),
            'DeLong_CI_lower': round(r['ci_lower'], 3),
            'DeLong_CI_upper': round(r['ci_upper'], 3),
            'Boot_CI_lower': round(r['boot_ci_lower'], 3),
            'Boot_CI_upper': round(r['boot_ci_upper'], 3),
            'n_diabetic': r['n_pos'],
            'n_nondiabetic': r['n_neg'],
        })
    summary = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, 'roc_diabetes_summary.csv')
    summary.to_csv(out_path, index=False)
    print(f"\nSaved summary: {out_path}")
    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Stiffness Index \u2014 Diabetes Prediction ROC Analysis")
    print("=" * 60)

    df_all = load_stiffness_data()

    output_dir = os.path.join(cap_flow_path, 'results', 'Stiffness', 'plots')
    minimal_dir = os.path.join(output_dir, 'no_annotations')
    os.makedirs(output_dir, exist_ok=True)

    all_results: List[Dict] = []

    # Encode Sex as numeric for logistic regression (F=0, M=1)
    df_all['Sex_num'] = (df_all['Sex'].str.upper() == 'M').astype(int)

    # Two comparison subsets
    # 1. Diabetic vs healthy controls only (cleanest)
    df_diab_vs_ctrl = df_all[
        (df_all['Diabetes'] == True) | (df_all['is_healthy'] == True)
    ].copy()
    # 2. Diabetic vs all non-diabetic (includes HTN/other)
    df_diab_vs_all = df_all.copy()

    subsets = [
        ('Diab vs Controls', df_diab_vs_ctrl),
        ('Diab vs All Non-diabetic', df_diab_vs_all),
    ]

    # ------------------------------------------------------------------
    # 1 & 2. Single-variable + multi-variable ROC
    # ------------------------------------------------------------------
    for subset_label, df_sub in subsets:
        print(f"\n--- {subset_label} ---")
        n_diab = df_sub['Diabetes'].sum()
        n_non = (~df_sub['Diabetes']).sum()
        print(f"  n_diabetic={n_diab}, n_nondiabetic={n_non}")

        # 1. Single variable
        sv = run_single_variable_roc(df_sub, SI_COL, subset_label)
        if sv:
            all_results.append(sv)
            plot_single_roc(sv, output_dir, minimal_dir)

        # 2. Multi-variable models
        models = [
            ('SI + Age',           [SI_COL, 'Age']),
            ('SI + Sex',           [SI_COL, 'Sex_num']),
            ('SI + Age + Sex',     [SI_COL, 'Age', 'Sex_num']),
            ('SI + Age + Sex + MAP', [SI_COL, 'Age', 'Sex_num', 'MAP']),
        ]

        overlay_list = [sv]
        for model_label, feat_cols in models:
            mv = run_multivariable_roc(
                df_sub, feat_cols, subset_label, model_label)
            if mv:
                all_results.append(mv)
                overlay_list.append(mv)

        # 3. Overlay plot
        plot_roc_overlay(overlay_list, output_dir, minimal_dir)

    # ------------------------------------------------------------------
    # 4. SI Boxplot by Diabetes Status
    # ------------------------------------------------------------------
    print("\n--- SI boxplots by diabetes status ---")
    for subset_label, df_sub in subsets:
        plot_si_boxplot(df_sub, output_dir, minimal_dir, subset_label)

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    build_summary_table(all_results, output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
