"""
Filename: src/analysis/stiffness_age_roc.py
--------------------------------------------------

ROC analysis testing whether the Stiffness Index (SI) can predict age.
Biological hypothesis: stiffer capillaries (higher SI) correlate with older age.

Analyses:
  1. Single-variable ROC: SI alone predicting old vs young
  2. Multi-variable ROC: Logistic regression with LOO cross-validation
  3. Overlay ROC plots comparing models
  4. Continuous SI vs Age scatter with regression
  5. Summary CSV table of all AUCs and CIs

By: Marcus Forst
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict
import warnings

from src.config import PATHS, load_source_sans
from src.analysis.plot_stiffness import (
    apply_font, save_figure, _ensure_healthy_flag,
    CONTROL_COLOR, DIABETES_COLOR,
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

# Age thresholds for binary classification
AGE_THRESHOLDS = [40, 50, 59]

# Overlay model colours
MODEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stiffness_data() -> pd.DataFrame:
    """Load stiffness coefficients CSV, drop rows without age, add healthy flag.

    Returns:
        DataFrame with SI metrics, demographics, and ``is_healthy`` column.
    """
    csv_path = os.path.join(cap_flow_path, 'results', 'Stiffness',
                            'stiffness_coefficients_log.csv')
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Age'])
    df = _ensure_healthy_flag(df)
    print(f"Loaded {len(df)} participants with age data "
          f"({df['is_healthy'].sum()} controls)")
    return df


# ---------------------------------------------------------------------------
# AUC / CI helpers (reused from age_score_roc.py)
# ---------------------------------------------------------------------------

def calculate_auc_ci_delong(y_true: np.ndarray, y_scores: np.ndarray,
                            alpha: float = 0.95
                            ) -> Tuple[float, float, float, float]:
    """AUC with Hanley-McNeil approximate CI.

    Returns:
        (auc, auc_var, ci_lower, ci_upper)
    """
    auc_score = roc_auc_score(y_true, y_scores)

    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    q1 = auc_score / (2 - auc_score)
    q2 = 2 * auc_score ** 2 / (1 + auc_score)

    auc_var = (auc_score * (1 - auc_score)
               + (n1 - 1) * (q1 - auc_score ** 2)
               + (n0 - 1) * (q2 - auc_score ** 2)) / (n1 * n0)

    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    auc_std = np.sqrt(max(auc_var, 0))
    ci_lower = max(0.0, auc_score - z * auc_std)
    ci_upper = min(1.0, auc_score + z * auc_std)
    return auc_score, auc_var, ci_lower, ci_upper


def bootstrap_roc(y_true: np.ndarray, y_scores: np.ndarray,
                  n_bootstraps: int = 1000, seed: int = 42
                  ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Bootstrap TPR curves and AUC CI.

    Returns:
        (tprs_array [n_boot x 100], aucs_array, boot_ci_lower, boot_ci_upper)
    """
    rng = np.random.RandomState(seed)
    tprs: List[np.ndarray] = []
    aucs: List[float] = []
    mean_fpr = np.linspace(0, 1, 100)

    for _ in range(n_bootstraps):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        yt, ys = y_true[idx], y_scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            fpr_b, tpr_b, _ = roc_curve(yt, ys)
            tprs.append(np.interp(mean_fpr, fpr_b, tpr_b))
            aucs.append(auc(fpr_b, tpr_b))
        except Exception:
            continue

    tprs_arr = np.array(tprs)
    aucs_arr = np.array(aucs)
    ci_lo = np.percentile(aucs_arr, 2.5)
    ci_hi = np.percentile(aucs_arr, 97.5)
    return tprs_arr, aucs_arr, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# LOO cross-validated logistic regression
# ---------------------------------------------------------------------------

def loo_cv_predict_proba(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out cross-validated predicted probabilities.

    Args:
        X: Feature matrix (n, p).
        y: Binary labels (n,).

    Returns:
        Array of predicted probabilities (n,) — each entry is the
        out-of-sample prediction for that participant.
    """
    n = len(y)
    proba = np.zeros(n)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        if len(np.unique(y_train)) < 2:
            proba[i] = np.nan
            continue
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X[i:i+1])
        clf = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf.fit(X_train_s, y_train)
        proba[i] = clf.predict_proba(X_test_s)[0, 1]
    return proba


# ---------------------------------------------------------------------------
# Single-variable ROC pipeline
# ---------------------------------------------------------------------------

def run_single_variable_roc(df: pd.DataFrame, feature: str,
                            age_threshold: int, subset_label: str
                            ) -> Dict:
    """Compute ROC for a single feature predicting old (>=threshold) vs young.

    Returns:
        Dict with keys: fpr, tpr, auc, ci_lower, ci_upper,
        boot_ci_lower, boot_ci_upper, tprs_boot, n_pos, n_neg,
        threshold, subset, model_label.
    """
    y = (df['Age'] >= age_threshold).astype(int).values
    scores = df[feature].values

    if len(np.unique(y)) < 2:
        warnings.warn(f"Only one class for threshold {age_threshold} "
                      f"in {subset_label} — skipping.")
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
        threshold=age_threshold, subset=subset_label,
        model_label=f'SI alone',
    )


# ---------------------------------------------------------------------------
# Multi-variable ROC pipeline (LOO CV)
# ---------------------------------------------------------------------------

def run_multivariable_roc(df: pd.DataFrame, feature_cols: List[str],
                          age_threshold: int, subset_label: str,
                          model_label: str) -> Dict:
    """Logistic regression ROC with LOO CV.

    Args:
        df: DataFrame (already subset to controls or all).
        feature_cols: Column names to include as predictors.
        age_threshold: Binary label threshold.
        subset_label: 'Controls' or 'All'.
        model_label: Human-readable model name for legends.

    Returns:
        Dict matching run_single_variable_roc output schema.
    """
    sub = df.dropna(subset=feature_cols).copy()
    y = (sub['Age'] >= age_threshold).astype(int).values
    X = sub[feature_cols].values

    if len(np.unique(y)) < 2 or len(sub) < 10:
        warnings.warn(f"Insufficient data for {model_label} at threshold "
                      f"{age_threshold} in {subset_label} — skipping.")
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
        threshold=age_threshold, subset=subset_label,
        model_label=model_label,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _font_kw():
    """Return dict for fontproperties kwarg if font available."""
    if source_sans:
        return {'fontproperties': source_sans}
    return {}


def plot_single_roc(result: Dict, output_dir: str, minimal_dir: str):
    """Plot an individual ROC curve with bootstrap CI shading."""
    if not result:
        return

    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    base_color = '#1f77b4'

    # ROC line
    ax.plot(result['fpr'], result['tpr'], color=base_color,
            label=f"AUC = {result['auc']:.2f} "
                  f"[{result['ci_lower']:.2f}–{result['ci_upper']:.2f}]")

    # Bootstrap CI band
    if len(result['tprs_boot']) > 1:
        lo = np.percentile(result['tprs_boot'], 2.5, axis=0)
        hi = np.percentile(result['tprs_boot'], 97.5, axis=0)
        ax.fill_between(np.linspace(0, 1, 100), lo, hi,
                        color=base_color, alpha=0.2, label='95% CI')

    # Diagonal
    ax.plot([0, 1], [0, 1], 'grey', linestyle='--', label='Random')

    ax.set_xlabel('False Positive Rate', **_font_kw())
    ax.set_ylabel('True Positive Rate', **_font_kw())
    ax.set_title(f"{result['subset']} | age >= {result['threshold']}  "
                 f"(n+={result['n_pos']}, n-={result['n_neg']})",
                 fontsize=7, **_font_kw())
    ax.legend(loc='lower right', fontsize=4,
              prop=source_sans if source_sans else None)
    ax.grid(True, linewidth=0.3)
    apply_font(ax, source_sans)
    plt.tight_layout()

    fname = (f"roc_SI_age{result['threshold']}_"
             f"{result['subset'].lower().replace(' ', '_')}")
    save_figure(fig, output_dir, fname, minimal_dir=minimal_dir)
    plt.close()


def plot_roc_overlay(results_list: List[Dict], output_dir: str,
                     minimal_dir: str, title_suffix: str = '',
                     filename_suffix: str = ''):
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
    ax.set_title(f"{ref['subset']} | age >= {ref['threshold']}"
                 f"{title_suffix}", fontsize=7, **_font_kw())
    ax.legend(loc='lower right', fontsize=4,
              prop=source_sans if source_sans else None)
    ax.grid(True, linewidth=0.3)
    apply_font(ax, source_sans)
    plt.tight_layout()

    fname = (f"roc_overlay_age{ref['threshold']}_"
             f"{ref['subset'].lower().replace(' ', '_')}"
             f"{filename_suffix}")
    save_figure(fig, output_dir, fname, minimal_dir=minimal_dir)
    plt.close()


def plot_si_vs_age_scatter(df: pd.DataFrame, output_dir: str,
                           minimal_dir: str, subset_label: str):
    """Scatter of SI vs Age with linear regression, R², and Spearman rho."""
    si = SI_COL
    sub = df.dropna(subset=[si, 'Age']).copy()
    if len(sub) < 5:
        return

    fig, ax = plt.subplots(figsize=(2.4, 2.0))

    if subset_label == 'Controls':
        ax.scatter(sub['Age'], sub[si], color=CONTROL_COLOR, alpha=0.6,
                   s=20, zorder=3, label='Control')
    else:
        # Colour by health status
        for is_h, color, label in [(True, CONTROL_COLOR, 'Control'),
                                    (False, DIABETES_COLOR, 'Unhealthy')]:
            mask = sub['is_healthy'] == is_h
            ax.scatter(sub.loc[mask, 'Age'], sub.loc[mask, si],
                       color=color, alpha=0.6, s=20, zorder=3, label=label)

    # Linear regression
    x, y = sub['Age'].values, sub[si].values
    slope, intercept, r_value, p_lin, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1,
            alpha=0.7)

    # Spearman
    rho, p_spear = stats.spearmanr(x, y)

    ax.text(0.05, 0.95,
            f'R² = {r_value**2:.3f}  (p = {p_lin:.3g})\n'
            f'ρ = {rho:.3f}  (p = {p_spear:.3g})',
            transform=ax.transAxes, ha='left', va='top', fontsize=5,
            **_font_kw(),
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Age (years)', **_font_kw())
    ax.set_ylabel('SI (log-velocity AUC, 0.2–1.2 psi)', **_font_kw())
    ax.set_title(f'SI vs Age — {subset_label}', fontsize=7, **_font_kw())
    ax.legend(loc='best', fontsize=5,
              prop=source_sans if source_sans else None)
    ax.grid(True, alpha=0.3)
    apply_font(ax, source_sans)
    plt.tight_layout()

    fname = f"scatter_SI_vs_age_{subset_label.lower().replace(' ', '_')}"
    save_figure(fig, output_dir, fname, minimal_dir=minimal_dir)
    plt.close()


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
            'Threshold': r['threshold'],
            'Model': r['model_label'],
            'AUC': round(r['auc'], 3),
            'DeLong_CI_lower': round(r['ci_lower'], 3),
            'DeLong_CI_upper': round(r['ci_upper'], 3),
            'Boot_CI_lower': round(r['boot_ci_lower'], 3),
            'Boot_CI_upper': round(r['boot_ci_upper'], 3),
            'n_old': r['n_pos'],
            'n_young': r['n_neg'],
        })
    summary = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, 'roc_age_summary.csv')
    summary.to_csv(out_path, index=False)
    print(f"\nSaved summary: {out_path}")
    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Stiffness Index — Age Prediction ROC Analysis")
    print("=" * 60)

    df_all = load_stiffness_data()
    df_ctrl = df_all[df_all['is_healthy'] == True].copy()

    output_dir = os.path.join(cap_flow_path, 'results', 'Stiffness', 'plots')
    minimal_dir = os.path.join(output_dir, 'no_annotations')
    os.makedirs(output_dir, exist_ok=True)

    all_results: List[Dict] = []

    # Encode Sex as numeric for logistic regression (F=0, M=1)
    for d in (df_all, df_ctrl):
        d['Sex_num'] = (d['Sex'].str.upper() == 'M').astype(int)

    # ------------------------------------------------------------------
    # 1 & 2. Single-variable + multi-variable ROC
    # ------------------------------------------------------------------
    subsets = [('Controls', df_ctrl), ('All', df_all)]

    for subset_label, df_sub in subsets:
        for thresh in AGE_THRESHOLDS:
            print(f"\n--- {subset_label} | threshold = {thresh} ---")

            # 1. Single variable
            sv = run_single_variable_roc(df_sub, SI_COL, thresh, subset_label)
            if sv:
                all_results.append(sv)
                plot_single_roc(sv, output_dir, minimal_dir)

            # 2. Multi-variable models
            models = [
                ('SI + MAP',  [SI_COL, 'MAP']),
                ('SI + Sex',  [SI_COL, 'Sex_num']),
                ('SI + MAP + Sex', [SI_COL, 'MAP', 'Sex_num']),
            ]
            # Full model with disease flags — all participants only
            if subset_label == 'All':
                models.append(
                    ('SI + MAP + Sex + Diab + HTN',
                     [SI_COL, 'MAP', 'Sex_num', 'Diabetes', 'Hypertension'])
                )

            overlay_list = [sv]  # start overlay with single-variable result
            for model_label, feat_cols in models:
                mv = run_multivariable_roc(
                    df_sub, feat_cols, thresh, subset_label, model_label)
                if mv:
                    all_results.append(mv)
                    overlay_list.append(mv)

            # 3. Overlay plot
            plot_roc_overlay(overlay_list, output_dir, minimal_dir)

    # ------------------------------------------------------------------
    # 4. Continuous scatter: SI vs Age
    # ------------------------------------------------------------------
    print("\n--- SI vs Age scatter plots ---")
    plot_si_vs_age_scatter(df_ctrl, output_dir, minimal_dir, 'Controls')
    plot_si_vs_age_scatter(df_all, output_dir, minimal_dir, 'All')

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    build_summary_table(all_results, output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
