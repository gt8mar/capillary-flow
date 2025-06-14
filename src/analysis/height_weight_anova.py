"""
Filename: src/analysis/height_weight_anova.py

This script evaluates whether Height and Weight are significant predictors of blood cell flow velocity
using (A) raw velocity and (B) log-transformed velocity. The workflow mirrors `anova.py`, but focuses on
Height and Weight instead of Age, Sex, and Blood Pressure.

Steps:
1. Load capillary velocity data and merge participant Height/Weight information.
2. Filter to the control cohort (SET == 'set01') and clean the dataset.
3. Fit ordinary least squares (OLS) regression models and generate Type-II ANOVA tables.
4. Display results to the console and optionally save a LaTeX table.
"""

import os
from typing import Dict

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import plot_partregress
from statsmodels.iolib.summary2 import summary_col

# Local imports
from src.config import PATHS
from src.analysis.height_weight_mixed_model import load_and_preprocess_data  # Re-use existing helper

# -----------------------------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------------------------

def perform_height_weight_anova(df: pd.DataFrame, log_transform: bool = False) -> Dict:
    """Run OLS regression and Type-II ANOVA to test Height and Weight effects.

    Args:
        df: Pre-processed dataframe returned by ``load_and_preprocess_data``.
        log_transform: If ``True`` the dependent variable is log-transformed velocity.

    Returns:
        Dictionary with fitted model and ANOVA table.
    """
    dependent = "Log_Video_Median_Velocity" if log_transform else "Video_Median_Velocity"

    formula = f"{dependent} ~ Height + Weight + Age + Pressure"  # adjust for Age & Pressure
    model = smf.ols(formula, data=df).fit()
    anova_tbl = sm.stats.anova_lm(model, typ=2)

    print("\n==================================================================")
    print(f"ANOVA for {'log-transformed' if log_transform else 'raw'} velocity")
    print("Formula:", formula)
    print("------------------------------------------------------------------")
    print(anova_tbl)

    return {"model": model, "anova": anova_tbl}

# -----------------------------------------------------------------------------
# LaTeX export (optional)
# -----------------------------------------------------------------------------

def export_anova_to_latex(anova_tbl: pd.DataFrame, output_path: str, caption_suffix: str = "") -> None:
    """Write a simple LaTeX table for the provided ANOVA results."""
    caption = (
        "ANOVA results for Height and Weight effects on blood flow velocity "
        + caption_suffix
    ).strip()

    with open(output_path, "w") as fh:
        fh.write("% LaTeX table generated by height_weight_anova.py\n")
        fh.write("\\begin{table}[htbp]\n")
        fh.write("\\centering\n")
        fh.write(f"\\caption{{{caption}}}\n")
        fh.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        fh.write("Factor & Sum Sq & DF & F value & PR($>$F) \\\n\\midrule\n")

        for _, row in anova_tbl.reset_index().iterrows():
            factor = row["index"]
            fh.write(
                f"{factor} & {row['sum_sq']:.3f} & {row['df']:.0f} "
                f"& {row['F']:.3f} & {row['PR(>F)']:.4f} \\\n"
            )

        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"LaTeX table written to {output_path}")

# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------

def plot_height_weight_effects(
    df: pd.DataFrame,
    model_raw: sm.regression.linear_model.RegressionResultsWrapper,
    output_dir: str,
) -> None:
    """Create scatter/line plots showing marginal effect of Height and Weight.

    The line shows model predictions while holding other predictors at their mean.
    Raw velocity is used for interpretability.
    """

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 10})

    mean_vals = {
        "Weight": df["Weight"].mean(),
        "Height": df["Height"].mean(),
        "Age": df["Age"].mean(),
        "Pressure": df["Pressure"].mean(),
    }

    # --------------------------------------------------------------
    # Height effect
    # --------------------------------------------------------------
    height_range = np.linspace(df["Height"].min(), df["Height"].max(), 100)
    pred_data = pd.DataFrame({
        "Height": height_range,
        "Weight": mean_vals["Weight"],
        "Age": mean_vals["Age"],
        "Pressure": mean_vals["Pressure"],
    })
    predicted = model_raw.predict(pred_data)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x="Height", y="Video_Median_Velocity", data=df, alpha=0.3, ax=ax)
    ax.plot(height_range, predicted, color="red", linewidth=2)
    ax.set_title("Effect of Height on Blood Flow Velocity (raw model)")
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Velocity (mm/s)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "height_effect_anova.png"), dpi=300)
    plt.close()

    # --------------------------------------------------------------
    # Weight effect
    # --------------------------------------------------------------
    weight_range = np.linspace(df["Weight"].min(), df["Weight"].max(), 100)
    pred_data = pd.DataFrame({
        "Height": mean_vals["Height"],
        "Weight": weight_range,
        "Age": mean_vals["Age"],
        "Pressure": mean_vals["Pressure"],
    })
    predicted = model_raw.predict(pred_data)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x="Weight", y="Video_Median_Velocity", data=df, alpha=0.3, ax=ax)
    ax.plot(weight_range, predicted, color="red", linewidth=2)
    ax.set_title("Effect of Weight on Blood Flow Velocity (raw model)")
    ax.set_xlabel("Weight (kg)")
    ax.set_ylabel("Velocity (mm/s)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weight_effect_anova.png"), dpi=300)
    plt.close()

def plot_added_variable(df: pd.DataFrame, output_dir: str) -> None:
    """Generate added-variable (partial regression) plots for Height and Weight."""

    # Re-fit model in statsmodels API form (raw velocity)
    model = smf.ols(
        "Video_Median_Velocity ~ Height + Weight + Age + Pressure", data=df
    ).fit()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Height partial regression
    plot_partregress(
        "Video_Median_Velocity",
        "Height",
        ["Weight", "Age", "Pressure"],
        data=df,
        ax=axes[0],
    )
    axes[0].set_title("Added-variable plot: Height")

    # Weight partial regression
    plot_partregress(
        "Video_Median_Velocity",
        "Weight",
        ["Height", "Age", "Pressure"],
        data=df,
        ax=axes[1],
    )
    axes[1].set_title("Added-variable plot: Weight")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "added_variable_plots.png"), dpi=300)
    plt.close()

# -----------------------------------------------------------------------------
# Cluster-robust (participant-cluster) OLS
# -----------------------------------------------------------------------------

def perform_cluster_robust_ols(df: pd.DataFrame, log_transform: bool = False) -> dict:
    """Run OLS with participant-clustered robust standard errors.

    Args:
        df: Pre-processed long-format dataframe.
        log_transform: Use log-velocity if True.

    Returns:
        Dict with keys ``model`` (plain OLS) and ``robust`` (cluster-robust results).
    """

    dv = "Log_Video_Median_Velocity" if log_transform else "Video_Median_Velocity"
    formula = f"{dv} ~ Height + Weight + Age + Pressure"

    ols_model = smf.ols(formula, data=df).fit()
    robust_res = ols_model.get_robustcov_results(
        cov_type="cluster", groups=df["Participant"], use_t=True
    )

    print("\n==================================================================")
    label = "log-transformed" if log_transform else "raw"
    print(f"Cluster-robust OLS for {label} velocity (clusters = participant)")
    print("------------------------------------------------------------------")
    print(robust_res.summary())

    return {"model": ols_model, "robust": robust_res}

# -----------------------------------------------------------------------------
# Participant-level aggregation method
# -----------------------------------------------------------------------------

def aggregate_to_participants(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with one row per participant (median velocity, etc.)."""

    agg_df = (
        df.groupby("Participant", as_index=False)
        .agg(
            Video_Median_Velocity=("Video_Median_Velocity", "median"),
            Age=("Age", "first"),
            Sex=("Sex", "first"),
            SYS_BP=("SYS_BP", "median"),
            Height=("Height", "first"),
            Weight=("Weight", "first"),
            Pressure=("Pressure", "median"),
        )
    )

    agg_df["Log_Video_Median_Velocity"] = np.log(agg_df["Video_Median_Velocity"] + 1)
    return agg_df

def perform_aggregated_anova(agg_df: pd.DataFrame, log_transform: bool = False) -> dict:
    """ANOVA on participant-level aggregated data."""

    dv = "Log_Video_Median_Velocity" if log_transform else "Video_Median_Velocity"
    formula = f"{dv} ~ Height + Weight + Age + Pressure"
    model = smf.ols(formula, data=agg_df).fit()
    anova_tbl = sm.stats.anova_lm(model, typ=2)

    print("\n==================================================================")
    label = "log-transformed" if log_transform else "raw"
    print(f"Aggregated-level ANOVA for {label} velocity (n = {len(agg_df)})")
    print("------------------------------------------------------------------")
    print(anova_tbl)

    return {"model": model, "anova": anova_tbl}

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main():
    print("\nRunning Height/Weight ANOVA analysis…")

    # ------------------------------------------------------------------
    # 1. Load and preprocess data (re-use existing helper)
    # ------------------------------------------------------------------
    df = load_and_preprocess_data()

    # ------------------------------------------------------------------
    # 2. Run ANOVA analyses
    # ------------------------------------------------------------------
    results_raw = perform_height_weight_anova(df, log_transform=False)
    results_log = perform_height_weight_anova(df, log_transform=True)

    # ------------------------------------------------------------------
    # 3. (Optional) Write LaTeX tables
    # ------------------------------------------------------------------
    output_dir = os.path.join(PATHS["cap_flow"], "results", "ANOVA_Analysis", "height_weight")
    os.makedirs(output_dir, exist_ok=True)

    export_anova_to_latex(
        results_raw["anova"],
        os.path.join(output_dir, "anova_height_weight_raw.tex"),
        caption_suffix="(raw velocity)",
    )
    export_anova_to_latex(
        results_log["anova"],
        os.path.join(output_dir, "anova_height_weight_log.tex"),
        caption_suffix="(log-transformed velocity)",
    )

    # ------------------------------------------------------------------
    # 4. Visualizations
    # ------------------------------------------------------------------
    plot_height_weight_effects(df, results_raw["model"], output_dir)
    plot_added_variable(df, output_dir)

    # ------------------------------------------------------------------
    # 5. Cluster-robust analysis
    # ------------------------------------------------------------------
    cluster_raw = perform_cluster_robust_ols(df, log_transform=False)
    cluster_log = perform_cluster_robust_ols(df, log_transform=True)

    # ------------------------------------------------------------------
    # 6. Aggregated participant-level analysis
    # ------------------------------------------------------------------
    agg_df = aggregate_to_participants(df)
    agg_raw = perform_aggregated_anova(agg_df, log_transform=False)
    agg_log = perform_aggregated_anova(agg_df, log_transform=True)

    print("\nHeight/Weight ANOVA analysis complete.")
    return 0


if __name__ == "__main__":
    main() 