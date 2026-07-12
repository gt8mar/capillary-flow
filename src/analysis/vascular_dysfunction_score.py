"""
Calculate a vascular dysfunction score from CRI and hysteresis penalties.

The score is calibrated to healthy controls and scaled so that controls are
centered near 50. Higher scores indicate worse vascular function.

Inputs:
    summary_df_nhp_video_stats.csv
    results/Stiffness/stiffness_coefficients_log.csv

Outputs:
    results/VascularScore/
"""

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

from src.config import PATHS, load_source_sans


CRI_COL = "CRI"
CRI_SOURCE_COL = "SI_logvel_averaged_02_12"
RAW_DATA_FILE = "summary_df_nhp_video_stats.csv"
STIFFNESS_FILE = os.path.join("results", "Stiffness", "stiffness_coefficients_log.csv")
OUTPUT_DIR_NAME = os.path.join("results", "VascularScore")

WEIGHTS = {
    "z_CRI": 0.70,
    "z_abs_hysteresis": 0.20,
    "z_positive_hysteresis": 0.10,
}

GROUP_ORDER = ["Control", "Diabetic", "Hypertensive", "Other"]
GROUP_PALETTE = {
    "Control": "#4C78A8",
    "Diabetic": "#E45756",
    "Hypertensive": "#F58518",
    "Other": "#72B7B2",
}


def get_cap_flow_path() -> str:
    """Return configured capillary-flow path, falling back to the repo root."""
    configured = PATHS.get("cap_flow")
    if configured and os.path.exists(configured):
        return configured
    return str(Path(__file__).resolve().parents[2])


def to_bool(value):
    """Convert mixed bool/string health columns to bool or NaN."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().upper()
    if text in {"TRUE", "T", "1", "YES", "Y"}:
        return True
    if text in {"FALSE", "F", "0", "NO", "N"}:
        return False
    return np.nan


def setup_plotting() -> None:
    """Set compact plotting defaults consistent with other analysis scripts."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 0.9,
        }
    )


def save_figure(fig, output_dir: str, filename_stem: str) -> None:
    """Save PNG and PDF copies of a figure."""
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{filename_stem}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{filename_stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def load_cri_data(cap_flow_path: str) -> pd.DataFrame:
    """Load participant-level CRI and metadata."""
    path = os.path.join(cap_flow_path, STIFFNESS_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing CRI file: {path}. Run src/analysis/stiffness_coeff.py first."
        )

    df = pd.read_csv(path)
    required = ["Participant", CRI_SOURCE_COL, "Age", "Diabetes", "Hypertension", "SET"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df[CRI_COL] = pd.to_numeric(df[CRI_SOURCE_COL], errors="coerce")
    for col in ["Age", "SYS_BP", "DIA_BP", "MAP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Diabetes", "Hypertension", "is_healthy"]:
        if col in df.columns:
            df[col] = df[col].map(to_bool)
    if "is_healthy" not in df.columns:
        df["is_healthy"] = df["SET"].astype(str).str.startswith("set01")

    keep_cols = [
        "Participant",
        CRI_COL,
        "Age",
        "Diabetes",
        "Hypertension",
        "SET",
        "Sex",
        "SYS_BP",
        "DIA_BP",
        "MAP",
        "is_healthy",
    ]
    return df[[col for col in keep_cols if col in df.columns]].copy()


def calculate_raw_hysteresis(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate raw velocity hysteresis as mean up velocity minus mean down velocity."""
    required = ["Participant", "UpDown", "Video_Median_Velocity"]
    missing = [col for col in required if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required raw hysteresis columns: {missing}")

    df = raw_df.copy()
    df["Video_Median_Velocity"] = pd.to_numeric(df["Video_Median_Velocity"], errors="coerce")
    df = df.dropna(subset=["Participant", "UpDown", "Video_Median_Velocity"])

    rows = []
    for participant, participant_df in df.groupby("Participant", sort=False):
        up = participant_df.loc[participant_df["UpDown"] == "U", "Video_Median_Velocity"]
        down = participant_df.loc[participant_df["UpDown"] == "D", "Video_Median_Velocity"]
        up_mean = float(up.mean()) if len(up) else np.nan
        down_mean = float(down.mean()) if len(down) else np.nan
        hysteresis = up_mean - down_mean if pd.notna(up_mean) and pd.notna(down_mean) else np.nan
        rows.append(
            {
                "Participant": participant,
                "up_velocity_mean": up_mean,
                "down_velocity_mean": down_mean,
                "up_down_diff": hysteresis,
                "abs_hysteresis": abs(hysteresis) if pd.notna(hysteresis) else np.nan,
                "positive_hysteresis": max(hysteresis, 0.0) if pd.notna(hysteresis) else np.nan,
                "n_up_velocity_rows": int(len(up)),
                "n_down_velocity_rows": int(len(down)),
            }
        )
    return pd.DataFrame(rows)


def load_raw_data(cap_flow_path: str) -> pd.DataFrame:
    """Load raw participant/video summary data."""
    path = os.path.join(cap_flow_path, RAW_DATA_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing raw data file: {path}")
    return pd.read_csv(path, low_memory=False)


def control_z_scores(df: pd.DataFrame, cols: Iterable[str]) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Add control-calibrated z-score columns for each source column."""
    scored = df.copy()
    controls = scored[scored["is_healthy"] == True]
    calibration = {}

    for col in cols:
        control_values = controls[col].dropna()
        mean = float(control_values.mean())
        sd = float(control_values.std(ddof=1))
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(f"Cannot z-score {col}: control SD is zero or non-finite.")
        z_col = f"z_{col}" if col != CRI_COL else "z_CRI"
        scored[z_col] = (scored[col] - mean) / sd
        calibration[col] = {"control_mean": mean, "control_sd": sd, "z_column": z_col}

    return scored, calibration


def assign_group(row: pd.Series) -> str:
    """Assign a display group with diabetes taking precedence over hypertension."""
    if row.get("is_healthy") is True:
        return "Control"
    if row.get("Diabetes") is True:
        return "Diabetic"
    if row.get("Hypertension") is True:
        return "Hypertensive"
    return "Other"


def calculate_vascular_score(cri_df: pd.DataFrame, hysteresis_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Merge metrics, calculate components, and return participant-level scores."""
    df = cri_df.merge(hysteresis_df, on="Participant", how="inner")
    df = df.dropna(subset=[CRI_COL, "up_down_diff", "abs_hysteresis", "positive_hysteresis"])
    df["group"] = df.apply(assign_group, axis=1)

    scored, calibration = control_z_scores(
        df,
        [CRI_COL, "abs_hysteresis", "positive_hysteresis"],
    )
    scored["CRI_component"] = WEIGHTS["z_CRI"] * scored["z_CRI"]
    scored["abs_hysteresis_component"] = (
        WEIGHTS["z_abs_hysteresis"] * scored["z_abs_hysteresis"]
    )
    scored["positive_hysteresis_component"] = (
        WEIGHTS["z_positive_hysteresis"] * scored["z_positive_hysteresis"]
    )
    scored["VDS_z"] = (
        scored["CRI_component"]
        + scored["abs_hysteresis_component"]
        + scored["positive_hysteresis_component"]
    )
    scored["Vascular_Dysfunction_Score"] = 50 + 10 * scored["VDS_z"]
    return scored, calibration


def q1(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def q3(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize score distributions by display group and binary disease flags."""
    rows = []
    summary_specs = [("group", group) for group in GROUP_ORDER if (df["group"] == group).any()]
    summary_specs.extend(
        [
            ("Diabetes", True),
            ("Diabetes", False),
            ("Hypertension", True),
            ("Hypertension", False),
        ]
    )

    for col, value in summary_specs:
        subset = df[df[col] == value].copy()
        if subset.empty:
            continue
        score = subset["Vascular_Dysfunction_Score"].dropna()
        rows.append(
            {
                "grouping": col,
                "group": str(value),
                "n": int(len(score)),
                "mean": float(score.mean()),
                "sd": float(score.std(ddof=1)) if len(score) > 1 else np.nan,
                "median": float(score.median()),
                "q1": q1(score),
                "q3": q3(score),
                "iqr": q3(score) - q1(score),
                "min": float(score.min()),
                "max": float(score.max()),
            }
        )
    return pd.DataFrame(rows)


def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Calculate Cohen's d for two independent samples."""
    x = group1.dropna().to_numpy(dtype=float)
    y = group2.dropna().to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
    return float((np.mean(x) - np.mean(y)) / pooled) if pooled > 0 else np.nan


def mann_whitney_row(df: pd.DataFrame, name: str, mask1, mask2, group1: str, group2: str) -> Dict:
    """Run a two-sided Mann-Whitney U comparison for vascular score."""
    values1 = df.loc[mask1, "Vascular_Dysfunction_Score"].dropna()
    values2 = df.loc[mask2, "Vascular_Dysfunction_Score"].dropna()
    if len(values1) == 0 or len(values2) == 0:
        stat, p_value = np.nan, np.nan
    else:
        stat, p_value = stats.mannwhitneyu(values1, values2, alternative="two-sided")

    return {
        "comparison": name,
        "test": "Mann-Whitney U",
        "group1": group1,
        "group2": group2,
        "n1": int(len(values1)),
        "n2": int(len(values2)),
        "median1": float(values1.median()) if len(values1) else np.nan,
        "median2": float(values2.median()) if len(values2) else np.nan,
        "statistic": float(stat) if pd.notna(stat) else np.nan,
        "p_value": float(p_value) if pd.notna(p_value) else np.nan,
        "cohens_d_group2_minus_group1": cohens_d(values2, values1),
    }


def compare_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Build the requested disease and hypertension comparisons."""
    comparisons = [
        (
            "Control vs Diabetic",
            df["group"] == "Control",
            df["Diabetes"] == True,
            "Control",
            "Diabetic",
        ),
        (
            "Control vs Hypertensive",
            df["group"] == "Control",
            (df["Hypertension"] == True) & (df["Diabetes"] != True),
            "Control",
            "Hypertensive non-diabetic",
        ),
        (
            "Control vs Disease/Other non-control",
            df["group"] == "Control",
            df["group"] != "Control",
            "Control",
            "Non-control",
        ),
        (
            "Diabetic vs Non-diabetic",
            df["Diabetes"] != True,
            df["Diabetes"] == True,
            "Non-diabetic",
            "Diabetic",
        ),
        (
            "Hypertensive vs Non-hypertensive",
            df["Hypertension"] != True,
            df["Hypertension"] == True,
            "Non-hypertensive",
            "Hypertensive",
        ),
    ]
    rows = [mann_whitney_row(df, *comparison) for comparison in comparisons]
    return pd.DataFrame(rows)


def write_calibration_report(
    output_dir: str,
    calibration: Dict,
    score_df: pd.DataFrame,
    significance_df: pd.DataFrame,
) -> None:
    """Write a short text report describing the score and main group results."""
    path = os.path.join(output_dir, "vascular_dysfunction_score_report.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Vascular Dysfunction Score Report\n")
        handle.write("=" * 35 + "\n\n")
        handle.write("Score definition:\n")
        handle.write(
            "VDS_z = 0.70*z_control(CRI) + 0.20*z_control(|H|) "
            "+ 0.10*z_control(max(H, 0))\n"
        )
        handle.write("Vascular_Dysfunction_Score = 50 + 10*VDS_z\n")
        handle.write("H = mean(up raw velocity) - mean(down raw velocity)\n")
        handle.write("Higher scores indicate worse vascular function.\n\n")

        handle.write("Control calibration:\n")
        for col, values in calibration.items():
            handle.write(
                f"  {col}: mean = {values['control_mean']:.6g}, "
                f"SD = {values['control_sd']:.6g}\n"
            )
        handle.write("\n")

        controls = score_df[score_df["group"] == "Control"]["Vascular_Dysfunction_Score"]
        handle.write(
            f"Controls: n={len(controls)}, mean={controls.mean():.3f}, "
            f"median={controls.median():.3f}\n\n"
        )

        handle.write("Mann-Whitney U comparisons:\n")
        handle.write(significance_df.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
        handle.write("\n")


def write_age_adjusted_models(df: pd.DataFrame, output_dir: str) -> None:
    """Fit age-adjusted OLS models and write text summaries."""
    path = os.path.join(output_dir, "vascular_dysfunction_age_adjusted_models.txt")
    model_specs = [
        ("Diabetes age-adjusted", "Vascular_Dysfunction_Score ~ Diabetes_int + Age"),
        ("Hypertension age-adjusted", "Vascular_Dysfunction_Score ~ Hypertension_int + Age"),
        (
            "Diabetes + Hypertension age-adjusted",
            "Vascular_Dysfunction_Score ~ Diabetes_int + Hypertension_int + Age",
        ),
    ]

    model_df = df.copy()
    model_df["Diabetes_int"] = (model_df["Diabetes"] == True).astype(int)
    model_df["Hypertension_int"] = (model_df["Hypertension"] == True).astype(int)

    with open(path, "w", encoding="utf-8") as handle:
        for title, formula in model_specs:
            sub = model_df.dropna(subset=["Vascular_Dysfunction_Score", "Age"]).copy()
            model = smf.ols(formula, data=sub).fit()
            handle.write(title + "\n")
            handle.write("=" * len(title) + "\n")
            handle.write(f"Formula: {formula}\n")
            handle.write(f"n = {int(model.nobs)}\n")
            handle.write(f"R-squared = {model.rsquared:.5f}\n")
            handle.write(f"Adjusted R-squared = {model.rsquared_adj:.5f}\n\n")
            handle.write(str(model.summary()))
            handle.write("\n\n")


def plot_score_by_group(df: pd.DataFrame, output_dir: str) -> None:
    """Plot score by multi-class participant group."""
    present_order = [group for group in GROUP_ORDER if group in set(df["group"])]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    sns.boxplot(
        data=df,
        x="group",
        y="Vascular_Dysfunction_Score",
        order=present_order,
        palette=[GROUP_PALETTE[group] for group in present_order],
        width=0.6,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="group",
        y="Vascular_Dysfunction_Score",
        order=present_order,
        color="black",
        alpha=0.45,
        size=3,
        jitter=0.12,
        ax=ax,
    )
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Vascular Dysfunction Score")
    ax.set_title("Vascular Dysfunction Score by Group")
    save_figure(fig, output_dir, "vascular_score_by_group")


def plot_score_by_binary(df: pd.DataFrame, output_dir: str, col: str, filename: str) -> None:
    """Plot score by a binary disease flag."""
    plot_df = df.copy()
    plot_df[col] = np.where(plot_df[col] == True, col, f"No {col}")
    order = [f"No {col}", col]
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    sns.boxplot(data=plot_df, x=col, y="Vascular_Dysfunction_Score", order=order, width=0.6, ax=ax)
    sns.stripplot(
        data=plot_df,
        x=col,
        y="Vascular_Dysfunction_Score",
        order=order,
        color="black",
        alpha=0.45,
        size=3,
        jitter=0.12,
        ax=ax,
    )
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Vascular Dysfunction Score")
    ax.set_title(f"Vascular Score by {col}")
    save_figure(fig, output_dir, filename)


def plot_score_vs_age(df: pd.DataFrame, output_dir: str) -> None:
    """Plot vascular score against chronological age."""
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    sns.scatterplot(
        data=df,
        x="Age",
        y="Vascular_Dysfunction_Score",
        hue="group",
        hue_order=[group for group in GROUP_ORDER if group in set(df["group"])],
        palette=GROUP_PALETTE,
        s=35,
        alpha=0.8,
        ax=ax,
    )
    sns.regplot(data=df, x="Age", y="Vascular_Dysfunction_Score", scatter=False, color="black", ax=ax)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Vascular Dysfunction Score")
    ax.set_title("Vascular Score vs Age")
    save_figure(fig, output_dir, "vascular_score_vs_age")


def plot_cri_vs_hysteresis(df: pd.DataFrame, output_dir: str) -> None:
    """Plot CRI versus signed hysteresis colored by score."""
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    scatter = ax.scatter(
        df["up_down_diff"],
        df[CRI_COL],
        c=df["Vascular_Dysfunction_Score"],
        cmap="viridis",
        s=35,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.2,
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Hysteresis, up - down velocity")
    ax.set_ylabel("CRI")
    ax.set_title("CRI vs Hysteresis")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Vascular Dysfunction Score")
    save_figure(fig, output_dir, "cri_vs_hysteresis_colored_by_vascular_score")


def plot_component_contributions(df: pd.DataFrame, output_dir: str) -> None:
    """Plot sorted weighted score components for each participant."""
    plot_df = df.sort_values("Vascular_Dysfunction_Score").copy()
    x = np.arange(len(plot_df))
    components = [
        ("CRI_component", "CRI"),
        ("abs_hysteresis_component", "|H|"),
        ("positive_hysteresis_component", "positive H"),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    bottom_pos = np.zeros(len(plot_df))
    bottom_neg = np.zeros(len(plot_df))
    colors = ["#4C78A8", "#54A24B", "#E45756"]

    for (col, label), color in zip(components, colors):
        values = plot_df[col].to_numpy()
        positive = np.where(values > 0, values, 0)
        negative = np.where(values < 0, values, 0)
        ax.bar(x, positive, bottom=bottom_pos, color=color, width=0.85, label=label)
        ax.bar(x, negative, bottom=bottom_neg, color=color, width=0.85)
        bottom_pos += positive
        bottom_neg += negative

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Participant"], rotation=90, fontsize=5)
    ax.set_ylabel("Weighted z-score contribution")
    ax.set_title("Vascular Score Component Contributions")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save_figure(fig, output_dir, "vascular_score_component_contributions")


def make_plots(df: pd.DataFrame, output_dir: str) -> None:
    """Generate all vascular score plots."""
    plot_score_by_group(df, output_dir)
    plot_score_by_binary(df, output_dir, "Diabetes", "vascular_score_by_diabetes")
    plot_score_by_binary(df, output_dir, "Hypertension", "vascular_score_by_hypertension")
    plot_score_vs_age(df, output_dir)
    plot_cri_vs_hysteresis(df, output_dir)
    plot_component_contributions(df, output_dir)


def main() -> int:
    """Run vascular dysfunction score analysis."""
    print("\nCalculating vascular dysfunction scores...")
    setup_plotting()
    cap_flow_path = get_cap_flow_path()
    output_dir = os.path.join(cap_flow_path, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)

    raw_df = load_raw_data(cap_flow_path)
    cri_df = load_cri_data(cap_flow_path)
    hysteresis_df = calculate_raw_hysteresis(raw_df)
    score_df, calibration = calculate_vascular_score(cri_df, hysteresis_df)

    score_path = os.path.join(output_dir, "vascular_dysfunction_scores.csv")
    summary_path = os.path.join(output_dir, "vascular_dysfunction_group_summary.csv")
    sig_path = os.path.join(output_dir, "vascular_dysfunction_significance.csv")
    calibration_path = os.path.join(output_dir, "vascular_dysfunction_calibration.csv")

    summary_df = summarize_groups(score_df)
    significance_df = compare_groups(score_df)
    calibration_df = pd.DataFrame(
        [
            {"metric": metric, **values}
            for metric, values in calibration.items()
        ]
    )

    score_df.to_csv(score_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    significance_df.to_csv(sig_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)
    write_calibration_report(output_dir, calibration, score_df, significance_df)
    write_age_adjusted_models(score_df, output_dir)
    make_plots(score_df, output_dir)

    print(f"Saved participant scores: {score_path}")
    print(f"Saved group summary: {summary_path}")
    print(f"Saved significance table: {sig_path}")
    print(f"Saved plots to: {output_dir}")
    print("\nGroup summary:")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nSignificance:")
    print(significance_df.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
