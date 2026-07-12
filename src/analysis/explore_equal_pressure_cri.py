"""Exploratory comparison of trapezoidal CRI and equal-pressure CRI.

This script does not overwrite the active CRI analysis. It recomputes the
participant-level CRI from summary_df_nhp_video_stats.csv using the current
trapezoidal definition and an exploratory equal-pressure version where each
pressure from 0.2 to 1.2 psi receives the same weight.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "summary_df_nhp_video_stats.csv"
OUT_DIR = REPO_ROOT / "results" / "exploratory_equal_pressure_cri"
PRESSURES = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=float)
PRESSURE_RANGE = float(PRESSURES.max() - PRESSURES.min())
LOG_VELOCITY_COL = "Log_Video_Median_Velocity"


def parse_bool(value: object) -> bool:
    return str(value).strip().upper() in {"TRUE", "T", "1", "YES", "Y"}


def get_curves(participant_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    up = (
        participant_df[participant_df["UpDown"] == "U"]
        .groupby("Pressure")[LOG_VELOCITY_COL]
        .mean()
        .sort_index()
    )
    down = (
        participant_df[participant_df["UpDown"] == "D"]
        .groupby("Pressure")[LOG_VELOCITY_COL]
        .mean()
        .sort_index()
    )
    return (
        up.index.to_numpy(dtype=float),
        up.to_numpy(dtype=float),
        down.index.to_numpy(dtype=float),
        down.to_numpy(dtype=float),
    )


def average_curve_at_targets(
    up_p: np.ndarray,
    up_v: np.ndarray,
    down_p: np.ndarray,
    down_v: np.ndarray,
    targets: np.ndarray = PRESSURES,
) -> np.ndarray:
    curves = []
    if len(up_p) >= 2:
        curves.append(np.interp(targets, up_p, up_v, left=np.nan, right=np.nan))
    if len(down_p) >= 2:
        curves.append(np.interp(targets, down_p, down_v, left=np.nan, right=np.nan))
    if not curves:
        return np.full(len(targets), np.nan)
    stacked = np.vstack(curves)
    counts = np.sum(~np.isnan(stacked), axis=0)
    summed = np.nansum(stacked, axis=0)
    averaged = np.full(len(targets), np.nan)
    averaged[counts > 0] = summed[counts > 0] / counts[counts > 0]
    return averaged


def trapezoidal_cri(avg_curve: np.ndarray, targets: np.ndarray = PRESSURES) -> float:
    valid = ~np.isnan(avg_curve)
    if valid.sum() < 2:
        return np.nan
    return float(np.trapz(avg_curve[valid], targets[valid]))


def equal_pressure_cri(avg_curve: np.ndarray) -> float:
    """Equal-pressure CRI: mean log velocity across pressures times pressure range."""
    valid = ~np.isnan(avg_curve)
    if valid.sum() < 2:
        return np.nan
    return float(np.nanmean(avg_curve[valid]) * PRESSURE_RANGE)


def participant_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant, part_df in df.groupby("Participant", sort=False):
        up_p, up_v, down_p, down_v = get_curves(part_df)
        avg_curve = average_curve_at_targets(up_p, up_v, down_p, down_v)
        row = {
            "Participant": participant,
            "CRI_trapezoidal": trapezoidal_cri(avg_curve),
            "CRI_equal_pressure": equal_pressure_cri(avg_curve),
            "n_valid_pressures": int((~np.isnan(avg_curve)).sum()),
        }
        for col in ["Age", "Diabetes", "Hypertension", "SET", "Sex", "SYS_BP", "DIA_BP"]:
            if col in part_df.columns:
                row[col] = part_df[col].iloc[0]
        rows.append(row)

    out = pd.DataFrame(rows)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["SYS_BP"] = pd.to_numeric(out["SYS_BP"], errors="coerce")
    out["DIA_BP"] = pd.to_numeric(out["DIA_BP"], errors="coerce")
    out["Diabetes_bool"] = out["Diabetes"].map(parse_bool)
    out["Hypertension_bool"] = out["Hypertension"].map(parse_bool)
    out["is_control"] = out["SET"].astype(str).eq("set01")
    out["is_diabetes"] = out["SET"].astype(str).eq("set03")
    out["is_nondiabetic_hypertension"] = out["SET"].astype(str).eq("set02")
    return out


def median_iqr(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "NA"
    q1, q3 = clean.quantile([0.25, 0.75])
    return f"{clean.median():.3f} ({q1:.3f}--{q3:.3f})"


def mw_row(df: pd.DataFrame, metric: str, analysis: str, mask1: pd.Series, mask2: pd.Series, g1: str, g2: str) -> dict:
    x = df.loc[mask1, metric].dropna()
    y = df.loc[mask2, metric].dropna()
    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    return {
        "metric": metric,
        "analysis": analysis,
        "test": "Mann-Whitney U",
        "group1": g1,
        "group2": g2,
        "n1": len(x),
        "n2": len(y),
        "group1_median_iqr": median_iqr(x),
        "group2_median_iqr": median_iqr(y),
        "statistic": float(stat),
        "p_value": float(p),
    }


def regression_row(df: pd.DataFrame, metric: str, analysis: str, mask: pd.Series, x_col: str) -> dict:
    sub = df.loc[mask, [metric, x_col]].dropna()
    r, p = stats.pearsonr(sub[x_col], sub[metric])
    slope, intercept, _, _, stderr = stats.linregress(sub[x_col], sub[metric])
    return {
        "metric": metric,
        "analysis": analysis,
        "test": "Pearson linear regression",
        "group1": metric,
        "group2": x_col,
        "n1": len(sub),
        "n2": len(sub),
        "group1_median_iqr": "",
        "group2_median_iqr": "",
        "statistic": float(r),
        "p_value": float(p),
        "slope": float(slope),
        "slope_se": float(stderr),
    }


def age_adjusted_row(df: pd.DataFrame, metric: str, analysis: str, mask: pd.Series, group_col: str) -> dict:
    sub = df.loc[mask, [metric, "Age", group_col]].dropna().copy()
    sub["Group"] = sub[group_col].astype(bool).astype(int)
    model = ols(f"{metric} ~ Group + Age", data=sub).fit()
    return {
        "metric": metric,
        "analysis": analysis,
        "test": f"OLS {metric} ~ Group + Age",
        "group1": "Group",
        "group2": "Age",
        "n1": int(model.nobs),
        "n2": int(model.nobs),
        "group1_median_iqr": "",
        "group2_median_iqr": "",
        "statistic": float(model.params["Group"]),
        "p_value": float(model.pvalues["Group"]),
        "age_coef": float(model.params["Age"]),
        "age_p_value": float(model.pvalues["Age"]),
        "r_squared": float(model.rsquared),
    }


def anova_rows(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df.loc[df["is_control"], [metric, "Age", "Sex", "SYS_BP"]].dropna().copy()
    model = ols(f"{metric} ~ Age + C(Sex) + SYS_BP", data=sub).fit()
    table = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "factor"})
    table.insert(0, "metric", metric)
    table.insert(1, "analysis", "Controls ANOVA: CRI ~ Age + Sex + SYS_BP")
    return table


def run_tests(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for metric in ["CRI_trapezoidal", "CRI_equal_pressure"]:
        rows.extend(
            [
                mw_row(
                    metrics_df,
                    metric,
                    "Diabetes vs healthy controls",
                    metrics_df["is_control"],
                    metrics_df["is_diabetes"],
                    "Control",
                    "Diabetes",
                ),
                mw_row(
                    metrics_df,
                    metric,
                    "Non-diabetic hypertension vs healthy controls",
                    metrics_df["is_control"],
                    metrics_df["is_nondiabetic_hypertension"],
                    "Control",
                    "Non-diabetic hypertension",
                ),
                mw_row(
                    metrics_df,
                    metric,
                    "Controls, age threshold 30",
                    metrics_df["is_control"] & (metrics_df["Age"] < 30),
                    metrics_df["is_control"] & (metrics_df["Age"] >= 30),
                    "<30",
                    ">=30",
                ),
                mw_row(
                    metrics_df,
                    metric,
                    "Controls, age threshold 60",
                    metrics_df["is_control"] & (metrics_df["Age"] < 60),
                    metrics_df["is_control"] & (metrics_df["Age"] >= 60),
                    "<60",
                    ">=60",
                ),
                regression_row(
                    metrics_df,
                    metric,
                    "Metric vs Age in controls",
                    metrics_df["is_control"],
                    "Age",
                ),
                regression_row(
                    metrics_df,
                    metric,
                    "Metric vs SYS_BP in controls",
                    metrics_df["is_control"],
                    "SYS_BP",
                ),
                age_adjusted_row(
                    metrics_df,
                    metric,
                    "Age-adjusted diabetes vs healthy controls",
                    metrics_df["is_control"] | metrics_df["is_diabetes"],
                    "is_diabetes",
                ),
                age_adjusted_row(
                    metrics_df,
                    metric,
                    "Age-adjusted non-diabetic hypertension vs healthy controls",
                    metrics_df["is_control"] | metrics_df["is_nondiabetic_hypertension"],
                    "is_nondiabetic_hypertension",
                ),
            ]
        )
    stats_df = pd.DataFrame(rows)
    anova_df = pd.concat(
        [anova_rows(metrics_df, "CRI_trapezoidal"), anova_rows(metrics_df, "CRI_equal_pressure")],
        ignore_index=True,
    )
    return stats_df, anova_df


def write_report(metrics_df: pd.DataFrame, stats_df: pd.DataFrame, anova_df: pd.DataFrame) -> None:
    comparison = stats_df.pivot_table(
        index="analysis",
        columns="metric",
        values="p_value",
        aggfunc="first",
    ).reset_index()
    comparison["delta_equal_minus_trap"] = (
        comparison["CRI_equal_pressure"] - comparison["CRI_trapezoidal"]
    )

    def markdown_table(df: pd.DataFrame, floatfmt: str = ".6g") -> str:
        display = df.copy()
        for col in display.columns:
            if pd.api.types.is_numeric_dtype(display[col]):
                display[col] = display[col].map(
                    lambda x: "" if pd.isna(x) else format(float(x), floatfmt)
                )
            else:
                display[col] = display[col].fillna("").astype(str)
        header = "| " + " | ".join(display.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
        body = [
            "| " + " | ".join(str(row[col]) for col in display.columns) + " |"
            for _, row in display.iterrows()
        ]
        return "\n".join([header, separator, *body])

    lines = [
        "# Exploratory Equal-Pressure CRI Check",
        "",
        "This is exploratory only. It does not overwrite the active CRI analysis.",
        "",
        "Definitions:",
        "- Current CRI: trapezoidal AUC of averaged up/down log velocity at 0.2--1.2 psi.",
        "- Exploratory equal-pressure CRI: mean averaged up/down log velocity across 0.2, 0.4, 0.6, 0.8, 1.0, and 1.2 psi, multiplied by the 1.0 psi pressure span. Each pressure receives equal weight, including endpoints.",
        "",
        f"Participants: {len(metrics_df)} total; controls n={int(metrics_df['is_control'].sum())}; diabetes n={int(metrics_df['is_diabetes'].sum())}; non-diabetic hypertension n={int(metrics_df['is_nondiabetic_hypertension'].sum())}.",
        "",
        "## P-value Comparison",
        "",
        markdown_table(comparison),
        "",
        "## ANOVA P-values",
        "",
    ]
    anova_p = anova_df[["metric", "factor", "F", "PR(>F)"]].copy()
    lines.append(markdown_table(anova_p))
    lines.extend(
        [
            "",
            "## Participant-Level Agreement",
            "",
            f"Pearson correlation between current and equal-pressure CRI: {metrics_df[['CRI_trapezoidal', 'CRI_equal_pressure']].corr().iloc[0, 1]:.6f}.",
            f"Median current CRI: {metrics_df['CRI_trapezoidal'].median():.3f}.",
            f"Median equal-pressure CRI: {metrics_df['CRI_equal_pressure'].median():.3f}.",
            "",
        ]
    )
    (OUT_DIR / "equal_pressure_cri_exploratory_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    if LOG_VELOCITY_COL not in df.columns:
        df[LOG_VELOCITY_COL] = np.log(pd.to_numeric(df["Video_Median_Velocity"], errors="coerce"))
    df["Pressure"] = pd.to_numeric(df["Pressure"], errors="coerce")
    df[LOG_VELOCITY_COL] = pd.to_numeric(df[LOG_VELOCITY_COL], errors="coerce")
    df = df.dropna(subset=["Participant", "Pressure", "UpDown", LOG_VELOCITY_COL])

    metrics_df = participant_metrics(df)
    stats_df, anova_df = run_tests(metrics_df)

    metrics_df.to_csv(OUT_DIR / "equal_pressure_cri_participant_metrics.csv", index=False)
    stats_df.to_csv(OUT_DIR / "equal_pressure_cri_significance_comparison.csv", index=False)
    anova_df.to_csv(OUT_DIR / "equal_pressure_cri_anova_comparison.csv", index=False)
    write_report(metrics_df, stats_df, anova_df)

    print(f"Saved exploratory outputs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
