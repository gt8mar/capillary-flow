"""Exploratory comparison of trapezoidal and equal-pressure hysteresis AUCs.

This script is intentionally separate from the active manuscript analysis. It
uses raw velocity, not log velocity, and compares:

1. Trapezoidal hysteresis AUC: integral of up-down raw velocity differences
   across 0.2--1.2 psi using trapezoidal endpoint weighting.
2. Equal-pressure hysteresis AUC: mean of the up-down raw velocity differences
   at 0.2, 0.4, 0.6, 0.8, 1.0, and 1.2 psi multiplied by the 1.0 psi span.

No existing analysis files are overwritten.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "summary_df_nhp_video_stats.csv"
OUT_DIR = REPO_ROOT / "results" / "exploratory_equal_pressure_hysteresis_auc"
STANDARD_PRESSURES = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=float)
PRESSURE_SPAN = float(STANDARD_PRESSURES.max() - STANDARD_PRESSURES.min())
VALUE_COL = "Video_Median_Velocity"


def parse_bool(value: object) -> bool:
    return str(value).strip().upper() in {"TRUE", "T", "1", "YES", "Y"}


def first_non_null(series: pd.Series) -> object:
    values = series.dropna()
    return values.iloc[0] if len(values) else np.nan


def pressure_differences(group: pd.DataFrame) -> pd.Series:
    """Return mean up-minus-down raw velocity difference at each pressure."""
    diffs: dict[float, float] = {}
    for pressure in STANDARD_PRESSURES:
        at_pressure = group[group["Pressure"].eq(pressure)]
        up = at_pressure.loc[at_pressure["UpDown"].eq("U"), VALUE_COL].mean()
        down = at_pressure.loc[at_pressure["UpDown"].eq("D"), VALUE_COL].mean()
        if pd.notna(up) and pd.notna(down):
            diffs[float(pressure)] = float(up - down)
    return pd.Series(diffs, dtype=float).sort_index()


def trapezoidal_auc(diffs: pd.Series) -> float:
    if len(diffs.dropna()) < 2:
        return np.nan
    clean = diffs.dropna().sort_index()
    return float(np.trapz(clean.to_numpy(dtype=float), clean.index.to_numpy(dtype=float)))


def equal_pressure_auc(diffs: pd.Series) -> float:
    if len(diffs.dropna()) < 2:
        return np.nan
    return float(diffs.dropna().mean() * PRESSURE_SPAN)


def build_participant_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant, group in raw.groupby("Participant", sort=False):
        diffs = pressure_differences(group)
        row = {
            "Participant": participant,
            "hysteresis_auc_trapezoidal_raw": trapezoidal_auc(diffs),
            "hysteresis_auc_equal_pressure_raw": equal_pressure_auc(diffs),
            "n_valid_pressures": int(diffs.dropna().shape[0]),
        }
        row["abs_hysteresis_auc_trapezoidal_raw"] = (
            abs(row["hysteresis_auc_trapezoidal_raw"])
            if pd.notna(row["hysteresis_auc_trapezoidal_raw"])
            else np.nan
        )
        row["abs_hysteresis_auc_equal_pressure_raw"] = (
            abs(row["hysteresis_auc_equal_pressure_raw"])
            if pd.notna(row["hysteresis_auc_equal_pressure_raw"])
            else np.nan
        )
        for pressure in STANDARD_PRESSURES:
            row[f"raw_diff_{pressure:.1f}psi"] = diffs.get(float(pressure), np.nan)
        for col in ["Age", "Diabetes", "Hypertension", "SET", "Sex", "SYS_BP", "DIA_BP"]:
            if col in group.columns:
                row[col] = first_non_null(group[col])
        rows.append(row)

    table = pd.DataFrame(rows)
    table["Age"] = pd.to_numeric(table["Age"], errors="coerce")
    table["SYS_BP"] = pd.to_numeric(table["SYS_BP"], errors="coerce")
    table["DIA_BP"] = pd.to_numeric(table["DIA_BP"], errors="coerce")
    table["Diabetes_bool"] = table["Diabetes"].map(parse_bool)
    table["Hypertension_bool"] = table["Hypertension"].map(parse_bool)
    table["is_control"] = table["SET"].astype(str).eq("set01")
    table["is_diabetes"] = table["SET"].astype(str).eq("set03")
    table["is_nondiabetic_hypertension"] = table["SET"].astype(str).eq("set02")
    return table


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
    slope, _, _, _, stderr = stats.linregress(sub[x_col], sub[metric])
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
    table.insert(1, "analysis", "Controls ANOVA: hysteresis AUC ~ Age + Sex + SYS_BP")
    return table


def pairwise_age_bin_rows(df: pd.DataFrame, metric: str, cohort_label: str, cohort_mask: pd.Series) -> list[dict]:
    sub = df.loc[cohort_mask & df[metric].notna() & df["Age"].notna()].copy()
    sub["age_bin"] = np.select(
        [
            sub["Age"] < 30,
            (sub["Age"] >= 30) & (sub["Age"] < 60),
            sub["Age"] >= 60,
        ],
        ["<30", "30--59", ">=60"],
        default=np.nan,
    )
    rows = []
    for g1, g2 in [("<30", "30--59"), ("<30", ">=60"), ("30--59", ">=60")]:
        x = sub.loc[sub["age_bin"].eq(g1), metric].dropna()
        y = sub.loc[sub["age_bin"].eq(g2), metric].dropna()
        stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append(
            {
                "metric": metric,
                "cohort": cohort_label,
                "group1": g1,
                "group2": g2,
                "n1": len(x),
                "n2": len(y),
                "group1_median_iqr": median_iqr(x),
                "group2_median_iqr": median_iqr(y),
                "statistic": float(stat),
                "p_value": float(p),
            }
        )
    groups = [sub.loc[sub["age_bin"].eq(label), metric].dropna() for label in ["<30", "30--59", ">=60"]]
    rows.append(
        {
            "metric": metric,
            "cohort": cohort_label,
            "group1": "all three age bins",
            "group2": "",
            "n1": int(sum(len(g) for g in groups)),
            "n2": "",
            "group1_median_iqr": "",
            "group2_median_iqr": "",
            "statistic": float(stats.kruskal(*groups).statistic),
            "p_value": float(stats.kruskal(*groups).pvalue),
            "test": "Kruskal-Wallis",
        }
    )
    return rows


def run_tests(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        "hysteresis_auc_trapezoidal_raw",
        "hysteresis_auc_equal_pressure_raw",
        "abs_hysteresis_auc_trapezoidal_raw",
        "abs_hysteresis_auc_equal_pressure_raw",
    ]
    rows = []
    age_rows = []
    for metric in metrics:
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
        if metric.startswith("abs_"):
            age_rows.extend(pairwise_age_bin_rows(metrics_df, metric, "controls", metrics_df["is_control"]))
            age_rows.extend(pairwise_age_bin_rows(metrics_df, metric, "all participants", pd.Series(True, index=metrics_df.index)))

    stats_df = pd.DataFrame(rows)
    anova_df = pd.concat([anova_rows(metrics_df, metric) for metric in metrics], ignore_index=True)
    age_pairwise_df = pd.DataFrame(age_rows)
    return stats_df, anova_df, age_pairwise_df


def markdown_table(df: pd.DataFrame, floatfmt: str = ".6g") -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
        else:
            display[col] = display[col].fillna("").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in display.columns) + " |" for _, row in display.iterrows()]
    return "\n".join([header, separator, *body])


def write_report(metrics_df: pd.DataFrame, stats_df: pd.DataFrame, anova_df: pd.DataFrame, age_pairwise_df: pd.DataFrame) -> None:
    signed = stats_df[stats_df["metric"].isin(["hysteresis_auc_trapezoidal_raw", "hysteresis_auc_equal_pressure_raw"])]
    signed_p = signed.pivot_table(index="analysis", columns="metric", values="p_value", aggfunc="first").reset_index()
    abs_df = stats_df[stats_df["metric"].isin(["abs_hysteresis_auc_trapezoidal_raw", "abs_hysteresis_auc_equal_pressure_raw"])]
    abs_p = abs_df.pivot_table(index="analysis", columns="metric", values="p_value", aggfunc="first").reset_index()

    lines = [
        "# Exploratory Equal-Pressure Hysteresis AUC Check",
        "",
        "This is exploratory only. It does not overwrite the active manuscript analysis.",
        "",
        "Definitions:",
        "- Raw trapezoidal hysteresis AUC: trapezoidal integral of compression-minus-recovery raw velocity differences from 0.2--1.2 psi.",
        "- Raw equal-pressure hysteresis AUC: mean compression-minus-recovery raw velocity difference across 0.2, 0.4, 0.6, 0.8, 1.0, and 1.2 psi, multiplied by the 1.0 psi pressure span. Each pressure receives equal weight, including endpoints.",
        "- Absolute versions take the absolute value after computing the signed participant-level hysteresis AUC.",
        "",
        f"Participants: {len(metrics_df)} total; controls n={int(metrics_df['is_control'].sum())}; diabetes n={int(metrics_df['is_diabetes'].sum())}; non-diabetic hypertension n={int(metrics_df['is_nondiabetic_hypertension'].sum())}.",
        "",
        "## Signed Raw Hysteresis AUC P-values",
        "",
        markdown_table(signed_p),
        "",
        "## Absolute Raw Hysteresis AUC P-values",
        "",
        markdown_table(abs_p),
        "",
        "## Absolute Raw Hysteresis AUC Age-Bin Pairwise P-values",
        "",
        markdown_table(age_pairwise_df),
        "",
        "## Controls ANOVA P-values",
        "",
        markdown_table(anova_df[["metric", "factor", "F", "PR(>F)"]]),
        "",
        "## Participant-Level Agreement",
        "",
        f"Signed AUC Pearson correlation: {metrics_df[['hysteresis_auc_trapezoidal_raw', 'hysteresis_auc_equal_pressure_raw']].corr().iloc[0, 1]:.6f}.",
        f"Absolute AUC Pearson correlation: {metrics_df[['abs_hysteresis_auc_trapezoidal_raw', 'abs_hysteresis_auc_equal_pressure_raw']].corr().iloc[0, 1]:.6f}.",
        f"Median signed trapezoidal AUC: {metrics_df['hysteresis_auc_trapezoidal_raw'].median():.3f}.",
        f"Median signed equal-pressure AUC: {metrics_df['hysteresis_auc_equal_pressure_raw'].median():.3f}.",
        f"Median absolute trapezoidal AUC: {metrics_df['abs_hysteresis_auc_trapezoidal_raw'].median():.3f}.",
        f"Median absolute equal-pressure AUC: {metrics_df['abs_hysteresis_auc_equal_pressure_raw'].median():.3f}.",
        "",
    ]
    (OUT_DIR / "equal_pressure_hysteresis_auc_exploratory_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    raw["Pressure"] = pd.to_numeric(raw["Pressure"], errors="coerce")
    raw[VALUE_COL] = pd.to_numeric(raw[VALUE_COL], errors="coerce")
    raw = raw.dropna(subset=["Participant", "Pressure", "UpDown", VALUE_COL])

    metrics_df = build_participant_table(raw)
    stats_df, anova_df, age_pairwise_df = run_tests(metrics_df)

    metrics_df.to_csv(OUT_DIR / "equal_pressure_hysteresis_auc_participant_metrics.csv", index=False)
    stats_df.to_csv(OUT_DIR / "equal_pressure_hysteresis_auc_significance_comparison.csv", index=False)
    anova_df.to_csv(OUT_DIR / "equal_pressure_hysteresis_auc_anova_comparison.csv", index=False)
    age_pairwise_df.to_csv(OUT_DIR / "equal_pressure_hysteresis_auc_age_pairwise.csv", index=False)
    write_report(metrics_df, stats_df, anova_df, age_pairwise_df)

    print(f"Saved exploratory outputs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
