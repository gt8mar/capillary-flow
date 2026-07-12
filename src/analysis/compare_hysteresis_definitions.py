"""Compare manuscript hysteresis definitions and classifier utility.

This script is deliberately scoped to the current manuscript correction work.
It computes participant-level hysteresis with equal pressure weighting, compares
age-bin p-values in controls, and checks whether hysteresis variants improve
diabetes classifiers relative to selected pressure features alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import kruskal, mannwhitneyu
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "summary_df_nhp_video_stats.csv"
RESULTS_DIR = REPO_ROOT / "results" / "part30_diabetes_correction"
CLASSIFIER_DIR = (
    REPO_ROOT
    / "results"
    / "Classifier"
    / "Velocity"
    / "Diabetes"
    / "hysteresis_metric_comparison"
)
DOCS_DIR = REPO_ROOT / "docs" / "current_draft" / "part30_correction_updates"
STANDARD_PRESSURES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]


def bool_true(value: object) -> bool:
    return str(value).strip().upper() in {"TRUE", "1", "YES", "Y"}


def first_non_null(series: pd.Series) -> object:
    values = series.dropna()
    return values.iloc[0] if len(values) else np.nan


def pressure_label(pressure: float) -> str:
    return f"{pressure:.1f}psi"


def read_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    for col in ["Age", "Pressure", "Video_Median_Velocity", "Log_Video_Median_Velocity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Log_Video_Median_Velocity"] = df["Log_Video_Median_Velocity"].clip(lower=0)
    for col in ["Diabetes", "Hypertension"]:
        df[col] = df[col].map(bool_true)
    return df


def mean_by_pressure_difference(group: pd.DataFrame, value_col: str) -> tuple[float, int]:
    """Return equal-pressure-weighted mean of up-down differences.

    Each pressure contributes one value: mean(up videos at pressure) minus
    mean(down videos at pressure). The participant hysteresis is the simple
    mean of those per-pressure differences.
    """
    diffs: list[float] = []
    for pressure in STANDARD_PRESSURES:
        at_pressure = group[group["Pressure"].eq(pressure)]
        up = at_pressure.loc[at_pressure["UpDown"].eq("U"), value_col].mean()
        down = at_pressure.loc[at_pressure["UpDown"].eq("D"), value_col].mean()
        if pd.notna(up) and pd.notna(down):
            diffs.append(float(up - down))
    return (float(np.mean(diffs)), len(diffs)) if diffs else (np.nan, 0)


def video_weighted_difference(group: pd.DataFrame, value_col: str) -> float:
    up = group.loc[group["UpDown"].eq("U"), value_col].mean()
    down = group.loc[group["UpDown"].eq("D"), value_col].mean()
    return float(up - down) if pd.notna(up) and pd.notna(down) else np.nan


def trapezoid_auc_difference(group: pd.DataFrame, value_col: str) -> float:
    diffs: list[tuple[float, float]] = []
    for pressure in STANDARD_PRESSURES:
        at_pressure = group[group["Pressure"].eq(pressure)]
        up = at_pressure.loc[at_pressure["UpDown"].eq("U"), value_col].mean()
        down = at_pressure.loc[at_pressure["UpDown"].eq("D"), value_col].mean()
        if pd.notna(up) and pd.notna(down):
            diffs.append((pressure, float(up - down)))
    if len(diffs) < 2:
        return np.nan
    pressures = np.asarray([item[0] for item in diffs], dtype=float)
    values = np.asarray([item[1] for item in diffs], dtype=float)
    order = np.argsort(pressures)
    return float(np.trapz(values[order], pressures[order]))


def build_participant_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for participant, group in raw.groupby("Participant", sort=False):
        row = {
            "Participant": participant,
            "SET": str(first_non_null(group["SET"])),
            "Age": pd.to_numeric(first_non_null(group["Age"]), errors="coerce"),
            "Diabetes": bool_true(first_non_null(group["Diabetes"])),
            "Hypertension": bool_true(first_non_null(group["Hypertension"])),
        }

        for pressure in STANDARD_PRESSURES:
            for value_col, prefix in [
                ("Video_Median_Velocity", "velocity_at"),
                ("Log_Video_Median_Velocity", "log_velocity_at"),
            ]:
                row[f"{prefix}_{pressure_label(pressure)}"] = float(
                    group.loc[group["Pressure"].eq(pressure), value_col].mean()
                )

        for value_col, suffix in [
            ("Video_Median_Velocity", "raw"),
            ("Log_Video_Median_Velocity", "log"),
        ]:
            equal_weight, n_pressures = mean_by_pressure_difference(group, value_col)
            video_weighted = video_weighted_difference(group, value_col)
            trapz_auc = trapezoid_auc_difference(group, value_col)
            row[f"hysteresis_equal_weight_{suffix}"] = equal_weight
            row[f"abs_hysteresis_equal_weight_{suffix}"] = abs(equal_weight) if pd.notna(equal_weight) else np.nan
            row[f"n_equal_weight_pressures_{suffix}"] = n_pressures
            row[f"hysteresis_video_weighted_{suffix}"] = video_weighted
            row[f"abs_hysteresis_video_weighted_{suffix}"] = abs(video_weighted) if pd.notna(video_weighted) else np.nan
            row[f"hysteresis_trapz_auc_{suffix}"] = trapz_auc
            row[f"abs_hysteresis_trapz_auc_{suffix}"] = abs(trapz_auc) if pd.notna(trapz_auc) else np.nan

        rows.append(row)
    table = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    table["is_control"] = table["SET"].eq("set01")
    table["is_diabetes_cohort"] = table["SET"].eq("set03")
    return table


def median_iqr(values: Iterable[float]) -> str:
    series = pd.Series(values).dropna()
    return f"{series.median():.1f} ({series.quantile(0.25):.1f}--{series.quantile(0.75):.1f})"


def median_sd(values: Iterable[float]) -> str:
    series = pd.Series(values).dropna()
    return f"{series.median():.1f} ({series.std(ddof=1):.1f})"


def fmt_p(value: float) -> str:
    return f"{value:.2e}" if value < 0.001 else f"{value:.3f}"


def age_bin_rows(participants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    controls = participants[participants["SET"].eq("set01")].copy()
    controls["age_bin"] = np.select(
        [
            controls["Age"] < 30,
            (controls["Age"] >= 30) & (controls["Age"] < 60),
            controls["Age"] >= 60,
        ],
        ["<30", "30--59", r"$\geq$60"],
        default=np.nan,
    )
    pairwise_rows: list[dict] = []
    summary_rows: list[dict] = []
    group_labels = ["<30", "30--59", r"$\geq$60"]
    metrics = [
        ("Signed hysteresis", "hysteresis_equal_weight_raw"),
        ("Absolute hysteresis", "abs_hysteresis_equal_weight_raw"),
    ]
    for metric_label, metric_col in metrics:
        groups = {label: controls.loc[controls["age_bin"].eq(label), metric_col].dropna() for label in group_labels}
        omnibus_p = kruskal(*[groups[label] for label in group_labels]).pvalue
        summary_rows.append(
            {
                "metric": metric_label,
                "lt30": median_sd(groups["<30"]),
                "mid": median_sd(groups["30--59"]),
                "ge60": median_sd(groups[r"$\geq$60"]),
                "omnibus_p": omnibus_p,
            }
        )
        for group1, group2 in [("<30", "30--59"), ("<30", r"$\geq$60"), ("30--59", r"$\geq$60")]:
            p_value = mannwhitneyu(groups[group1], groups[group2], alternative="two-sided").pvalue
            row = {
                "metric": metric_label,
                "group1": group1,
                "n1": len(groups[group1]),
                "median_iqr1": median_iqr(groups[group1]),
                "group2": group2,
                "n2": len(groups[group2]),
                "median_iqr2": median_iqr(groups[group2]),
                "p_value": p_value,
            }
            pairwise_rows.append(row)

    pairwise = pd.DataFrame(pairwise_rows)
    summary = pd.DataFrame(summary_rows)
    return pairwise, summary, pairwise_latex(pairwise), summary_latex(summary)


def pairwise_latex(pairwise: pd.DataFrame) -> str:
    abs_rows = pairwise[pairwise["metric"].eq("Absolute hysteresis")]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Pairwise comparisons of absolute pressure-flow hysteresis among control age groups. Hysteresis is calculated by giving each pressure equal weight: for each pressure from 0.2--1.2 psi, the mean recovery velocity is subtracted from the mean compression velocity, and the resulting pressure-specific differences are averaged. Values are median (IQR); p-values are two-sided Mann-Whitney U tests.}",
        r"\label{tab:absolute_hysteresis_pairwise}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lcllclc@{}}",
        r"\toprule",
        r"Group 1 & n & Median (IQR) & Group 2 & n & Median (IQR) & p-value \\",
        r"\midrule",
    ]
    for _, row in abs_rows.iterrows():
        lines.append(
            f"{row['group1']} & {row['n1']} & {row['median_iqr1']} & "
            f"{row['group2']} & {row['n2']} & {row['median_iqr2']} & {fmt_p(row['p_value'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def summary_latex(summary: pd.DataFrame) -> str:
    p_lookup = {row["metric"]: fmt_p(row["omnibus_p"]) for _, row in summary.iterrows()}
    summary_lookup = {row["metric"]: row for _, row in summary.iterrows()}
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Control participant hysteresis by age group using equal pressure weighting. Values are median (SD). Omnibus p-values are Kruskal-Wallis tests.}",
        r"\label{tab:hysteresis_age_bins}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Metric & Statistic & $<30$ & 30--59 & $\geq$60 \\",
        r"\midrule",
    ]
    for metric in ["Signed hysteresis", "Absolute hysteresis"]:
        row = summary_lookup[metric]
        lines.append(f"{metric} & Median (SD) & {row['lt30']} & {row['mid']} & {row['ge60']} \\\\")
        lines.append(rf" & Omnibus p & \multicolumn{{3}}{{c}}{{{p_lookup[metric]}}} \\")
        if metric == "Signed hysteresis":
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


@dataclass
class ModelSpec:
    name: str
    estimator: object


def make_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("l2_logistic", LogisticRegression(solver="liblinear", C=1.0, random_state=42, max_iter=1000)),
        ModelSpec(
            "xgboost",
            xgb.XGBClassifier(
                n_estimators=80,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
            ),
        ),
    ]


def evaluate_classifier_sets(participants: pd.DataFrame) -> pd.DataFrame:
    cohort = participants[participants["SET"].isin(["set01", "set03"])].copy()
    y = cohort["SET"].eq("set03").to_numpy()
    feature_sets = {
        "selected_pressures_only": ["log_velocity_at_0.4psi", "log_velocity_at_1.2psi"],
        "selected_plus_equal_signed_raw": [
            "log_velocity_at_0.4psi",
            "log_velocity_at_1.2psi",
            "hysteresis_equal_weight_raw",
        ],
        "selected_plus_equal_abs_raw": [
            "log_velocity_at_0.4psi",
            "log_velocity_at_1.2psi",
            "abs_hysteresis_equal_weight_raw",
        ],
        "selected_plus_equal_signed_log": [
            "log_velocity_at_0.4psi",
            "log_velocity_at_1.2psi",
            "hysteresis_equal_weight_log",
        ],
        "selected_plus_equal_abs_log": [
            "log_velocity_at_0.4psi",
            "log_velocity_at_1.2psi",
            "abs_hysteresis_equal_weight_log",
        ],
        "selected_plus_video_weighted_signed_raw": [
            "log_velocity_at_0.4psi",
            "log_velocity_at_1.2psi",
            "hysteresis_video_weighted_raw",
        ],
        "selected_plus_auc_signed_raw": [
            "log_velocity_at_0.4psi",
            "log_velocity_at_1.2psi",
            "hysteresis_trapz_auc_raw",
        ],
        "selected_plus_auc_abs_raw": [
            "log_velocity_at_0.4psi",
            "log_velocity_at_1.2psi",
            "abs_hysteresis_trapz_auc_raw",
        ],
        "equal_signed_raw_only": ["hysteresis_equal_weight_raw"],
        "equal_abs_raw_only": ["abs_hysteresis_equal_weight_raw"],
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    rows: list[dict] = []
    for feature_set, features in feature_sets.items():
        X = cohort[features].to_numpy(dtype=float)
        for spec in make_model_specs():
            aucs: list[float] = []
            accuracies: list[float] = []
            balanced: list[float] = []
            f1s: list[float] = []
            for train_idx, test_idx in cv.split(X, y):
                pipe = Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("smote", SMOTE(random_state=42, k_neighbors=5)),
                        ("model", spec.estimator),
                    ]
                )
                pipe.fit(X[train_idx], y[train_idx])
                pred = pipe.predict(X[test_idx])
                if hasattr(pipe.named_steps["model"], "predict_proba"):
                    score = pipe.predict_proba(X[test_idx])[:, 1]
                else:
                    score = pred.astype(float)
                aucs.append(roc_auc_score(y[test_idx], score))
                accuracies.append(accuracy_score(y[test_idx], pred))
                balanced.append(balanced_accuracy_score(y[test_idx], pred))
                f1s.append(f1_score(y[test_idx], pred, zero_division=0))
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": spec.name,
                    "n_participants": len(cohort),
                    "n_diabetes": int(y.sum()),
                    "n_control": int((~y).sum()),
                    "features": ";".join(features),
                    "roc_auc_mean": float(np.mean(aucs)),
                    "roc_auc_sd": float(np.std(aucs, ddof=0)),
                    "accuracy_mean": float(np.mean(accuracies)),
                    "balanced_accuracy_mean": float(np.mean(balanced)),
                    "f1_mean": float(np.mean(f1s)),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "roc_auc_mean"], ascending=[True, False])


def write_methods_update() -> None:
    text = r"""Equal-pressure hysteresis methods update
=========================================

Use this wording to replace the current Hysteresis Methods paragraph. This keeps
the manuscript definition aligned with equal pressure weighting and avoids the
trapezoid/AUC endpoint weighting issue.

Pasteable Methods paragraph
---------------------------

\subsection{Hysteresis}\label{subhyst}
Hysteresis measurements were made by comparing the compression and recovery velocity curves over the 0.2 to 1.2 psi pressure range (Fig. \ref{fig:suppHist}). For each participant, velocities were first averaged separately for the compression and recovery ramps at each external pressure. At each pressure, we subtracted the average recovery velocity from the average compression velocity. We then averaged these pressure-specific differences across the measured pressures from 0.2 to 1.2 psi, giving each pressure equal weight. Positive values indicate higher velocities during compression than recovery, while negative values indicate higher velocities during recovery than compression. For comparisons of hysteresis magnitude, we used the absolute value of this signed hysteresis metric.

Suggested Results definition replacement
----------------------------------------

Replace the current sentence defining hysteresis as an area-under-curve subtraction with:

We define hysteresis as the mean pressure-specific difference between the compression and recovery curves over 0.2 to 1.2 psi, after first averaging velocities at each pressure within each ramp (Methods, Fig. \ref{fig:suppHist}). This definition gives each pressure equal weight and captures the asymmetry between blood cell velocities during compression and recovery.

Suggested figure caption replacement
------------------------------------

Replace:

f) Hysteresis is calculated by subtracting the area under the recovery curve from the area under the compression curve over 0.2-1.2 psi.

With:

f) Hysteresis is calculated by subtracting recovery velocity from compression velocity at each pressure from 0.2 to 1.2 psi and averaging those pressure-specific differences.
"""
    (DOCS_DIR / "equal_pressure_hysteresis_methods_update.txt").write_text(text, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    raw = read_raw()
    participants = build_participant_table(raw)
    participants.to_csv(RESULTS_DIR / "hysteresis_equal_pressure_participant_metrics.csv", index=False)

    pairwise, summary, pairwise_tex, summary_tex = age_bin_rows(participants)
    pairwise.to_csv(RESULTS_DIR / "hysteresis_equal_pressure_age_pairwise.csv", index=False)
    summary.to_csv(RESULTS_DIR / "hysteresis_equal_pressure_age_summary.csv", index=False)
    (DOCS_DIR / "absolute_hysteresis_pairwise_table.txt").write_text(pairwise_tex, encoding="utf-8")
    (DOCS_DIR / "hysteresis_age_bin_comparison_table.txt").write_text(summary_tex, encoding="utf-8")
    write_methods_update()

    classifier_summary = evaluate_classifier_sets(participants)
    classifier_summary.to_csv(CLASSIFIER_DIR / "hysteresis_metric_classifier_summary.csv", index=False)
    top_rows = classifier_summary.groupby("model", as_index=False).head(8)
    top_rows.to_csv(CLASSIFIER_DIR / "hysteresis_metric_classifier_top_rows.csv", index=False)

    report = [
        "Hysteresis definition comparison",
        "================================",
        "",
        "Participant metrics:",
        str(RESULTS_DIR / "hysteresis_equal_pressure_participant_metrics.csv"),
        "",
        "Control age-bin pairwise results:",
        pairwise.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "Classifier summary, top rows by model:",
        top_rows.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
    ]
    (CLASSIFIER_DIR / "hysteresis_metric_classifier_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
