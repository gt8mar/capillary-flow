"""Explore diabetes classifiers with expanded participant-level features.

This script is intentionally separate from ``health_classifier.py``. It builds a
participant-level table, evaluates prespecified feature families, and keeps
imputation, scaling, SMOTE, and model fitting inside each cross-validation fold.

Outputs are written to ``results/Classifier/Velocity/Diabetes/feature_exploration``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import PATHS


PRESSURES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
DEFAULT_OUTPUT_SUBDIR = os.path.join(
    "results", "Classifier", "Velocity", "Diabetes", "feature_exploration"
)


def _bool_true(value: object) -> bool:
    return str(value).strip().upper() == "TRUE"


def _first_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def _pressure_feature_name(prefix: str, pressure: float) -> str:
    return f"{prefix}_{pressure:.1f}psi"


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Pandas mangles duplicate headers as Set, Set.1. Keep that behavior but
    # guard against exact duplicates from manually edited files.
    return df.loc[:, ~df.columns.duplicated()].copy()


def build_participant_table(cap_flow_path: str) -> pd.DataFrame:
    """Build one row per participant with velocity, metadata, and stiffness features."""
    raw_path = os.path.join(cap_flow_path, "summary_df_nhp_video_stats.csv")
    raw = _read_csv(raw_path)
    raw["Log_Video_Median_Velocity"] = raw["Log_Video_Median_Velocity"].clip(lower=0)

    rows: List[dict] = []
    for participant, group in raw.groupby("Participant", sort=False):
        diabetes = _bool_true(_first_non_null(group["Diabetes"]))
        hypertension = _bool_true(_first_non_null(group["Hypertension"]))
        set_name = str(_first_non_null(group["SET"]))
        sex = str(_first_non_null(group["Sex"])).upper()

        row = {
            "Participant": participant,
            "Diabetes": diabetes,
            "Hypertension": hypertension,
            "is_healthy": set_name.startswith("set01") and not diabetes and not hypertension,
            "SET": set_name,
            "Age": pd.to_numeric(_first_non_null(group["Age"]), errors="coerce"),
            "Sex_M": 1.0 if sex == "M" else 0.0 if sex == "F" else np.nan,
            "Height": pd.to_numeric(_first_non_null(group.get("Height", pd.Series(dtype=float))), errors="coerce"),
            "Weight": pd.to_numeric(_first_non_null(group.get("Weight", pd.Series(dtype=float))), errors="coerce"),
            "SYS_BP": pd.to_numeric(_first_non_null(group.get("SYS_BP", pd.Series(dtype=float))), errors="coerce"),
            "DIA_BP": pd.to_numeric(_first_non_null(group.get("DIA_BP", pd.Series(dtype=float))), errors="coerce"),
            "Pulse": pd.to_numeric(_first_non_null(group.get("Pulse", pd.Series(dtype=float))), errors="coerce"),
        }

        row["MAP"] = (row["SYS_BP"] + 2 * row["DIA_BP"]) / 3 if pd.notna(row["SYS_BP"]) and pd.notna(row["DIA_BP"]) else np.nan
        row["Pulse_Pressure"] = row["SYS_BP"] - row["DIA_BP"] if pd.notna(row["SYS_BP"]) and pd.notna(row["DIA_BP"]) else np.nan
        row["BMI"] = (
            703.0 * row["Weight"] / (row["Height"] ** 2)
            if pd.notna(row["Height"]) and pd.notna(row["Weight"]) and row["Height"] > 0
            else np.nan
        )

        for value_col, prefix in [
            ("Log_Video_Median_Velocity", "log_velocity_at"),
            ("Video_Median_Velocity", "velocity_at"),
        ]:
            pressure_stats = (
                group.pivot_table(index="Participant", columns="Pressure", values=value_col, aggfunc="mean")
                .fillna(0)
            )
            for pressure in PRESSURES:
                row[_pressure_feature_name(prefix, pressure)] = (
                    float(pressure_stats[pressure].iloc[0]) if pressure in pressure_stats.columns else 0.0
                )

        up_raw = group.loc[group["UpDown"].eq("U"), "Video_Median_Velocity"]
        down_raw = group.loc[group["UpDown"].eq("D"), "Video_Median_Velocity"]
        up_log = group.loc[group["UpDown"].eq("U"), "Log_Video_Median_Velocity"]
        down_log = group.loc[group["UpDown"].eq("D"), "Log_Video_Median_Velocity"]

        row["hysteresis_raw_mean_up_minus_down"] = float(up_raw.mean() - down_raw.mean()) if len(up_raw) and len(down_raw) else 0.0
        row["hysteresis_log_mean_up_minus_down"] = float(up_log.mean() - down_log.mean()) if len(up_log) and len(down_log) else 0.0
        raw_by_pressure = group.pivot_table(
            index="Pressure", columns="UpDown", values="Video_Median_Velocity", aggfunc="mean"
        )
        log_by_pressure = group.pivot_table(
            index="Pressure", columns="UpDown", values="Log_Video_Median_Velocity", aggfunc="mean"
        )
        for pressure in PRESSURES:
            if {"U", "D"}.issubset(raw_by_pressure.columns) and pressure in raw_by_pressure.index:
                row[f"hysteresis_raw_{pressure:.1f}_up_minus_down"] = float(
                    raw_by_pressure.loc[pressure, "U"] - raw_by_pressure.loc[pressure, "D"]
                )
            else:
                row[f"hysteresis_raw_{pressure:.1f}_up_minus_down"] = np.nan
            if {"U", "D"}.issubset(log_by_pressure.columns) and pressure in log_by_pressure.index:
                row[f"hysteresis_log_{pressure:.1f}_up_minus_down"] = float(
                    log_by_pressure.loc[pressure, "U"] - log_by_pressure.loc[pressure, "D"]
                )
            else:
                row[f"hysteresis_log_{pressure:.1f}_up_minus_down"] = np.nan
        raw_poles = [row.get("hysteresis_raw_0.2_up_minus_down"), row.get("hysteresis_raw_1.2_up_minus_down")]
        log_poles = [row.get("hysteresis_log_0.2_up_minus_down"), row.get("hysteresis_log_1.2_up_minus_down")]
        row["hysteresis_raw_pole_mean_up_minus_down"] = float(np.nanmean(raw_poles))
        row["hysteresis_log_pole_mean_up_minus_down"] = float(np.nanmean(log_poles))
        row["abs_hysteresis_raw"] = abs(row["hysteresis_raw_mean_up_minus_down"])
        row["abs_hysteresis_log"] = abs(row["hysteresis_log_mean_up_minus_down"])
        row["positive_hysteresis_raw"] = max(row["hysteresis_raw_mean_up_minus_down"], 0.0)
        row["positive_hysteresis_log"] = max(row["hysteresis_log_mean_up_minus_down"], 0.0)

        rows.append(row)

    participant_df = pd.DataFrame(rows)

    stiffness_path = os.path.join(cap_flow_path, "summary_df_with_stiffness.csv")
    if os.path.exists(stiffness_path):
        stiffness = _read_csv(stiffness_path)
        stiffness_cols = [
            "Stiffness_Index",
            "Compliance",
            "Baseline_Velocity",
            "Pressure_Sensitivity",
            "Model_R_Squared",
            "N_Observations",
            "Velocity_Std",
            "Pressure_Range",
            "Total_Area",
        ]
        available = ["Participant"] + [c for c in stiffness_cols if c in stiffness.columns]
        stiffness_participant = (
            stiffness[available]
            .groupby("Participant", sort=False)
            .first()
            .reset_index()
            .rename(columns={"Total_Area": "area_total"})
        )
        participant_df = participant_df.merge(stiffness_participant, on="Participant", how="left")

    participant_df = participant_df.replace([np.inf, -np.inf], np.nan)
    return participant_df


def feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
    log_pressures = [_pressure_feature_name("log_velocity_at", p) for p in PRESSURES]
    selected_pressures = [
        _pressure_feature_name("log_velocity_at", 0.4),
        _pressure_feature_name("log_velocity_at", 0.8),
        _pressure_feature_name("log_velocity_at", 1.2),
    ]
    manuscript_raw = [
        "hysteresis_raw_mean_up_minus_down",
        _pressure_feature_name("log_velocity_at", 0.4),
        _pressure_feature_name("log_velocity_at", 1.2),
    ]
    manuscript_log = [
        "hysteresis_log_mean_up_minus_down",
        _pressure_feature_name("log_velocity_at", 0.4),
        _pressure_feature_name("log_velocity_at", 1.2),
    ]
    hysteresis = [
        "hysteresis_raw_mean_up_minus_down",
        "hysteresis_log_mean_up_minus_down",
        "hysteresis_raw_pole_mean_up_minus_down",
        "hysteresis_log_pole_mean_up_minus_down",
        "hysteresis_raw_0.2_up_minus_down",
        "hysteresis_raw_1.2_up_minus_down",
        "hysteresis_log_0.2_up_minus_down",
        "hysteresis_log_1.2_up_minus_down",
        "abs_hysteresis_raw",
        "abs_hysteresis_log",
        "positive_hysteresis_raw",
        "positive_hysteresis_log",
    ]
    clinical = [
        "Age",
        "Sex_M",
        "SYS_BP",
        "DIA_BP",
        "MAP",
        "Pulse_Pressure",
        "Pulse",
        "Height",
        "Weight",
        "BMI",
    ]
    stiffness = [
        "Stiffness_Index",
        "Compliance",
        "Baseline_Velocity",
        "Pressure_Sensitivity",
        "Velocity_Std",
        "area_total",
    ]

    families = {
        "clinical_only": clinical,
        "selected_pressures": selected_pressures,
        "manuscript_raw_hyst": manuscript_raw,
        "manuscript_log_hyst": manuscript_log,
        "log_pressures_all": log_pressures,
        "log_pressures_plus_hyst": log_pressures + hysteresis,
        "stiffness_only": stiffness,
        "stiffness_plus_clinical": stiffness + clinical,
        "selected_pressures_plus_clinical": selected_pressures + clinical,
        "manuscript_raw_plus_clinical": manuscript_raw + clinical,
        "log_pressures_hyst_clinical": log_pressures + hysteresis + clinical,
        "all_available_low_dim": log_pressures + hysteresis + clinical + stiffness,
    }

    available = set(df.columns)
    cleaned = {}
    for name, cols in families.items():
        unique_cols = []
        for col in cols:
            if col in available and col not in unique_cols:
                unique_cols.append(col)
        if unique_cols:
            cleaned[name] = unique_cols
    return cleaned


@dataclass
class ModelSpec:
    name: str
    estimator: object
    sparse_coefficients: bool = False


def model_specs(models: Sequence[str]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    if "lasso" in models:
        for c in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
            specs.append(
                ModelSpec(
                    name=f"lasso_logistic_C{c:g}",
                    estimator=LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        C=c,
                        random_state=42,
                        max_iter=1000,
                    ),
                    sparse_coefficients=True,
                )
            )
    if "logistic" in models:
        for c in [0.1, 0.3, 1.0, 3.0]:
            specs.append(
                ModelSpec(
                    name=f"l2_logistic_C{c:g}",
                    estimator=LogisticRegression(
                        penalty="l2",
                        solver="liblinear",
                        C=c,
                        random_state=42,
                        max_iter=1000,
                    ),
                    sparse_coefficients=True,
                )
            )
    if "rf" in models:
        specs.append(
            ModelSpec(
                name="random_forest",
                estimator=RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
            )
        )
    if "xgb" in models:
        specs.append(
            ModelSpec(
                name="xgboost",
                estimator=xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=2,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss",
                ),
            )
        )
    return specs


def make_pipeline(estimator: object, use_smote: bool) -> Pipeline:
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=5)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def evaluate_one(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    model_spec: ModelSpec,
    cv: RepeatedStratifiedKFold,
    use_smote: bool,
) -> Tuple[dict, List[dict]]:
    metrics: Dict[str, List[float]] = defaultdict(list)
    coefficients: List[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        pipe = make_pipeline(model_spec.estimator, use_smote=use_smote)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[test_idx])
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_score = pipe.predict_proba(X[test_idx])[:, 1]
        else:
            y_score = y_pred.astype(float)

        metrics["roc_auc"].append(roc_auc_score(y[test_idx], y_score))
        metrics["accuracy"].append(accuracy_score(y[test_idx], y_pred))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(y[test_idx], y_pred))
        metrics["precision_macro"].append(precision_score(y[test_idx], y_pred, average="macro", zero_division=0))
        metrics["f1_macro"].append(f1_score(y[test_idx], y_pred, average="macro", zero_division=0))

        model = pipe.named_steps["model"]
        if hasattr(model, "coef_"):
            for feature, coef in zip(feature_names, model.coef_[0]):
                coefficients.append(
                    {
                        "fold": fold_idx,
                        "feature": feature,
                        "coef": float(coef),
                        "nonzero": bool(abs(coef) > 1e-9),
                    }
                )
        elif hasattr(model, "feature_importances_"):
            for feature, importance in zip(feature_names, model.feature_importances_):
                coefficients.append(
                    {
                        "fold": fold_idx,
                        "feature": feature,
                        "coef": float(importance),
                        "nonzero": bool(importance > 0),
                    }
                )

    summary = {}
    for metric, values in metrics.items():
        arr = np.asarray(values, dtype=float)
        summary[f"{metric}_mean"] = float(arr.mean())
        summary[f"{metric}_sd"] = float(arr.std(ddof=0))
    return summary, coefficients


def summarize_coefficients(coef_rows: List[dict], group_cols: Sequence[str]) -> pd.DataFrame:
    if not coef_rows:
        return pd.DataFrame()
    coef_df = pd.DataFrame(coef_rows)
    grouped = coef_df.groupby(list(group_cols) + ["feature"], dropna=False)
    return (
        grouped.agg(
            nonzero_fraction=("nonzero", "mean"),
            mean_coef=("coef", "mean"),
            mean_abs_coef=("coef", lambda s: float(np.mean(np.abs(s)))),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
        .sort_values(list(group_cols) + ["nonzero_fraction", "mean_abs_coef"], ascending=[True] * len(group_cols) + [False, False])
    )


def run(args: argparse.Namespace) -> None:
    cap_flow_path = args.cap_flow_path or PATHS["cap_flow"]
    output_dir = args.output_dir or os.path.join(cap_flow_path, DEFAULT_OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    participant_df = build_participant_table(cap_flow_path)
    participant_path = os.path.join(output_dir, "participant_feature_table.csv")
    participant_df.to_csv(participant_path, index=False)

    cohorts = {
        "diabetes_vs_controls": participant_df[(participant_df["Diabetes"]) | (participant_df["is_healthy"])].copy(),
        "diabetes_vs_all_nondiabetes": participant_df.copy(),
    }
    families = feature_families(participant_df)

    selected_family_names = args.feature_sets or list(families.keys())
    selected_models = model_specs(args.models)
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=42)

    summary_rows: List[dict] = []
    all_coef_rows: List[dict] = []

    for cohort_name, cohort in cohorts.items():
        for family_name in selected_family_names:
            if family_name not in families:
                raise ValueError(f"Unknown feature set: {family_name}")
            feature_names = families[family_name]
            sub = cohort.dropna(subset=["Diabetes"]).copy()
            X = sub[feature_names].to_numpy(dtype=float)
            y = sub["Diabetes"].astype(bool).to_numpy()
            if len(np.unique(y)) < 2:
                continue
            for spec in selected_models:
                summary, coef_rows = evaluate_one(
                    X,
                    y,
                    feature_names=feature_names,
                    model_spec=spec,
                    cv=cv,
                    use_smote=not args.no_smote,
                )
                row = {
                    "cohort": cohort_name,
                    "feature_set": family_name,
                    "model": spec.name,
                    "smote": not args.no_smote,
                    "n_participants": int(len(sub)),
                    "n_diabetes": int(y.sum()),
                    "n_negative": int((~y).sum()),
                    "n_features": int(len(feature_names)),
                    "features": ";".join(feature_names),
                }
                row.update(summary)
                summary_rows.append(row)
                for coef_row in coef_rows:
                    coef_row.update(
                        {
                            "cohort": cohort_name,
                            "feature_set": family_name,
                            "model": spec.name,
                            "smote": not args.no_smote,
                        }
                    )
                    all_coef_rows.append(coef_row)

    summary_df = pd.DataFrame(summary_rows).sort_values("roc_auc_mean", ascending=False)
    coef_summary = summarize_coefficients(all_coef_rows, ["cohort", "feature_set", "model", "smote"])

    summary_path = os.path.join(output_dir, "classifier_feature_exploration_summary.csv")
    best_path = os.path.join(output_dir, "classifier_feature_exploration_best.csv")
    coef_path = os.path.join(output_dir, "classifier_feature_exploration_coefficients.csv")
    report_path = os.path.join(output_dir, "classifier_feature_exploration_report.txt")
    metadata_path = os.path.join(output_dir, "classifier_feature_exploration_metadata.json")

    summary_df.to_csv(summary_path, index=False)
    summary_df.groupby(["cohort"], as_index=False).head(args.top_n).to_csv(best_path, index=False)
    coef_summary.to_csv(coef_path, index=False)

    metadata = {
        "participant_feature_table": participant_path,
        "folds": args.folds,
        "repeats": args.repeats,
        "smote": not args.no_smote,
        "models": args.models,
        "feature_sets": selected_family_names,
        "cohort_counts": {
            name: {
                "n": int(len(df)),
                "n_diabetes": int(df["Diabetes"].sum()),
                "n_negative": int((~df["Diabetes"]).sum()),
            }
            for name, df in cohorts.items()
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("Diabetes classifier feature exploration\n")
        handle.write("=" * 48 + "\n")
        handle.write(f"Participant feature table: {participant_path}\n")
        handle.write(f"CV: {args.folds} folds x {args.repeats} repeats\n")
        handle.write(f"SMOTE inside each training fold: {not args.no_smote}\n\n")
        handle.write("Top models by ROC-AUC\n")
        handle.write(summary_df.head(args.report_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        handle.write("\n\nCoefficient/importances for top rows\n")
        for _, row in summary_df.head(args.top_n).iterrows():
            subset = coef_summary[
                (coef_summary["cohort"] == row["cohort"])
                & (coef_summary["feature_set"] == row["feature_set"])
                & (coef_summary["model"] == row["model"])
                & (coef_summary["smote"] == row["smote"])
            ].head(20)
            handle.write(f"\n{row['cohort']} | {row['feature_set']} | {row['model']}\n")
            handle.write(subset.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            handle.write("\n")

    print(f"Saved participant table: {participant_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved best rows: {best_path}")
    print(f"Saved coefficient summary: {coef_path}")
    print(f"Saved report: {report_path}")
    print("\nTop models by ROC-AUC:")
    print(summary_df.head(args.report_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-flow-path", default=None, help="Repository/data root. Defaults to src.config PATHS['cap_flow'].")
    parser.add_argument("--output-dir", default=None, help="Directory for output CSV/report files.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=["lasso", "logistic"], choices=["lasso", "logistic", "rf", "xgb"])
    parser.add_argument("--feature-sets", nargs="+", default=None, help="Feature family names to evaluate.")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE inside training folds.")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--report-rows", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
