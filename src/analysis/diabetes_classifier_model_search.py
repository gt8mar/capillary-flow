"""Search for high-performing diabetes classifiers from participant-level data.

This is an exploratory model search. It keeps imputation, scaling, optional
SMOTE, optional feature expansion, and model fitting inside each CV training
fold. It is meant to answer "what works best?" rather than to provide a final
locked validation estimate.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from src.config import PATHS
from src.analysis.diabetes_classifier_feature_exploration import (
    DEFAULT_OUTPUT_SUBDIR,
    PRESSURES,
    _pressure_feature_name,
    build_participant_table,
)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for prefix in ["log_velocity_at", "velocity_at"]:
        values = {p: _pressure_feature_name(prefix, p) for p in PRESSURES}
        for p1, p2 in [(0.2, 0.4), (0.4, 0.8), (0.4, 1.2), (0.8, 1.2), (0.2, 1.2)]:
            c1, c2 = values[p1], values[p2]
            if c1 in out.columns and c2 in out.columns:
                out[f"{prefix}_delta_{p1:.1f}_{p2:.1f}psi"] = out[c2] - out[c1]
        if all(values[p] in out.columns for p in PRESSURES):
            pressure_array = np.asarray(PRESSURES, dtype=float)
            y = out[[values[p] for p in PRESSURES]].to_numpy(dtype=float)
            out[f"{prefix}_mean_02_12"] = np.nanmean(y, axis=1)
            out[f"{prefix}_std_02_12"] = np.nanstd(y, axis=1)
            out[f"{prefix}_slope_02_12"] = [
                np.polyfit(pressure_array, row, deg=1)[0] if np.isfinite(row).all() else np.nan
                for row in y
            ]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    log_pressures = [_pressure_feature_name("log_velocity_at", p) for p in PRESSURES]
    raw_pressures = [_pressure_feature_name("velocity_at", p) for p in PRESSURES]
    selected = [
        _pressure_feature_name("log_velocity_at", 0.4),
        _pressure_feature_name("log_velocity_at", 0.8),
        _pressure_feature_name("log_velocity_at", 1.2),
    ]
    manuscript_raw = [
        "hysteresis_raw_mean_up_minus_down",
        _pressure_feature_name("log_velocity_at", 0.4),
        _pressure_feature_name("log_velocity_at", 1.2),
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
    hyst = [
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
    stiffness = [
        "Stiffness_Index",
        "Compliance",
        "Baseline_Velocity",
        "Pressure_Sensitivity",
        "Velocity_Std",
        "area_total",
    ]
    engineered = [
        c
        for c in df.columns
        if "_delta_" in c or c.endswith("_mean_02_12") or c.endswith("_std_02_12") or c.endswith("_slope_02_12")
    ]

    candidates = {
        "selected_pressures": selected,
        "manuscript_raw": manuscript_raw,
        "log_pressures": log_pressures,
        "log_pressures_hyst": log_pressures + hyst,
        "log_pressures_engineered": log_pressures + engineered,
        "log_pressures_hyst_engineered": log_pressures + hyst + engineered,
        "selected_plus_clinical": selected + clinical,
        "manuscript_raw_plus_clinical": manuscript_raw + clinical,
        "log_pressures_hyst_clinical": log_pressures + hyst + clinical,
        "clinical": clinical,
        "stiffness_clinical": stiffness + clinical,
        "all_no_raw_velocity": log_pressures + hyst + engineered + clinical + stiffness,
        "all_with_raw_velocity": log_pressures + raw_pressures + hyst + engineered + clinical + stiffness,
    }

    available = set(df.columns)
    cleaned: Dict[str, List[str]] = {}
    for name, cols in candidates.items():
        unique = []
        for col in cols:
            if col in available and col not in unique:
                unique.append(col)
        cleaned[name] = unique
    return cleaned


def model_grid() -> Dict[str, object]:
    return {
        "lasso_C0.3": LogisticRegression(penalty="l1", solver="liblinear", C=0.3, random_state=42, max_iter=1000),
        "lasso_C1": LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42, max_iter=1000),
        "l2_C0.3": LogisticRegression(penalty="l2", solver="liblinear", C=0.3, random_state=42, max_iter=1000),
        "l2_C1": LogisticRegression(penalty="l2", solver="liblinear", C=1.0, random_state=42, max_iter=1000),
        "svm_rbf_C0.3": SVC(C=0.3, kernel="rbf", gamma="scale", probability=True, random_state=42),
        "svm_rbf_C1": SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42),
        "knn3": KNeighborsClassifier(n_neighbors=3),
        "knn5": KNeighborsClassifier(n_neighbors=5),
        "gaussian_nb": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2, class_weight="balanced", random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2, class_weight="balanced", random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42),
        "xgboost_depth1": xgb.XGBClassifier(n_estimators=100, max_depth=1, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss"),
        "xgboost_depth2": xgb.XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss"),
    }


def make_pipeline(model: object, use_smote: bool, selector_k: int | None, polynomial: bool) -> Pipeline:
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if polynomial:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    if selector_k is not None:
        steps.append(("select", SelectKBest(score_func=f_classif, k=selector_k)))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=5)))
    steps.append(("model", clone(model)))
    return Pipeline(steps)


def evaluate_pipeline(X: np.ndarray, y: np.ndarray, pipe: Pipeline, cv: RepeatedStratifiedKFold) -> dict:
    metrics: Dict[str, List[float]] = defaultdict(list)
    for train_idx, test_idx in cv.split(X, y):
        pipe_fold = clone(pipe)
        pipe_fold.fit(X[train_idx], y[train_idx])
        pred = pipe_fold.predict(X[test_idx])
        if hasattr(pipe_fold.named_steps["model"], "predict_proba"):
            score = pipe_fold.predict_proba(X[test_idx])[:, 1]
        elif hasattr(pipe_fold.named_steps["model"], "decision_function"):
            score = pipe_fold.decision_function(X[test_idx])
        else:
            score = pred.astype(float)
        metrics["roc_auc"].append(roc_auc_score(y[test_idx], score))
        metrics["accuracy"].append(accuracy_score(y[test_idx], pred))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(y[test_idx], pred))
        metrics["precision_macro"].append(precision_score(y[test_idx], pred, average="macro", zero_division=0))
        metrics["f1_macro"].append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
    row = {}
    for name, vals in metrics.items():
        arr = np.asarray(vals)
        row[f"{name}_mean"] = float(arr.mean())
        row[f"{name}_sd"] = float(arr.std(ddof=0))
    return row


def run(args: argparse.Namespace) -> None:
    cap_flow_path = args.cap_flow_path or PATHS["cap_flow"]
    output_dir = args.output_dir or os.path.join(cap_flow_path, DEFAULT_OUTPUT_SUBDIR, "model_search")
    os.makedirs(output_dir, exist_ok=True)

    df = add_engineered_features(build_participant_table(cap_flow_path))
    df.to_csv(os.path.join(output_dir, "participant_feature_table.csv"), index=False)

    cohorts = {
        "diabetes_vs_controls": df[(df["Diabetes"]) | (df["is_healthy"])].copy(),
        "diabetes_vs_all_nondiabetes": df.copy(),
    }
    if args.cohorts:
        cohorts = {name: cohorts[name] for name in args.cohorts}
    families = feature_sets(df)
    if args.feature_sets:
        families = {name: families[name] for name in args.feature_sets}

    models = model_grid()
    if args.models:
        models = {name: models[name] for name in args.models}

    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=42)
    rows = []
    for cohort_name, cohort in cohorts.items():
        for family_name, features in families.items():
            X = cohort[features].to_numpy(dtype=float)
            y = cohort["Diabetes"].astype(bool).to_numpy()
            selector_options: List[int | None] = [None]
            if args.select_k and len(features) > args.select_k:
                selector_options.append(args.select_k)
            for model_name, model in models.items():
                for use_smote in ([True, False] if args.compare_no_smote else [True]):
                    for selector_k in selector_options:
                        poly_options = [False]
                        if args.polynomial and model_name.startswith(("lasso", "l2")) and len(features) <= args.max_poly_features:
                            poly_options.append(True)
                        for polynomial in poly_options:
                            pipe = make_pipeline(model, use_smote=use_smote, selector_k=selector_k, polynomial=polynomial)
                            result = evaluate_pipeline(X, y, pipe, cv=cv)
                            rows.append(
                                {
                                    "cohort": cohort_name,
                                    "feature_set": family_name,
                                    "model": model_name,
                                    "smote": use_smote,
                                    "selector_k": selector_k,
                                    "polynomial": polynomial,
                                    "n_participants": int(len(cohort)),
                                    "n_diabetes": int(y.sum()),
                                    "n_negative": int((~y).sum()),
                                    "n_features": int(len(features)),
                                    "features": ";".join(features),
                                    **result,
                                }
                            )

    summary = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    summary_path = os.path.join(output_dir, "model_search_summary.csv")
    best_path = os.path.join(output_dir, "model_search_best.csv")
    report_path = os.path.join(output_dir, "model_search_report.txt")
    summary.to_csv(summary_path, index=False)
    summary.groupby("cohort", as_index=False).head(args.top_n).to_csv(best_path, index=False)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("Diabetes classifier model search\n")
        handle.write("=" * 40 + "\n")
        handle.write(f"CV: {args.folds} folds x {args.repeats} repeats\n\n")
        handle.write(summary.head(args.report_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved summary: {summary_path}")
    print(f"Saved best rows: {best_path}")
    print(f"Saved report: {report_path}")
    print(summary.head(args.report_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-flow-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--feature-sets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--cohorts", nargs="+", default=None, choices=["diabetes_vs_controls", "diabetes_vs_all_nondiabetes"])
    parser.add_argument("--select-k", type=int, default=8)
    parser.add_argument("--polynomial", action="store_true")
    parser.add_argument("--max-poly-features", type=int, default=8)
    parser.add_argument("--compare-no-smote", action="store_true")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--report-rows", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
