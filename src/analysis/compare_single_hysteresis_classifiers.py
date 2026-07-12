"""Compare velocity classifiers with one hysteresis metric at a time.

Exploratory analysis for diabetes vs healthy controls. This keeps the cohort,
SMOTE, imputation, scaling, and CV rules aligned with the classifier reruns, but
restricts each feature set to log pressure velocities plus one hysteresis metric.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import PATHS
from src.analysis.diabetes_classifier_feature_exploration import (
    DEFAULT_OUTPUT_SUBDIR,
    PRESSURES,
    _pressure_feature_name,
    build_participant_table,
)
from src.analysis.diabetes_classifier_model_search import (
    evaluate_pipeline,
    make_pipeline,
    model_grid,
)


HYSTERESIS_METRICS = [
    "hysteresis_raw_mean_up_minus_down",
    "hysteresis_log_mean_up_minus_down",
    "hysteresis_raw_pole_mean_up_minus_down",
    "hysteresis_log_pole_mean_up_minus_down",
    "hysteresis_raw_0.2_up_minus_down",
    "hysteresis_log_0.2_up_minus_down",
    "hysteresis_raw_1.2_up_minus_down",
    "hysteresis_log_1.2_up_minus_down",
    "abs_hysteresis_raw",
    "abs_hysteresis_log",
    "positive_hysteresis_raw",
    "positive_hysteresis_log",
]


def summarize_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    model_name: str,
    repeats: int,
    folds: int,
) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
    model = model_grid()[model_name]
    pipe = make_pipeline(model, use_smote=True, selector_k=None, polynomial=False)
    values: dict[str, list[float]] = defaultdict(list)
    nonzero: dict[str, int] = defaultdict(int)
    n_folds = 0
    for train_idx, _test_idx in cv.split(X, y):
        fold_pipe = clone(pipe)
        fold_pipe.fit(X[train_idx], y[train_idx])
        estimator = fold_pipe.named_steps["model"]
        if hasattr(estimator, "coef_"):
            scores = np.ravel(estimator.coef_)
        elif hasattr(estimator, "feature_importances_"):
            scores = estimator.feature_importances_
        else:
            continue
        for feature, score in zip(features, scores):
            values[feature].append(float(score))
            if abs(float(score)) > 1e-12:
                nonzero[feature] += 1
        n_folds += 1

    rows = []
    for feature in features:
        arr = np.asarray(values[feature], dtype=float)
        rows.append(
            {
                "feature": feature,
                "nonzero_fraction": nonzero[feature] / n_folds if n_folds else np.nan,
                "mean_score": float(np.nanmean(arr)) if arr.size else np.nan,
                "mean_abs_score": float(np.nanmean(np.abs(arr))) if arr.size else np.nan,
                "n_folds": n_folds,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    cap_flow_path = args.cap_flow_path or PATHS["cap_flow"]
    output_dir = Path(args.output_dir or os.path.join(cap_flow_path, DEFAULT_OUTPUT_SUBDIR, "single_hysteresis"))
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_participant_table(cap_flow_path)
    cohort = df[(df["Diabetes"]) | (df["is_healthy"])].copy()
    velocity_features = [_pressure_feature_name("log_velocity_at", p) for p in PRESSURES]
    requested_metrics = args.hysteresis_metrics or HYSTERESIS_METRICS
    metrics = [m for m in requested_metrics if m in cohort.columns]
    models = {name: model_grid()[name] for name in args.models}
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=42)

    rows = []
    score_rows = []
    y = cohort["Diabetes"].astype(bool).to_numpy()
    for hyst_metric in metrics:
        features = velocity_features + [hyst_metric]
        X = cohort[features].to_numpy(dtype=float)
        for model_name, model in models.items():
            pipe = make_pipeline(model, use_smote=True, selector_k=None, polynomial=False)
            result = evaluate_pipeline(X, y, pipe=pipe, cv=cv)
            rows.append(
                {
                    "cohort": "diabetes_vs_controls",
                    "feature_set": f"log_velocities_plus_{hyst_metric}",
                    "hysteresis_metric": hyst_metric,
                    "model": model_name,
                    "smote": True,
                    "n_participants": int(len(cohort)),
                    "n_diabetes": int(y.sum()),
                    "n_control": int((~y).sum()),
                    "n_features": len(features),
                    "features": ";".join(features),
                    **result,
                }
            )
            if not args.skip_feature_scores:
                scores = summarize_coefficients(X, y, features, model_name, args.repeats, args.folds)
                scores.insert(0, "model", model_name)
                scores.insert(0, "hysteresis_metric", hyst_metric)
                score_rows.append(scores)

    summary = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    summary.to_csv(output_dir / "single_hysteresis_model_summary.csv", index=False)
    summary.groupby("hysteresis_metric", as_index=False).first().sort_values(
        "roc_auc_mean", ascending=False
    ).to_csv(output_dir / "single_hysteresis_best_by_metric.csv", index=False)
    if score_rows:
        pd.concat(score_rows, ignore_index=True).to_csv(
            output_dir / "single_hysteresis_feature_scores.csv", index=False
        )

    report = [
        "Single hysteresis metric classifier comparison",
        "",
        "Definition: diabetes vs healthy controls; part30 included as diabetes; all videos included; "
        "features are log velocities at 0.2-1.2 psi plus one hysteresis metric; SMOTE inside CV folds.",
        "",
        "Top rows by ROC-AUC:",
        summary.head(args.report_rows).to_string(index=False),
    ]
    (output_dir / "single_hysteresis_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(summary.head(args.report_rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-flow-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--report-rows", type=int, default=20)
    parser.add_argument("--skip-feature-scores", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lasso_C0.3", "lasso_C1", "random_forest", "extra_trees"],
    )
    parser.add_argument("--hysteresis-metrics", nargs="+", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
