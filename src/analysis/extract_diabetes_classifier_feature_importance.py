"""Extract feature-selection summaries for diabetes classifiers.

Exploratory helper for the up-down hysteresis classifier rerun. This does not
change classifier definitions; it summarizes already used feature families with
fold-level model coefficients/importances.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import PATHS
from src.analysis.diabetes_classifier_model_search import (
    add_engineered_features,
    feature_sets,
    make_pipeline,
    model_grid,
)
from src.analysis.diabetes_classifier_feature_exploration import (
    DEFAULT_OUTPUT_SUBDIR,
    build_participant_table,
)


def selected_feature_names(pipe, input_features: list[str]) -> list[str]:
    names = list(input_features)
    if "poly" in pipe.named_steps:
        names = list(pipe.named_steps["poly"].get_feature_names_out(names))
    if "select" in pipe.named_steps:
        support = pipe.named_steps["select"].get_support()
        names = [name for name, keep in zip(names, support) if keep]
    return names


def summarize_model(
    cohort: pd.DataFrame,
    features: list[str],
    model_name: str,
    selector_k: int | None,
    repeats: int,
    folds: int,
) -> pd.DataFrame:
    X = cohort[features].to_numpy(dtype=float)
    y = cohort["Diabetes"].astype(bool).to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
    model = model_grid()[model_name]
    pipe = make_pipeline(model, use_smote=True, selector_k=selector_k, polynomial=False)

    values: dict[str, list[float]] = defaultdict(list)
    selected: dict[str, int] = defaultdict(int)
    n_folds = 0
    for train_idx, _test_idx in cv.split(X, y):
        fold_pipe = clone(pipe)
        fold_pipe.fit(X[train_idx], y[train_idx])
        names = selected_feature_names(fold_pipe, features)
        estimator = fold_pipe.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            scores = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            scores = np.ravel(estimator.coef_)
        else:
            continue
        for name, score in zip(names, scores):
            selected[name] += 1
            values[name].append(float(score))
        n_folds += 1

    rows = []
    for name in sorted(set(values) | set(selected)):
        arr = np.asarray(values[name], dtype=float)
        rows.append(
            {
                "model": model_name,
                "selector_k": selector_k,
                "feature": name,
                "selected_fraction": selected[name] / n_folds if n_folds else np.nan,
                "mean_score": float(np.nanmean(arr)) if arr.size else 0.0,
                "mean_abs_score": float(np.nanmean(np.abs(arr))) if arr.size else 0.0,
                "n_folds": n_folds,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selected_fraction", "mean_abs_score"], ascending=[False, False]
    )


def run(args: argparse.Namespace) -> None:
    cap_flow_path = args.cap_flow_path or PATHS["cap_flow"]
    output_dir = args.output_dir or os.path.join(
        cap_flow_path,
        DEFAULT_OUTPUT_SUBDIR,
        "feature_importance_updown_hysteresis",
    )
    os.makedirs(output_dir, exist_ok=True)

    df = add_engineered_features(build_participant_table(cap_flow_path))
    cohort = df[(df["Diabetes"]) | (df["is_healthy"])].copy()
    families = feature_sets(df)

    rows = []
    for feature_set in args.feature_sets:
        features = families[feature_set]
        selector_options = [None]
        if args.select_k and len(features) > args.select_k:
            selector_options.append(args.select_k)
        for model_name in args.models:
            for selector_k in selector_options:
                summary = summarize_model(
                    cohort,
                    features,
                    model_name=model_name,
                    selector_k=selector_k,
                    repeats=args.repeats,
                    folds=args.folds,
                )
                summary.insert(0, "feature_set", feature_set)
                summary.insert(0, "cohort", "diabetes_vs_controls")
                rows.append(summary)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(output_dir, "feature_importance_summary.csv"), index=False)

    top = (
        out.sort_values(
            ["feature_set", "model", "selector_k", "selected_fraction", "mean_abs_score"],
            ascending=[True, True, True, False, False],
        )
        .groupby(["feature_set", "model", "selector_k"], dropna=False)
        .head(args.report_rows)
    )
    top.to_csv(os.path.join(output_dir, "feature_importance_top.csv"), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-flow-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--select-k", type=int, default=8)
    parser.add_argument("--report-rows", type=int, default=15)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["manuscript_raw_plus_clinical", "manuscript_raw", "all_no_raw_velocity"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["random_forest", "extra_trees", "xgboost_depth1", "lasso_C0.3", "lasso_C1"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
