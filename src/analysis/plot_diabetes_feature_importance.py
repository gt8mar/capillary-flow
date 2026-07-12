"""Plot diabetes classifier feature importance summaries.

This is an exploratory plotting helper. It reads saved CV summaries and creates
standalone Lasso and random-forest feature plots for device-derived
velocity+hysteresis features.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PATHS


DEFAULT_LASSO = Path(
    "results/Classifier/Velocity/Diabetes/"
    "feature_exploration_updown_hysteresis_lasso/"
    "classifier_feature_exploration_coefficients.csv"
)
DEFAULT_RF = Path(
    "results/Classifier/Velocity/Diabetes/"
    "feature_importance_updown_hysteresis_velocity_hyst/"
    "feature_importance_summary.csv"
)
DEFAULT_OUT = Path(
    "results/Classifier/Velocity/Diabetes/"
    "feature_importance_plots_no_earlydrop"
)


LABELS = {
    "log_velocity_at_0.2psi": "Log velocity at 0.2 psi",
    "log_velocity_at_0.4psi": "Log velocity at 0.4 psi",
    "log_velocity_at_0.6psi": "Log velocity at 0.6 psi",
    "log_velocity_at_0.8psi": "Log velocity at 0.8 psi",
    "log_velocity_at_1.0psi": "Log velocity at 1.0 psi",
    "log_velocity_at_1.2psi": "Log velocity at 1.2 psi",
    "hysteresis_log_pole_mean_up_minus_down": "Log hysteresis, pole mean",
    "hysteresis_log_0.2_up_minus_down": "Log hysteresis at 0.2 psi",
    "hysteresis_log_1.2_up_minus_down": "Log hysteresis at 1.2 psi",
    "hysteresis_log_mean_up_minus_down": "Log hysteresis, mean",
    "abs_hysteresis_log": "Absolute log hysteresis",
    "positive_hysteresis_log": "Positive log hysteresis",
    "hysteresis_raw_mean_up_minus_down": "Raw hysteresis, mean",
    "hysteresis_raw_pole_mean_up_minus_down": "Raw hysteresis, pole mean",
    "hysteresis_raw_0.2_up_minus_down": "Raw hysteresis at 0.2 psi",
    "hysteresis_raw_1.2_up_minus_down": "Raw hysteresis at 1.2 psi",
    "abs_hysteresis_raw": "Absolute raw hysteresis",
    "positive_hysteresis_raw": "Positive raw hysteresis",
}


def pretty_label(feature: str) -> str:
    return LABELS.get(feature, feature.replace("_", " "))


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6b7280")
    ax.spines["bottom"].set_color("#6b7280")
    ax.tick_params(axis="both", colors="#111827", labelsize=9)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def plot_lasso(lasso_csv: Path, out_dir: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(lasso_csv)
    subset = df[
        (df["cohort"] == "diabetes_vs_controls")
        & (df["feature_set"] == "log_pressures_plus_hyst")
        & (df["model"] == "lasso_logistic_C0.3")
        & (df["nonzero_fraction"] > 0)
    ].copy()
    subset = subset.sort_values("mean_abs_coef", ascending=False).head(top_n)
    subset = subset.sort_values("nonzero_fraction", ascending=True)
    subset["label"] = subset["feature"].map(pretty_label)
    subset["direction"] = np.where(subset["mean_coef"] >= 0, "Higher in diabetes direction", "Lower in diabetes direction")

    colors = np.where(subset["mean_coef"] >= 0, "#2563eb", "#dc2626")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(subset))
    ax.barh(y, subset["nonzero_fraction"] * 100, color=colors, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(subset["label"])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Selected in cross-validation folds (%)", fontsize=10)
    ax.set_title("Lasso-selected velocity and hysteresis features", fontsize=12, weight="bold", loc="left")
    for i, (_, row) in enumerate(subset.iterrows()):
        ax.text(
            row["nonzero_fraction"] * 100 + 1.5,
            i,
            f"{row['mean_coef']:+.2f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#111827",
        )
    style_axes(ax)
    save_figure(fig, out_dir, "lasso_velocity_hysteresis_selection")
    plt.close(fig)
    return subset


def plot_rf(rf_csv: Path, out_dir: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(rf_csv)
    subset = df[
        (df["cohort"] == "diabetes_vs_controls")
        & (df["feature_set"] == "log_pressures_hyst")
        & (df["model"] == "random_forest")
        & (df["selector_k"].isna())
    ].copy()
    subset = subset.sort_values("mean_abs_score", ascending=False).head(top_n)
    subset = subset.sort_values("mean_abs_score", ascending=True)
    subset["label"] = subset["feature"].map(pretty_label)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(subset))
    ax.barh(y, subset["mean_abs_score"], color="#059669", height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(subset["label"])
    ax.set_xlabel("Mean random-forest feature importance", fontsize=10)
    ax.set_title("Random-forest velocity and hysteresis feature importance", fontsize=12, weight="bold", loc="left")
    for i, (_, row) in enumerate(subset.iterrows()):
        ax.text(
            row["mean_abs_score"] + 0.002,
            i,
            f"{row['mean_abs_score']:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#111827",
        )
    ax.set_xlim(0, subset["mean_abs_score"].max() * 1.22)
    style_axes(ax)
    save_figure(fig, out_dir, "random_forest_velocity_hysteresis_importance")
    plt.close(fig)
    return subset


def run(args: argparse.Namespace) -> None:
    root = Path(args.cap_flow_path or PATHS["cap_flow"])
    lasso_csv = root / args.lasso_csv
    rf_csv = root / args.rf_csv
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lasso = plot_lasso(lasso_csv, out_dir, args.top_n)
    rf = plot_rf(rf_csv, out_dir, args.top_n)
    lasso.to_csv(out_dir / "lasso_velocity_hysteresis_plotted_features.csv", index=False)
    rf.to_csv(out_dir / "random_forest_velocity_hysteresis_plotted_features.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-flow-path", default=None)
    parser.add_argument("--lasso-csv", default=str(DEFAULT_LASSO))
    parser.add_argument("--rf-csv", default=str(DEFAULT_RF))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
