"""Sparse exploratory check of diabetes duration against pressure-flow metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import PATHS
from src.analysis.pressure_flow_response_analysis import (
    load_analysis_data,
    participant_response_metrics,
)


DEFAULT_METADATA = Path("results/diabetes_medication_metadata_analysis/diabetes_participant_summary.csv")
DEFAULT_OUTPUT = Path("results/diabetes_duration_pressure_metric_check")


METRICS = [
    "log_velocity_0.4",
    "early_drop_log_0.2_minus_0.4",
    "hysteresis_log_pole_mean_up_minus_down",
    "hysteresis_log_0.2_up_minus_down",
    "cri_log_auc_02_12",
]


def run(args: argparse.Namespace) -> None:
    root = Path(args.cap_flow_path or PATHS["cap_flow"])
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(root / args.metadata_csv)
    df = load_analysis_data(str(root))
    metrics = participant_response_metrics(df)
    diabetes_metrics = metrics[metrics["Group"] == "Diabetes"].copy()
    merged = diabetes_metrics.merge(
        metadata[
            [
                "Participant",
                "diabetes_diagnosis_year",
                "diabetes_diagnosis_age",
                "years_since_diabetes_diagnosis",
                "diabetes_type_extra",
                "has_diabetes_med_note",
                "has_antihypertensive_med_note",
            ]
        ],
        on="Participant",
        how="left",
    )
    merged.to_csv(output_dir / "diabetes_duration_pressure_metrics.csv", index=False)

    rows = []
    for metric in METRICS:
        sub = merged.dropna(subset=["years_since_diabetes_diagnosis", metric])
        if len(sub) >= 3:
            rho, p_value = stats.spearmanr(sub["years_since_diabetes_diagnosis"], sub[metric])
        else:
            rho, p_value = np.nan, np.nan
        rows.append(
            {
                "metric": metric,
                "n_with_duration": int(len(sub)),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "duration_min": float(sub["years_since_diabetes_diagnosis"].min()) if len(sub) else np.nan,
                "duration_max": float(sub["years_since_diabetes_diagnosis"].max()) if len(sub) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "duration_metric_correlations.csv", index=False)

    lines = [
        "Diabetes Duration Pressure-Metric Check",
        "=======================================",
        "",
        "Exploratory only. Diagnosis duration is available for a small subset of diabetes participants.",
        f"Diabetes participants: {len(merged)}",
        f"With calculable years since diagnosis: {int(merged['years_since_diabetes_diagnosis'].notna().sum())}",
        "",
        summary.to_string(index=False),
    ]
    (output_dir / "duration_pressure_metric_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote sparse diabetes-duration checks to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-flow-path", default=None)
    parser.add_argument("--metadata-csv", default=str(DEFAULT_METADATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
