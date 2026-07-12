"""
Age-matched CRI and systolic blood-pressure comparisons.

This analysis compares healthy controls against hypertensive and diabetic
participants after filtering controls to the same age floor as each disease
cohort. For diabetes, the young participants are omitted and the older diabetic
cohort starts at the first participant in the older cluster.
"""

import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import PATHS


CRI_COL = "SI_logvel_averaged_02_12"
CONTROL_SET = "set01"
OUTPUT_SUBDIR = "AgeMatched_CRI_BP_Analysis"


def get_cap_flow_path() -> str:
    """Return configured capillary-flow path, falling back to repo root."""
    configured = PATHS.get("cap_flow")
    if configured and os.path.exists(configured):
        return configured
    return str(Path(__file__).resolve().parents[2])


def load_data(cap_flow_path: str) -> pd.DataFrame:
    """Load participant-level stiffness/CRI data."""
    data_path = os.path.join(
        cap_flow_path, "results", "Stiffness", "stiffness_coefficients_log.csv"
    )
    df = pd.read_csv(data_path)
    required = [
        "Participant",
        "Age",
        "SET",
        "Diabetes",
        "Hypertension",
        "SYS_BP",
        CRI_COL,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    for col in ["Age", "SYS_BP", CRI_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Diabetes", "Hypertension"]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.upper().eq("TRUE")

    df = df.dropna(subset=["Age", "SYS_BP", CRI_COL]).copy()
    df["is_healthy_control"] = df["SET"].eq(CONTROL_SET)
    return df.reset_index(drop=True)


def infer_diabetes_age_floor(df: pd.DataFrame) -> float:
    """Choose the older diabetic cohort floor after omitting young participants."""
    diabetic_ages = sorted(df.loc[df["Diabetes"], "Age"].dropna().unique())
    older_ages = [age for age in diabetic_ages if age >= 50]
    if not older_ages:
        raise ValueError("Could not infer older diabetes age floor; no diabetic ages >= 50.")
    return float(min(older_ages))


def cohort_specs(df: pd.DataFrame) -> List[Dict]:
    """Return age-matched comparison specifications."""
    diabetes_floor = infer_diabetes_age_floor(df)
    hypertension_floor = float(df.loc[df["SET"].eq("set02"), "Age"].min())
    return [
        {
            "condition": "Diabetes",
            "condition_col": "Diabetes",
            "condition_label": "Older diabetic",
            "set": "set03",
            "age_floor": diabetes_floor,
            "rationale": "Diabetic participants below 50 were omitted; older cohort starts at the lowest remaining diabetic age.",
        },
        {
            "condition": "Hypertension",
            "condition_col": "Hypertension",
            "condition_label": "Hypertensive",
            "set": "set02",
            "age_floor": hypertension_floor,
            "rationale": "Age floor is the lowest-age hypertension-cohort participant (SET == 'set02').",
        },
    ]


def summarize_group(values: pd.Series) -> Dict:
    """Return n, median, quartiles, and IQR."""
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return {
        "n": int(values.count()),
        "median": float(values.median()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def run_comparison(df: pd.DataFrame, spec: Dict, outcome: str) -> Dict:
    """Compare age-filtered healthy controls to one disease cohort."""
    age_floor = spec["age_floor"]
    condition_col = spec["condition_col"]

    controls = df[df["is_healthy_control"] & (df["Age"] >= age_floor)].copy()
    disease = df[
        df[condition_col] & df["SET"].eq(spec["set"]) & (df["Age"] >= age_floor)
    ].copy()

    control_values = controls[outcome].dropna()
    disease_values = disease[outcome].dropna()
    stat, p_value = stats.mannwhitneyu(
        control_values, disease_values, alternative="two-sided"
    )

    rows = []
    for group_label, group_df, values in [
        ("Age-matched healthy controls", controls, control_values),
        (spec["condition_label"], disease, disease_values),
    ]:
        summary = summarize_group(values)
        rows.append(
            {
                "condition": spec["condition"],
                "outcome": outcome,
                "group": group_label,
                "age_floor": age_floor,
                "age_min": float(group_df["Age"].min()),
                "age_max": float(group_df["Age"].max()),
                "participant_ids": ";".join(group_df["Participant"].astype(str)),
                "test": "Mann-Whitney U",
                "statistic": float(stat),
                "p_value": float(p_value),
                **summary,
            }
        )
    return {"summary_rows": rows, "controls": controls, "disease": disease}


def plot_comparison(plot_df: pd.DataFrame, outcome: str, title: str, output_path: str) -> None:
    """Save a boxplot with overlaid participant points."""
    label = "Systolic BP (mmHg)" if outcome == "SYS_BP" else "CRI"
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    sns.boxplot(data=plot_df, x="group", y=outcome, ax=ax, width=0.55, fliersize=3)
    sns.stripplot(data=plot_df, x="group", y=outcome, ax=ax, color="black", size=3, alpha=0.55)
    ax.set_xlabel("")
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> int:
    """Run the age-matched comparison analysis."""
    cap_flow_path = get_cap_flow_path()
    output_dir = os.path.join(cap_flow_path, "results", OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")
    df = load_data(cap_flow_path)

    all_summary_rows = []
    dataset_rows = []
    cutoff_rows = []

    for spec in cohort_specs(df):
        cutoff_rows.append(
            {
                "condition": spec["condition"],
                "age_floor": spec["age_floor"],
                "rationale": spec["rationale"],
            }
        )
        for outcome in ["SYS_BP", CRI_COL]:
            result = run_comparison(df, spec, outcome)
            all_summary_rows.extend(result["summary_rows"])

            controls = result["controls"].copy()
            controls["group"] = "Age-matched healthy controls"
            disease = result["disease"].copy()
            disease["group"] = spec["condition_label"]
            plot_df = pd.concat([controls, disease], ignore_index=True)
            plot_df["comparison_condition"] = spec["condition"]
            dataset_rows.append(plot_df)

            outcome_slug = "sbp" if outcome == "SYS_BP" else "cri"
            plot_comparison(
                plot_df,
                outcome,
                f"{outcome_slug.upper()} by {spec['condition']} (age >= {spec['age_floor']:.0f})",
                os.path.join(output_dir, f"{spec['condition'].lower()}_{outcome_slug}.png"),
            )

    summary_df = pd.DataFrame(all_summary_rows)
    dataset_df = pd.concat(dataset_rows, ignore_index=True)
    cutoff_df = pd.DataFrame(cutoff_rows)

    summary_path = os.path.join(output_dir, "age_matched_summary.csv")
    dataset_path = os.path.join(output_dir, "age_matched_analysis_dataset.csv")
    cutoff_path = os.path.join(output_dir, "age_matching_cutoffs.csv")
    report_path = os.path.join(output_dir, "age_matched_summary.txt")

    summary_df.to_csv(summary_path, index=False)
    dataset_df.to_csv(dataset_path, index=False)
    cutoff_df.to_csv(cutoff_path, index=False)

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("Age-Matched CRI and Systolic BP Analysis\n")
        handle.write("=" * 43 + "\n\n")
        handle.write("Controls are healthy controls only: SET == 'set01'.\n")
        handle.write("Disease groups use SET == 'set02' for hypertension and SET == 'set03' for diabetes.\n")
        handle.write(f"CRI column: {CRI_COL}\n\n")
        handle.write("Age floors:\n")
        handle.write(cutoff_df.to_string(index=False))
        handle.write("\n\nSummary:\n")
        display_cols = [
            "condition",
            "outcome",
            "group",
            "n",
            "age_min",
            "age_max",
            "median",
            "q1",
            "q3",
            "iqr",
            "p_value",
        ]
        handle.write(summary_df[display_cols].to_string(index=False))
        handle.write("\n")

    print(f"Saved age-matched results to: {output_dir}")
    print(summary_df[["condition", "outcome", "group", "n", "median", "iqr", "p_value"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
