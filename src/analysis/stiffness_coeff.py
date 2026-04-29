"""
Filename: src/analysis/stiffness_coeff.py

Calculate and plot the Compression Resistance Index (CRI).

CRI is the chosen paper stiffness metric:
    CRI = AUC of the averaged up/down log-velocity curve from 0.2 to 1.2 psi

The legacy multi-metric implementation is archived at
src/analysis/archive/stiffness_coeff_legacy.py.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

from src.config import PATHS


CRI_COLUMN = "CRI"
CRI_LEGACY_COLUMN = "SI_logvel_averaged_02_12"
LOG_VELOCITY_COLUMN = "Log_Video_Median_Velocity"
PRESSURE_MIN = 0.2
PRESSURE_MAX = 1.2


def get_cap_flow_path() -> str:
    """Return configured capillary-flow path, falling back to the repo root."""
    configured = PATHS.get("cap_flow")
    if configured and os.path.exists(configured):
        return configured
    return str(Path(__file__).resolve().parents[2])


cap_flow_path = get_cap_flow_path()


def get_up_down_curves(
    participant_df: pd.DataFrame,
    velocity_column: str = LOG_VELOCITY_COLUMN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract sorted up/down pressure and velocity curves for one participant."""
    up_grouped = (
        participant_df[participant_df["UpDown"] == "U"]
        .groupby("Pressure")[velocity_column]
        .mean()
        .sort_index()
    )
    down_grouped = (
        participant_df[participant_df["UpDown"] == "D"]
        .groupby("Pressure")[velocity_column]
        .mean()
        .sort_index()
    )

    return (
        up_grouped.index.to_numpy(dtype=float),
        up_grouped.to_numpy(dtype=float),
        down_grouped.index.to_numpy(dtype=float),
        down_grouped.to_numpy(dtype=float),
    )


def calculate_cri_from_curves(
    up_pressures: np.ndarray,
    up_velocities: np.ndarray,
    down_pressures: np.ndarray,
    down_velocities: np.ndarray,
    pressure_min: float = PRESSURE_MIN,
    pressure_max: float = PRESSURE_MAX,
) -> float:
    """Calculate CRI as AUC of averaged up/down log velocity over 0.2-1.2 psi."""
    if len(up_pressures) == 0 and len(down_pressures) == 0:
        return np.nan

    all_pressures = np.unique(np.concatenate([up_pressures, down_pressures]))
    pressure_mask = (all_pressures >= pressure_min) & (all_pressures <= pressure_max)
    filtered_pressures = all_pressures[pressure_mask]
    if len(filtered_pressures) < 2:
        return np.nan

    curves = []
    if len(up_pressures) >= 2:
        curves.append(
            np.interp(filtered_pressures, up_pressures, up_velocities, left=np.nan, right=np.nan)
        )
    if len(down_pressures) >= 2:
        curves.append(
            np.interp(filtered_pressures, down_pressures, down_velocities, left=np.nan, right=np.nan)
        )
    if not curves:
        return np.nan

    averaged_log_velocity = np.nanmean(curves, axis=0)
    valid = ~np.isnan(averaged_log_velocity)
    if np.sum(valid) < 2:
        return np.nan

    return float(np.trapz(averaged_log_velocity[valid], filtered_pressures[valid]))


def calculate_cri_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate one participant-level CRI row per participant."""
    required = ["Participant", "Pressure", "UpDown", LOG_VELOCITY_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required CRI input columns: {missing}")

    working = df.copy()
    working["Pressure"] = pd.to_numeric(working["Pressure"], errors="coerce")
    working[LOG_VELOCITY_COLUMN] = pd.to_numeric(
        working[LOG_VELOCITY_COLUMN], errors="coerce"
    )
    working = working.dropna(subset=required)

    participant_rows = []
    for participant, participant_df in working.groupby("Participant", sort=False):
        up_p, up_v, down_p, down_v = get_up_down_curves(participant_df)
        cri = calculate_cri_from_curves(up_p, up_v, down_p, down_v)

        row = {
            "Participant": participant,
            CRI_COLUMN: cri,
            CRI_LEGACY_COLUMN: cri,
        }

        for col in ["Age", "Diabetes", "Hypertension", "SET", "Sex", "SYS_BP", "DIA_BP"]:
            if col in participant_df.columns:
                row[col] = participant_df[col].iloc[0]

        if pd.notna(row.get("SYS_BP")) and pd.notna(row.get("DIA_BP")):
            sys_bp = pd.to_numeric(row["SYS_BP"], errors="coerce")
            dia_bp = pd.to_numeric(row["DIA_BP"], errors="coerce")
            row["MAP"] = dia_bp + (sys_bp - dia_bp) / 3
        else:
            row["MAP"] = np.nan

        row["is_healthy"] = str(row.get("SET", "")).startswith("set01")
        for bool_col in ["Diabetes", "Hypertension"]:
            if bool_col in row and isinstance(row[bool_col], str):
                row[bool_col] = row[bool_col].upper() == "TRUE"

        participant_rows.append(row)

    results_df = pd.DataFrame(participant_rows)
    print("\nCRI SUMMARY")
    print("=" * 40)
    print(f"Participants: {len(results_df)}")
    print(f"Valid CRI values: {results_df[CRI_COLUMN].notna().sum()}")
    print(f"Mean CRI: {results_df[CRI_COLUMN].mean():.3f}")
    print(f"Median CRI: {results_df[CRI_COLUMN].median():.3f}")
    print(f"Range: {results_df[CRI_COLUMN].min():.3f} to {results_df[CRI_COLUMN].max():.3f}")
    return results_df


def age_adjusted_analysis(
    results_df: pd.DataFrame,
    stiffness_col: str,
    group_col: str = "Diabetes",
) -> Dict:
    """Run OLS for CRI-like outcome adjusted by age.

    Kept as a small public helper because the paper plotting functions call it.
    """
    df_clean = results_df[[stiffness_col, group_col, "Age"]].dropna().copy()
    if len(df_clean) < 10:
        return {"error": "Insufficient data for analysis"}

    df_clean["Group"] = df_clean[group_col].astype(bool).astype(int)
    model = ols(f"{stiffness_col} ~ Group + Age", data=df_clean).fit()

    return {
        "formula": f"{stiffness_col} ~ Group + Age",
        "n": int(model.nobs),
        "group_coef": float(model.params["Group"]),
        "group_pvalue": float(model.pvalues["Group"]),
        "age_coef": float(model.params["Age"]),
        "age_pvalue": float(model.pvalues["Age"]),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue) if model.fvalue is not None else np.nan,
        "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        "model_summary": str(model.summary()),
    }


def save_cri_outputs(results_df: pd.DataFrame, output_dir: str) -> None:
    """Write CRI outputs using both explicit and compatibility filenames."""
    os.makedirs(output_dir, exist_ok=True)
    cri_path = os.path.join(output_dir, "cri_metrics.csv")
    compatibility_path = os.path.join(output_dir, "stiffness_coefficients_log.csv")

    results_df.to_csv(cri_path, index=False)
    results_df.to_csv(compatibility_path, index=False)
    print(f"Saved CRI metrics to: {cri_path}")
    print(f"Saved compatibility CRI file to: {compatibility_path}")


def generate_cri_paper_plots(results_df: pd.DataFrame, output_dir: str) -> None:
    """Generate the paper plots that use the selected CRI metric."""
    from src.analysis.plot_stiffness import (
        _ensure_healthy_flag,
        plot_age_adjusted_analysis,
        plot_bp_by_age_group_control,
        plot_bp_by_disease_group,
        plot_bp_si_by_condition,
        plot_bp_si_correlation_all,
        plot_log_stiffness_by_group,
        plot_si_by_age_brackets_control,
        plot_si_by_age_group_control,
        plot_si_by_disease_group,
        plot_si_correlation_control,
    )

    os.makedirs(output_dir, exist_ok=True)
    minimal_dir = os.path.join(output_dir, "no_annotations")
    os.makedirs(minimal_dir, exist_ok=True)

    plot_df = _ensure_healthy_flag(results_df)
    significance_rows = []

    plot_log_stiffness_by_group(
        plot_df,
        output_dir,
        stiffness_col=CRI_LEGACY_COLUMN,
        minimal_dir=minimal_dir,
    )
    plot_age_adjusted_analysis(
        plot_df,
        output_dir,
        stiffness_col=CRI_LEGACY_COLUMN,
        minimal_dir=minimal_dir,
    )
    significance_rows.extend(
        plot_bp_si_by_condition(
            plot_df,
            plot_df,
            output_dir,
            stiffness_col=CRI_LEGACY_COLUMN,
            bp_col="SYS_BP",
            minimal_dir=minimal_dir,
        )
        or []
    )
    significance_rows.extend(
        plot_si_by_age_group_control(
            plot_df,
            output_dir,
            stiffness_col=CRI_LEGACY_COLUMN,
            minimal_dir=minimal_dir,
        )
        or []
    )
    significance_rows.extend(
        plot_si_by_age_brackets_control(
            plot_df,
            output_dir,
            stiffness_col=CRI_LEGACY_COLUMN,
            minimal_dir=minimal_dir,
        )
        or []
    )
    significance_rows.extend(
        plot_bp_by_age_group_control(plot_df, output_dir, minimal_dir=minimal_dir) or []
    )
    significance_rows.extend(
        plot_bp_by_disease_group(plot_df, output_dir, minimal_dir=minimal_dir) or []
    )
    significance_rows.extend(
        plot_si_by_disease_group(
            plot_df,
            output_dir,
            stiffness_col=CRI_LEGACY_COLUMN,
            minimal_dir=minimal_dir,
        )
        or []
    )
    significance_rows.extend(
        plot_si_correlation_control(
            plot_df,
            output_dir,
            stiffness_col=CRI_LEGACY_COLUMN,
            minimal_dir=minimal_dir,
        )
        or []
    )
    significance_rows.extend(
        plot_bp_si_correlation_all(
            plot_df,
            output_dir,
            stiffness_col=CRI_LEGACY_COLUMN,
            minimal_dir=minimal_dir,
        )
        or []
    )

    if significance_rows:
        sig_path = os.path.join(output_dir, "cri_paper_plot_significance.csv")
        pd.DataFrame(significance_rows).to_csv(sig_path, index=False)
        print(f"Saved CRI paper plot significance to: {sig_path}")


def main() -> int:
    """Calculate CRI and generate CRI paper plots."""
    print("\nStarting CRI calculation and paper plot generation...")
    data_path = os.path.join(cap_flow_path, "summary_df_nhp_video_stats.csv")
    if not os.path.exists(data_path):
        print(f"Error: data file not found at {data_path}")
        return 1

    df = pd.read_csv(data_path)
    if LOG_VELOCITY_COLUMN not in df.columns:
        if "Video_Median_Velocity" not in df.columns:
            raise ValueError("Missing both log and raw video median velocity columns.")
        df[LOG_VELOCITY_COLUMN] = np.log(pd.to_numeric(df["Video_Median_Velocity"], errors="coerce"))

    results_df = calculate_cri_metrics(df)

    stiffness_dir = os.path.join(cap_flow_path, "results", "Stiffness")
    save_cri_outputs(results_df, stiffness_dir)

    plot_dir = os.path.join(stiffness_dir, "plots")
    generate_cri_paper_plots(results_df, plot_dir)

    print("\nCRI analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
