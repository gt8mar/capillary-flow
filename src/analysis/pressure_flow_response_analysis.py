"""Pressure-flow response analysis for diabetes versus healthy controls.

This script keeps part30 as diabetes and uses all standard-range videos
(0.2-1.2 psi). It produces:

1. Adjusted pressure-flow curves from a clustered repeated-measures model.
2. Pressure-specific diabetes-control effect estimates.
3. Participant-level response metrics and adjusted group comparisons.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.config import (
    MANUSCRIPT_BOX_STYLE,
    MANUSCRIPT_COLORS,
    PATHS,
    apply_manuscript_plot_style,
    load_source_sans,
    save_manuscript_figure,
)


STANDARD_PRESSURES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
OUTPUT_SUBDIR = os.path.join("results", "PressureFlowResponse")
CONTROL_COLOR = MANUSCRIPT_COLORS["control"]
DIABETES_COLOR = MANUSCRIPT_COLORS["diabetes"]

apply_manuscript_plot_style()
source_sans = load_source_sans()


def apply_font(ax) -> None:
    """Apply Source Sans to axis text when the local font is available."""
    if source_sans:
        for label in ax.get_xticklabels():
            label.set_fontproperties(source_sans)
        for label in ax.get_yticklabels():
            label.set_fontproperties(source_sans)
        ax.xaxis.label.set_fontproperties(source_sans)
        ax.yaxis.label.set_fontproperties(source_sans)
        if ax.get_title():
            ax.title.set_fontproperties(source_sans)


def _bool_true(value: object) -> bool:
    return str(value).strip().upper() == "TRUE"


def _first_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def load_analysis_data(cap_flow_path: str) -> pd.DataFrame:
    path = os.path.join(cap_flow_path, "summary_df_nhp_video_stats.csv")
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["Diabetes_bool"] = df["Diabetes"].map(_bool_true)
    df["Hypertension_bool"] = df["Hypertension"].map(_bool_true)
    df["is_healthy"] = (
        df["SET"].astype(str).str.startswith("set01")
        & ~df["Diabetes_bool"]
        & ~df["Hypertension_bool"]
    )
    df["Log_Video_Median_Velocity"] = pd.to_numeric(
        df["Log_Video_Median_Velocity"], errors="coerce"
    ).clip(lower=0)
    df["Video_Median_Velocity"] = pd.to_numeric(
        df["Video_Median_Velocity"], errors="coerce"
    )
    df["Pressure"] = pd.to_numeric(df["Pressure"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["SYS_BP"] = pd.to_numeric(df["SYS_BP"], errors="coerce")
    df["DIA_BP"] = pd.to_numeric(df["DIA_BP"], errors="coerce")
    df["Sex_M"] = df["Sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0})
    df["UpDown"] = df["UpDown"].fillna("Unknown").astype(str)

    # Primary paper comparison: diabetes versus healthy controls.
    df = df[df["Diabetes_bool"] | df["is_healthy"]].copy()
    df = df[df["Pressure"].isin(STANDARD_PRESSURES)].copy()
    df["Group"] = np.where(df["Diabetes_bool"], "Diabetes", "Control")
    df["Diabetes_int"] = df["Diabetes_bool"].astype(int)
    df["PressureCat"] = df["Pressure"].map(lambda p: f"{p:.1f}")
    return df


def participant_metadata(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant, group in df.groupby("Participant", sort=False):
        rows.append(
            {
                "Participant": participant,
                "Group": "Diabetes" if bool(group["Diabetes_bool"].iloc[0]) else "Control",
                "Diabetes_int": int(group["Diabetes_bool"].iloc[0]),
                "Age": pd.to_numeric(_first_non_null(group["Age"]), errors="coerce"),
                "Sex_M": pd.to_numeric(_first_non_null(group["Sex_M"]), errors="coerce"),
                "SYS_BP": pd.to_numeric(_first_non_null(group["SYS_BP"]), errors="coerce"),
                "DIA_BP": pd.to_numeric(_first_non_null(group["DIA_BP"]), errors="coerce"),
                "n_rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    model_df = df.dropna(
        subset=[
            "Log_Video_Median_Velocity",
            "PressureCat",
            "Diabetes_int",
            "Age",
            "Sex_M",
            "SYS_BP",
            "DIA_BP",
            "UpDown",
            "Participant",
        ]
    ).copy()

    centers = {
        "Age": float(model_df["Age"].mean()),
        "SYS_BP": float(model_df["SYS_BP"].mean()),
        "DIA_BP": float(model_df["DIA_BP"].mean()),
        "Sex_M": float(model_df["Sex_M"].mean()),
    }
    for col in ["Age", "SYS_BP", "DIA_BP"]:
        model_df[f"{col}_c"] = model_df[col] - centers[col]
    model_df["PressureCat"] = pd.Categorical(
        model_df["PressureCat"], categories=[f"{p:.1f}" for p in STANDARD_PRESSURES], ordered=True
    )
    return model_df, centers


def fit_pressure_response_model(model_df: pd.DataFrame):
    formula = (
        'Log_Video_Median_Velocity ~ C(PressureCat, Treatment(reference="0.2")) '
        "* Diabetes_int + Age_c + Sex_M + SYS_BP_c + DIA_BP_c + C(UpDown)"
    )
    model = smf.gee(
        formula=formula,
        groups="Participant",
        data=model_df,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    return model.fit(), formula


def design_matrix(result, new_df: pd.DataFrame) -> pd.DataFrame:
    design_info = result.model.data.design_info
    matrices = patsy.build_design_matrices([design_info], new_df, return_type="dataframe")
    return matrices[0]


def linear_estimate(result, design_row: np.ndarray) -> tuple[float, float, float, float, float]:
    params = result.params.to_numpy()
    cov = result.cov_params().to_numpy()
    est = float(design_row @ params)
    se = float(np.sqrt(max(design_row @ cov @ design_row.T, 0.0)))
    z = est / se if se > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return est, se, est - 1.96 * se, est + 1.96 * se, p


def adjusted_curve_table(result, centers: dict) -> pd.DataFrame:
    rows = []
    for group_name, diabetes_int in [("Control", 0), ("Diabetes", 1)]:
        for pressure in STANDARD_PRESSURES:
            for updown in ["U", "D"]:
                rows.append(
                    {
                        "Group": group_name,
                        "Diabetes_int": diabetes_int,
                        "Pressure": pressure,
                        "PressureCat": f"{pressure:.1f}",
                        "Age_c": 0.0,
                        "Sex_M": centers["Sex_M"],
                        "SYS_BP_c": 0.0,
                        "DIA_BP_c": 0.0,
                        "UpDown": updown,
                    }
                )
    expanded_df = pd.DataFrame(rows)
    expanded_df["PressureCat"] = pd.Categorical(
        expanded_df["PressureCat"], categories=[f"{p:.1f}" for p in STANDARD_PRESSURES], ordered=True
    )
    dm = design_matrix(result, expanded_df)

    pred_rows = []
    for (group, pressure), idx in expanded_df.groupby(["Group", "Pressure"], sort=False).groups.items():
        avg_design = dm.loc[list(idx)].mean(axis=0).to_numpy()
        est, se, low, high, p = linear_estimate(result, avg_design)
        pred_rows.append(
            {
                "Group": group,
                "Diabetes_int": int(expanded_df.loc[list(idx), "Diabetes_int"].iloc[0]),
                "Pressure": pressure,
                "adjusted_log_velocity": est,
                "se": se,
                "ci_low": low,
                "ci_high": high,
                "p_value": p,
            }
        )
    return pd.DataFrame(pred_rows)


def pressure_specific_effects(result, centers: dict) -> pd.DataFrame:
    rows = []
    for pressure in STANDARD_PRESSURES:
        base = {
            "Pressure": pressure,
            "PressureCat": f"{pressure:.1f}",
            "Age_c": 0.0,
            "Sex_M": centers["Sex_M"],
            "SYS_BP_c": 0.0,
            "DIA_BP_c": 0.0,
            "UpDown": "U",
        }
        control = {**base, "Group": "Control", "Diabetes_int": 0}
        diabetes = {**base, "Group": "Diabetes", "Diabetes_int": 1}
        pair = pd.DataFrame([control, diabetes])
        pair["PressureCat"] = pd.Categorical(
            pair["PressureCat"], categories=[f"{p:.1f}" for p in STANDARD_PRESSURES], ordered=True
        )
        dm = design_matrix(result, pair)
        contrast = dm.iloc[1].to_numpy() - dm.iloc[0].to_numpy()
        est, se, low, high, p = linear_estimate(result, contrast)
        rows.append(
            {
                "Pressure": pressure,
                "diabetes_minus_control_log_velocity": est,
                "se": se,
                "ci_low": low,
                "ci_high": high,
                "p_value": p,
            }
        )
    out = pd.DataFrame(rows)
    out["p_fdr_bh"] = multipletest_bh(out["p_value"].to_numpy())
    return out


def multipletest_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p)
    adjusted = np.full_like(p, np.nan)
    if not valid.any():
        return adjusted
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[valid])]
    ranked = p[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    adjusted[order] = np.clip(q, 0, 1)
    return adjusted


def _area(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return np.nan
    order = np.argsort(x_arr[mask])
    return float(np.trapz(y_arr[mask][order], x_arr[mask][order]))


def participant_response_metrics(df: pd.DataFrame) -> pd.DataFrame:
    meta = participant_metadata(df)
    rows = []
    for participant, group in df.groupby("Participant", sort=False):
        pressure_avg = (
            group.groupby(["Pressure", "UpDown"], dropna=False)["Log_Video_Median_Velocity"]
            .mean()
            .unstack()
        )
        raw_pressure_avg = (
            group.groupby(["Pressure", "UpDown"], dropna=False)["Video_Median_Velocity"]
            .mean()
            .unstack()
        )
        avg_curve = pressure_avg.reindex(STANDARD_PRESSURES).mean(axis=1, skipna=True)
        raw_avg_curve = raw_pressure_avg.reindex(STANDARD_PRESSURES).mean(axis=1, skipna=True)

        raw_up = raw_pressure_avg["U"] if "U" in raw_pressure_avg.columns else pd.Series(dtype=float)
        raw_down = raw_pressure_avg["D"] if "D" in raw_pressure_avg.columns else pd.Series(dtype=float)
        up = pressure_avg["U"] if "U" in pressure_avg.columns else pd.Series(dtype=float)
        down = pressure_avg["D"] if "D" in pressure_avg.columns else pd.Series(dtype=float)
        common = [p for p in STANDARD_PRESSURES if p in up.index and p in down.index]
        log_up_down_diffs = up.reindex(common) - down.reindex(common) if common else pd.Series(dtype=float)
        raw_up_down_diffs = raw_up.reindex(common) - raw_down.reindex(common) if common else pd.Series(dtype=float)
        poles = [p for p in [0.2, 1.2] if p in common]
        log_pole_diffs = log_up_down_diffs.reindex(poles) if poles else pd.Series(dtype=float)
        raw_pole_diffs = raw_up_down_diffs.reindex(poles) if poles else pd.Series(dtype=float)

        row = {
            "Participant": participant,
            "cri_log_auc_02_12": _area(STANDARD_PRESSURES, avg_curve.to_numpy()),
            "raw_velocity_auc_02_12": _area(STANDARD_PRESSURES, raw_avg_curve.to_numpy()),
            "hysteresis_log_mean_up_minus_down": float(log_up_down_diffs.mean()) if len(log_up_down_diffs) else np.nan,
            "hysteresis_raw_mean_up_minus_down": float(raw_up_down_diffs.mean()) if len(raw_up_down_diffs) else np.nan,
            "hysteresis_log_pole_mean_up_minus_down": float(log_pole_diffs.mean()) if len(log_pole_diffs) else np.nan,
            "hysteresis_raw_pole_mean_up_minus_down": float(raw_pole_diffs.mean()) if len(raw_pole_diffs) else np.nan,
            "hysteresis_log_0.2_up_minus_down": float(log_up_down_diffs.get(0.2, np.nan)),
            "hysteresis_log_1.2_up_minus_down": float(log_up_down_diffs.get(1.2, np.nan)),
            "hysteresis_raw_0.2_up_minus_down": float(raw_up_down_diffs.get(0.2, np.nan)),
            "hysteresis_raw_1.2_up_minus_down": float(raw_up_down_diffs.get(1.2, np.nan)),
            "log_velocity_0.2": avg_curve.get(0.2, np.nan),
            "log_velocity_0.4": avg_curve.get(0.4, np.nan),
            "log_velocity_1.2": avg_curve.get(1.2, np.nan),
            "early_delta_log_0.4_minus_0.2": avg_curve.get(0.4, np.nan) - avg_curve.get(0.2, np.nan),
            "early_drop_log_0.2_minus_0.4": avg_curve.get(0.2, np.nan) - avg_curve.get(0.4, np.nan),
            "late_delta_log_1.2_minus_0.4": avg_curve.get(1.2, np.nan) - avg_curve.get(0.4, np.nan),
            "n_standard_rows": len(group),
        }
        rows.append(row)
    metrics = pd.DataFrame(rows).merge(meta, on="Participant", how="left")
    return metrics


def metric_group_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "cri_log_auc_02_12",
        "raw_velocity_auc_02_12",
        "hysteresis_log_mean_up_minus_down",
        "hysteresis_raw_mean_up_minus_down",
        "hysteresis_log_pole_mean_up_minus_down",
        "hysteresis_raw_pole_mean_up_minus_down",
        "hysteresis_log_0.2_up_minus_down",
        "hysteresis_log_1.2_up_minus_down",
        "log_velocity_0.2",
        "log_velocity_0.4",
        "log_velocity_1.2",
        "early_delta_log_0.4_minus_0.2",
        "early_drop_log_0.2_minus_0.4",
        "late_delta_log_1.2_minus_0.4",
    ]
    rows = []
    for metric in metric_cols:
        sub = metrics.dropna(subset=[metric, "Diabetes_int", "Age", "Sex_M", "SYS_BP", "DIA_BP"]).copy()
        control = sub.loc[sub["Diabetes_int"].eq(0), metric]
        diabetes = sub.loc[sub["Diabetes_int"].eq(1), metric]
        if len(control) and len(diabetes):
            u_stat, mw_p = stats.mannwhitneyu(control, diabetes, alternative="two-sided")
        else:
            u_stat, mw_p = np.nan, np.nan
        model = smf.ols(
            f'Q("{metric}") ~ Diabetes_int + Age + Sex_M + SYS_BP + DIA_BP',
            data=sub,
        ).fit(cov_type="HC3")
        rows.append(
            {
                "metric": metric,
                "n_control": int(len(control)),
                "n_diabetes": int(len(diabetes)),
                "control_median": float(control.median()) if len(control) else np.nan,
                "diabetes_median": float(diabetes.median()) if len(diabetes) else np.nan,
                "median_difference_diabetes_minus_control": float(diabetes.median() - control.median()) if len(control) and len(diabetes) else np.nan,
                "mannwhitney_u": float(u_stat) if pd.notna(u_stat) else np.nan,
                "mannwhitney_p": float(mw_p) if pd.notna(mw_p) else np.nan,
                "age_bp_adjusted_diabetes_coef": float(model.params.get("Diabetes_int", np.nan)),
                "age_bp_adjusted_diabetes_se": float(model.bse.get("Diabetes_int", np.nan)),
                "age_bp_adjusted_diabetes_p": float(model.pvalues.get("Diabetes_int", np.nan)),
                "model_r2": float(model.rsquared),
            }
        )
    out = pd.DataFrame(rows)
    out["age_bp_adjusted_diabetes_p_fdr_bh"] = multipletest_bh(out["age_bp_adjusted_diabetes_p"].to_numpy())
    return out


def plot_adjusted_curves(curves: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    for group, color in [("Control", CONTROL_COLOR), ("Diabetes", DIABETES_COLOR)]:
        sub = curves[curves["Group"].eq(group)]
        ax.plot(sub["Pressure"], sub["adjusted_log_velocity"], marker="o", color=color, label=group)
        ax.fill_between(sub["Pressure"], sub["ci_low"], sub["ci_high"], color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Pressure (psi)")
    ax.set_ylabel("Adjusted log velocity")
    ax.set_title("Adjusted Pressure-Flow Response")
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.3)
    apply_font(ax)
    fig.tight_layout()
    save_manuscript_figure(fig, output_dir, "adjusted_pressure_flow_curves")
    plt.close(fig)


def plot_pressure_effects(effects: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.axhline(0, color="black", linewidth=0.7)
    ax.errorbar(
        effects["Pressure"],
        effects["diabetes_minus_control_log_velocity"],
        yerr=[
            effects["diabetes_minus_control_log_velocity"] - effects["ci_low"],
            effects["ci_high"] - effects["diabetes_minus_control_log_velocity"],
        ],
        fmt="o-",
        color=DIABETES_COLOR,
        markersize=3,
        capsize=2,
    )
    ax.set_xlabel("Pressure (psi)")
    ax.set_ylabel("Diabetes - control\nadjusted log velocity")
    ax.set_title("Pressure-Specific Diabetes Effect")
    ax.grid(True, linewidth=0.3)
    apply_font(ax)
    fig.tight_layout()
    save_manuscript_figure(fig, output_dir, "pressure_specific_diabetes_effects")
    plt.close(fig)


def plot_metric_panels(metrics: pd.DataFrame, output_dir: str) -> None:
    plot_metrics = [
        ("cri_log_auc_02_12", "CRI log AUC"),
        ("hysteresis_log_mean_up_minus_down", "Mean up-down diff"),
        ("hysteresis_log_pole_mean_up_minus_down", "Pole up-down diff"),
        ("early_drop_log_0.2_minus_0.4", "Early drop 0.2-0.4"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2))
    palette = {"Control": CONTROL_COLOR, "Diabetes": DIABETES_COLOR}
    for ax, (metric, label) in zip(axes, plot_metrics):
        sub = metrics.dropna(subset=[metric])
        sns.boxplot(
            data=sub,
            x="Group",
            y=metric,
            ax=ax,
            palette=palette,
            width=MANUSCRIPT_BOX_STYLE["box_width"],
            fliersize=0,
        )
        sns.stripplot(
            data=sub,
            x="Group",
            y=metric,
            ax=ax,
            color=MANUSCRIPT_BOX_STYLE["point_color"],
            size=MANUSCRIPT_BOX_STYLE["point_size"] / 3,
            alpha=MANUSCRIPT_BOX_STYLE["point_alpha"],
            jitter=0.12,
        )
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", linewidth=0.3)
        apply_font(ax)
    fig.tight_layout()
    save_manuscript_figure(fig, output_dir, "participant_response_metric_panels")
    plt.close(fig)


def plot_individual_metric_boxplots(metrics: pd.DataFrame, output_dir: str) -> None:
    """Save standalone participant metric boxplots using stiffness plot styling."""
    plot_metrics = [
        (
            "cri_log_auc_02_12",
            "CRI (AUC of log velocity)",
            "CRI by Diabetes",
            "participant_response_cri_log_auc_02_12_stiffness_diabetes_style",
        ),
        (
            "hysteresis_log_mean_up_minus_down",
            "Velocity Hysteresis (up-down)",
            "Mean Hysteresis by Diabetes",
            "participant_response_hysteresis_log_mean_up_minus_down_stiffness_diabetes_style",
        ),
        (
            "hysteresis_log_pole_mean_up_minus_down",
            "Velocity Hysteresis (up-down)",
            "Pole Hysteresis by Diabetes",
            "participant_response_hysteresis_log_pole_mean_up_minus_down_stiffness_diabetes_style",
        ),
        (
            "early_drop_log_0.2_minus_0.4",
            "Early Drop (0.2-0.4 psi)",
            "Early Drop by Diabetes",
            "participant_response_early_drop_log_02_minus_04_stiffness_diabetes_style",
        ),
    ]
    group_labels = ["Control", "Diabetic"]
    palette = [CONTROL_COLOR, DIABETES_COLOR]
    box_dir = os.path.join(output_dir, "individual_boxplots_stiffness_style")
    os.makedirs(box_dir, exist_ok=True)

    for metric, ylabel, title, filename in plot_metrics:
        sub = metrics.dropna(subset=[metric]).copy()
        sub["Plot_Group"] = sub["Group"].replace({"Diabetes": "Diabetic"})

        fig, ax = plt.subplots(figsize=MANUSCRIPT_BOX_STYLE["figsize_single"])
        sns.boxplot(
            data=sub,
            x="Plot_Group",
            y=metric,
            ax=ax,
            order=group_labels,
            palette=palette,
            width=MANUSCRIPT_BOX_STYLE["box_width"],
        )
        for i, group in enumerate(group_labels):
            group_vals = sub.loc[sub["Plot_Group"].eq(group), metric]
            x_pos = i + np.random.normal(
                0,
                MANUSCRIPT_BOX_STYLE["point_jitter_sd"],
                len(group_vals),
            )
            ax.scatter(
                x_pos,
                group_vals,
                alpha=MANUSCRIPT_BOX_STYLE["point_alpha"],
                s=MANUSCRIPT_BOX_STYLE["point_size"],
                color=MANUSCRIPT_BOX_STYLE["point_color"],
                zorder=3,
            )

        ax.set_xlabel("Group", fontproperties=source_sans if source_sans else None)
        ax.set_ylabel(ylabel, fontproperties=source_sans if source_sans else None)
        ax.set_title(title, fontproperties=source_sans if source_sans else None)
        apply_font(ax)
        fig.tight_layout()
        save_manuscript_figure(fig, box_dir, filename, png_subdir=False)
        plt.close(fig)


def write_report(
    output_dir: str,
    model_df: pd.DataFrame,
    model_result,
    formula: str,
    effects: pd.DataFrame,
    metric_stats: pd.DataFrame,
) -> None:
    path = os.path.join(output_dir, "pressure_flow_response_report.txt")
    participant_counts = model_df.groupby("Group")["Participant"].nunique()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Pressure-flow response analysis\n")
        handle.write("=" * 36 + "\n")
        handle.write("Cohort: diabetes versus healthy controls; part30 included as diabetes.\n")
        handle.write("Videos: all standard-range videos at 0.2-1.2 psi retained.\n\n")
        handle.write("Participant counts:\n")
        handle.write(participant_counts.to_string())
        handle.write(f"\nRows in repeated-measures model: {len(model_df)}\n")
        handle.write(f"Formula: {formula}\n\n")
        handle.write("GEE model coefficient table:\n")
        handle.write(model_result.summary().as_text())
        handle.write("\n\nPressure-specific adjusted diabetes-control effects:\n")
        handle.write(effects.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        handle.write("\n\nParticipant-level metric comparisons:\n")
        handle.write(metric_stats.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


def run(args: argparse.Namespace) -> None:
    cap_flow_path = args.cap_flow_path or PATHS["cap_flow"]
    output_dir = args.output_dir or os.path.join(cap_flow_path, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    df = load_analysis_data(cap_flow_path)
    model_df, centers = prepare_model_data(df)
    result, formula = fit_pressure_response_model(model_df)

    curves = adjusted_curve_table(result, centers)
    effects = pressure_specific_effects(result, centers)
    metrics = participant_response_metrics(df)
    metric_stats = metric_group_comparisons(metrics)

    model_df.to_csv(os.path.join(output_dir, "pressure_flow_model_rows.csv"), index=False)
    curves.to_csv(os.path.join(output_dir, "adjusted_pressure_flow_curves.csv"), index=False)
    effects.to_csv(os.path.join(output_dir, "pressure_specific_diabetes_effects.csv"), index=False)
    metrics.to_csv(os.path.join(output_dir, "participant_pressure_response_metrics.csv"), index=False)
    metric_stats.to_csv(os.path.join(output_dir, "participant_metric_group_comparisons.csv"), index=False)

    plot_adjusted_curves(curves, output_dir)
    plot_pressure_effects(effects, output_dir)
    plot_metric_panels(metrics, output_dir)
    plot_individual_metric_boxplots(metrics, output_dir)
    write_report(output_dir, model_df, result, formula, effects, metric_stats)

    print(f"Saved outputs to: {output_dir}")
    print("\nPressure-specific adjusted diabetes effects:")
    print(effects.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\nParticipant-level metric comparisons:")
    print(metric_stats.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-flow-path", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
