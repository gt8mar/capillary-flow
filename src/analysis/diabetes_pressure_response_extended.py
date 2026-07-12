"""Exploratory diabetes pressure-response models and trajectory plots.

This script does not update manuscript replacement text. It uses the current
diabetes-vs-healthy-control cohort definition, keeps part30 as diabetes, and
includes all standard pressure videos.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.config import PATHS
from src.analysis.pressure_flow_response_analysis import (
    DIABETES_COLOR,
    CONTROL_COLOR,
    STANDARD_PRESSURES,
    _bool_true,
    multipletest_bh,
)


OUTPUT_SUBDIR = Path("results/PressureFlowResponse_extended_diabetes")
HEIGHT_WEIGHT_PATH = "height_weight.csv"


def load_data(cap_flow_path: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(cap_flow_path, "summary_df_nhp_video_stats.csv"))
    df = df.loc[:, ~df.columns.duplicated()].copy()
    height_weight_path = os.path.join(cap_flow_path, HEIGHT_WEIGHT_PATH)
    if os.path.exists(height_weight_path):
        height_weight = pd.read_csv(height_weight_path)
        height_weight = height_weight.loc[:, ~height_weight.columns.duplicated()].copy()
        height_weight = height_weight.rename(
            columns={"Height": "Height_from_height_weight", "Weight": "Weight_from_height_weight"}
        )
        df = df.merge(
            height_weight[["Participant", "Height_from_height_weight", "Weight_from_height_weight"]],
            on="Participant",
            how="left",
        )
    df["Diabetes_bool"] = df["Diabetes"].map(_bool_true)
    df["Hypertension_bool"] = df["Hypertension"].map(_bool_true)
    df["is_healthy"] = (
        df["SET"].astype(str).str.startswith("set01")
        & ~df["Diabetes_bool"]
        & ~df["Hypertension_bool"]
    )
    for col in [
        "Log_Video_Median_Velocity",
        "Video_Median_Velocity",
        "Pressure",
        "Age",
        "SYS_BP",
        "DIA_BP",
        "Height",
        "Weight",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Height", "Weight"]:
        fill_col = f"{col}_from_height_weight"
        if fill_col in df.columns:
            df[fill_col] = pd.to_numeric(df[fill_col], errors="coerce")
            df[col] = df[col].fillna(df[fill_col])
    df["Log_Video_Median_Velocity"] = df["Log_Video_Median_Velocity"].clip(lower=0)
    df["Sex_M"] = df["Sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0})
    df["BMI"] = np.where(
        (df["Height"] > 0) & df["Weight"].notna(),
        703.0 * df["Weight"] / (df["Height"] ** 2),
        np.nan,
    )
    df = df[df["Diabetes_bool"] | df["is_healthy"]].copy()
    df = df[df["Pressure"].isin(STANDARD_PRESSURES)].copy()
    df["Group"] = np.where(df["Diabetes_bool"], "Diabetes", "Control")
    df["Diabetes_int"] = df["Diabetes_bool"].astype(int)
    df["PressureCat"] = pd.Categorical(
        df["Pressure"].map(lambda p: f"{p:.1f}"),
        categories=[f"{p:.1f}" for p in STANDARD_PRESSURES],
        ordered=True,
    )
    df["UpDown"] = df["UpDown"].fillna("Unknown").astype(str)
    return df


def prepare_model_df(df: pd.DataFrame, include_bmi: bool) -> tuple[pd.DataFrame, dict[str, float]]:
    covariates = ["Age", "Sex_M", "SYS_BP", "DIA_BP"]
    if include_bmi:
        covariates.append("BMI")
    model_df = df.dropna(
        subset=["Log_Video_Median_Velocity", "PressureCat", "Diabetes_int", "Participant", "UpDown"]
        + covariates
    ).copy()
    centers = {col: float(model_df[col].mean()) for col in covariates}
    for col in [c for c in covariates if c != "Sex_M"]:
        model_df[f"{col}_c"] = model_df[col] - centers[col]
    return model_df, centers


def fit_gee(model_df: pd.DataFrame, include_bmi: bool):
    covariates = ["Age_c", "Sex_M", "SYS_BP_c", "DIA_BP_c"]
    if include_bmi:
        covariates.append("BMI_c")
    formula = (
        'Log_Video_Median_Velocity ~ C(PressureCat, Treatment(reference="0.2")) '
        f"* Diabetes_int + {' + '.join(covariates)} + C(UpDown)"
    )
    model = smf.gee(
        formula=formula,
        groups="Participant",
        data=model_df,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    return model.fit(), formula


def fit_mixedlm(model_df: pd.DataFrame, include_bmi: bool):
    covariates = ["Age_c", "Sex_M", "SYS_BP_c", "DIA_BP_c"]
    if include_bmi:
        covariates.append("BMI_c")
    formula = (
        'Log_Video_Median_Velocity ~ C(PressureCat, Treatment(reference="0.2")) '
        f"* Diabetes_int + {' + '.join(covariates)} + C(UpDown)"
    )
    model = smf.mixedlm(formula=formula, groups="Participant", data=model_df)
    return model.fit(method="lbfgs", maxiter=500, reml=False), formula


def design_matrix(result, new_df: pd.DataFrame) -> pd.DataFrame:
    design_info = result.model.data.design_info
    return patsy.build_design_matrices([design_info], new_df, return_type="dataframe")[0]


def linear_estimate(result, design_row: np.ndarray) -> tuple[float, float, float, float, float]:
    params = result.params.to_numpy()
    cov = result.cov_params().to_numpy()
    if cov.shape[0] != len(params):
        cov = cov[: len(params), : len(params)]
    est = float(design_row @ params)
    se = float(np.sqrt(max(design_row @ cov @ design_row.T, 0.0)))
    z = est / se if se > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return est, se, est - 1.96 * se, est + 1.96 * se, p


def pressure_effects(result, centers: dict[str, float], include_bmi: bool) -> pd.DataFrame:
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
        if include_bmi:
            base["BMI_c"] = 0.0
        pair = pd.DataFrame(
            [
                {**base, "Group": "Control", "Diabetes_int": 0},
                {**base, "Group": "Diabetes", "Diabetes_int": 1},
            ]
        )
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


def interaction_terms(result) -> pd.DataFrame:
    rows = []
    for name, coef in result.params.items():
        if "Diabetes_int" in name and "PressureCat" in name:
            rows.append(
                {
                    "term": name,
                    "coef": float(coef),
                    "se": float(result.bse[name]),
                    "p_value": float(result.pvalues[name]),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_fdr_bh"] = multipletest_bh(out["p_value"].to_numpy())
    return out


def participant_curve_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Participant", "Group", "Diabetes_int", "Pressure"], as_index=False)
        .agg(
            mean_log_velocity=("Log_Video_Median_Velocity", "mean"),
            median_log_velocity=("Log_Video_Median_Velocity", "median"),
            n_videos=("Log_Video_Median_Velocity", "count"),
        )
    )


def plot_spaghetti(curves: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {"Control": CONTROL_COLOR, "Diabetes": DIABETES_COLOR}
    for (participant, group), sub in curves.groupby(["Participant", "Group"]):
        sub = sub.sort_values("Pressure")
        ax.plot(
            sub["Pressure"],
            sub["mean_log_velocity"],
            color=colors[group],
            alpha=0.20,
            linewidth=1.0,
        )
    summary = curves.groupby(["Group", "Pressure"], as_index=False).agg(
        mean=("mean_log_velocity", "mean"),
        se=("mean_log_velocity", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan),
    )
    for group, sub in summary.groupby("Group"):
        sub = sub.sort_values("Pressure")
        ax.plot(sub["Pressure"], sub["mean"], color=colors[group], linewidth=2.8, label=group)
        ax.fill_between(
            sub["Pressure"].to_numpy(dtype=float),
            (sub["mean"] - 1.96 * sub["se"]).to_numpy(dtype=float),
            (sub["mean"] + 1.96 * sub["se"]).to_numpy(dtype=float),
            color=colors[group],
            alpha=0.16,
            linewidth=0,
        )
    ax.set_xlabel("External pressure (psi)")
    ax.set_ylabel("Participant mean log velocity")
    ax.set_title("Participant pressure-flow trajectories", loc="left", fontsize=12, weight="bold")
    ax.set_xticks(STANDARD_PRESSURES)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "participant_pressure_flow_trajectories.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "participant_pressure_flow_trajectories.svg", bbox_inches="tight")
    fig.savefig(output_dir / "participant_pressure_flow_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_updown_spaghetti(df: pd.DataFrame, output_dir: Path) -> None:
    participant_updown = (
        df.groupby(["Participant", "Group", "UpDown", "Pressure"], as_index=False)
        .agg(mean_log_velocity=("Log_Video_Median_Velocity", "mean"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    colors = {"Control": CONTROL_COLOR, "Diabetes": DIABETES_COLOR}
    for ax, updown in zip(axes, ["U", "D"]):
        sub_df = participant_updown[participant_updown["UpDown"] == updown].copy()
        for (_participant, group), sub in sub_df.groupby(["Participant", "Group"]):
            sub = sub.sort_values("Pressure")
            ax.plot(sub["Pressure"], sub["mean_log_velocity"], color=colors[group], alpha=0.18, linewidth=0.9)
        summary = sub_df.groupby(["Group", "Pressure"], as_index=False).agg(mean=("mean_log_velocity", "mean"))
        for group, sub in summary.groupby("Group"):
            sub = sub.sort_values("Pressure")
            ax.plot(sub["Pressure"], sub["mean"], color=colors[group], linewidth=2.5, label=group)
        ax.set_title("Up ramp" if updown == "U" else "Down ramp", fontsize=11, weight="bold")
        ax.set_xlabel("External pressure (psi)")
        ax.set_xticks(STANDARD_PRESSURES)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    axes[0].set_ylabel("Participant mean log velocity")
    axes[1].legend(frameon=False)
    fig.savefig(output_dir / "participant_pressure_flow_trajectories_updown.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "participant_pressure_flow_trajectories_updown.svg", bbox_inches="tight")
    fig.savefig(output_dir / "participant_pressure_flow_trajectories_updown.pdf", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    cap_flow_path = args.cap_flow_path or PATHS["cap_flow"]
    output_dir = Path(args.output_dir or Path(cap_flow_path) / OUTPUT_SUBDIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(cap_flow_path)
    participant_curves = participant_curve_table(df)
    participant_curves.to_csv(output_dir / "participant_pressure_flow_curves.csv", index=False)
    plot_spaghetti(participant_curves, output_dir)
    plot_updown_spaghetti(df, output_dir)

    report_lines = [
        "Exploratory Diabetes Pressure-Response Extension",
        "================================================",
        "",
        "Definition: diabetes vs healthy controls; part30 included as diabetes; all standard 0.2-1.2 psi videos included; outcome is clipped log video median velocity.",
        "GEE adjustment: pressure category * diabetes + age + sex + systolic BP + diastolic BP + BMI + up/down, clustered by participant.",
        "",
    ]

    for include_bmi, label in [(False, "age_sex_bp"), (True, "age_sex_bp_bmi")]:
        model_df, centers = prepare_model_df(df, include_bmi=include_bmi)
        gee_result, gee_formula = fit_gee(model_df, include_bmi=include_bmi)
        effects = pressure_effects(gee_result, centers, include_bmi=include_bmi)
        interactions = interaction_terms(gee_result)
        effects.to_csv(output_dir / f"pressure_specific_diabetes_effects_{label}.csv", index=False)
        interactions.to_csv(output_dir / f"pressure_diabetes_interaction_terms_{label}.csv", index=False)

        report_lines.extend(
            [
                f"Model: {label}",
                f"Formula: {gee_formula}",
                f"Rows: {len(model_df)}, participants: {model_df['Participant'].nunique()}, diabetes participants: {model_df.loc[model_df['Diabetes_bool'], 'Participant'].nunique()}",
                "Pressure-specific diabetes-control effects:",
                effects.to_string(index=False),
                "",
                "Pressure interaction terms:",
                interactions.to_string(index=False) if len(interactions) else "None",
                "",
            ]
        )

        try:
            mixed_result, mixed_formula = fit_mixedlm(model_df, include_bmi=include_bmi)
            mixed_summary = pd.DataFrame(
                {
                    "term": mixed_result.params.index,
                    "coef": mixed_result.params.to_numpy(),
                    "se": mixed_result.bse.to_numpy(),
                    "p_value": mixed_result.pvalues.to_numpy(),
                }
            )
            mixed_summary.to_csv(output_dir / f"mixedlm_coefficients_{label}.csv", index=False)
            report_lines.extend(
                [
                    f"MixedLM random-intercept model also fit for {label}.",
                    f"MixedLM formula: {mixed_formula}",
                    "",
                ]
            )
        except Exception as exc:  # noqa: BLE001 - record convergence/fitting failures in report
            report_lines.extend([f"MixedLM random-intercept model failed for {label}: {exc}", ""])

    (output_dir / "extended_pressure_response_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(f"Wrote exploratory pressure-response outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-flow-path", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
