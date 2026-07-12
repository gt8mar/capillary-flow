"""Exploratory CRI sensitivity to log velocity offset.

This script does not modify the manuscript CRI pipeline. It recomputes CRI from
raw video median velocity using log(velocity + offset), compares offsets 1 and
10, and reruns the main CRI-related significance checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf

from src.config import PATHS


STANDARD_PRESSURES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
DEFAULT_OUTPUT = Path("results/CRI_log_offset_sensitivity")


def bool_true(value: object) -> bool:
    return str(value).strip().upper() == "TRUE"


def load_raw(cap_flow_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cap_flow_path / "summary_df_nhp_video_stats.csv")
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["Pressure"] = pd.to_numeric(df["Pressure"], errors="coerce")
    df["Video_Median_Velocity"] = pd.to_numeric(df["Video_Median_Velocity"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["SYS_BP"] = pd.to_numeric(df["SYS_BP"], errors="coerce")
    df["DIA_BP"] = pd.to_numeric(df["DIA_BP"], errors="coerce")
    df["Diabetes_bool"] = df["Diabetes"].map(bool_true)
    df["Hypertension_bool"] = df["Hypertension"].map(bool_true)
    df["is_healthy"] = (
        df["SET"].astype(str).str.startswith("set01")
        & ~df["Diabetes_bool"]
        & ~df["Hypertension_bool"]
    )
    df["Sex_M"] = df["Sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0})

    height_weight_path = cap_flow_path / "height_weight.csv"
    if height_weight_path.exists():
        hw = pd.read_csv(height_weight_path).rename(
            columns={"Height": "Height_hw", "Weight": "Weight_hw"}
        )
        df = df.merge(hw[["Participant", "Height_hw", "Weight_hw"]], on="Participant", how="left")
    for col in ["Height", "Weight", "Height_hw", "Weight_hw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Height_hw" in df.columns:
        df["Height"] = df["Height"].fillna(df["Height_hw"])
    if "Weight_hw" in df.columns:
        df["Weight"] = df["Weight"].fillna(df["Weight_hw"])
    df["BMI"] = np.where(
        (df["Height"] > 0) & df["Weight"].notna(),
        703.0 * df["Weight"] / (df["Height"] ** 2),
        np.nan,
    )
    return df[df["Pressure"].isin(STANDARD_PRESSURES)].copy()


def auc(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    order = np.argsort(x[valid])
    return float(np.trapz(y[valid][order], x[valid][order]))


def calculate_offset_cri(df: pd.DataFrame, offset: float) -> pd.DataFrame:
    working = df.copy()
    working["offset_log_velocity"] = np.log(working["Video_Median_Velocity"].clip(lower=0) + offset)
    rows = []
    for participant, group in working.groupby("Participant", sort=False):
        pressure_updown = (
            group.groupby(["Pressure", "UpDown"], dropna=False)["offset_log_velocity"]
            .mean()
            .unstack()
            .reindex(STANDARD_PRESSURES)
        )
        curves = []
        if "U" in pressure_updown.columns:
            curves.append(pressure_updown["U"].to_numpy(dtype=float))
        if "D" in pressure_updown.columns:
            curves.append(pressure_updown["D"].to_numpy(dtype=float))
        if curves:
            mean_curve = np.nanmean(np.vstack(curves), axis=0)
            cri = auc(np.asarray(STANDARD_PRESSURES, dtype=float), mean_curve)
        else:
            cri = np.nan
        first = group.iloc[0]
        rows.append(
            {
                "Participant": participant,
                f"CRI_log_offset_{offset:g}": cri,
                "Age": first["Age"],
                "Sex_M": first["Sex_M"],
                "SYS_BP": first["SYS_BP"],
                "DIA_BP": first["DIA_BP"],
                "BMI": first["BMI"],
                "Diabetes": bool(first["Diabetes_bool"]),
                "Hypertension": bool(first["Hypertension_bool"]),
                "is_healthy": bool(first["is_healthy"]),
                "SET": first["SET"],
            }
        )
    return pd.DataFrame(rows)


def mwu_row(name: str, df: pd.DataFrame, value_col: str, mask_a, mask_b, group_a: str, group_b: str) -> dict:
    a = df.loc[mask_a, value_col].dropna()
    b = df.loc[mask_b, value_col].dropna()
    p = np.nan
    stat = np.nan
    if len(a) and len(b):
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "analysis": name,
        "test": "Mann-Whitney U",
        "outcome": value_col,
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "median_a": float(a.median()) if len(a) else np.nan,
        "iqr_a_low": float(a.quantile(0.25)) if len(a) else np.nan,
        "iqr_a_high": float(a.quantile(0.75)) if len(a) else np.nan,
        "median_b": float(b.median()) if len(b) else np.nan,
        "iqr_b_low": float(b.quantile(0.25)) if len(b) else np.nan,
        "iqr_b_high": float(b.quantile(0.75)) if len(b) else np.nan,
        "statistic": float(stat) if np.isfinite(stat) else np.nan,
        "p_value": float(p) if np.isfinite(p) else np.nan,
    }


def ols_term_row(name: str, df: pd.DataFrame, formula: str, term: str) -> dict:
    variables = sorted(set([term.split(":")[0], "CRI"]))
    model = smf.ols(formula, data=df).fit()
    return {
        "analysis": name,
        "test": "OLS",
        "outcome": "CRI",
        "formula": formula,
        "term": term,
        "n": int(model.nobs),
        "coef": float(model.params.get(term, np.nan)),
        "p_value": float(model.pvalues.get(term, np.nan)),
        "r_squared": float(model.rsquared),
    }


def run_tests(metrics: pd.DataFrame, cri_col: str) -> pd.DataFrame:
    d = metrics.copy()
    d["CRI"] = d[cri_col]
    d["Group"] = d["Diabetes"].astype(bool).astype(int)
    rows = []
    controls = d["is_healthy"].astype(bool)
    diabetes = d["Diabetes"].astype(bool)
    hypertension = d["Hypertension"].astype(bool) & d["SET"].astype(str).eq("set02")

    rows.append(
        mwu_row(
            "Control CRI, age <30 vs >=30",
            d,
            "CRI",
            controls & (d["Age"] < 30),
            controls & (d["Age"] >= 30),
            "Control <30",
            "Control >=30",
        )
    )
    rows.append(
        mwu_row(
            "Control CRI, age <60 vs >=60",
            d,
            "CRI",
            controls & (d["Age"] < 60),
            controls & (d["Age"] >= 60),
            "Control <60",
            "Control >=60",
        )
    )
    rows.append(
        mwu_row(
            "CRI, diabetes vs healthy controls",
            d,
            "CRI",
            controls,
            diabetes,
            "Control",
            "Diabetes",
        )
    )
    rows.append(
        mwu_row(
            "CRI, hypertension vs healthy controls",
            d,
            "CRI",
            controls,
            hypertension,
            "Control",
            "Hypertension",
        )
    )

    diabetes_subset = d[controls | diabetes].dropna(subset=["CRI", "Age", "Group"]).copy()
    rows.append(ols_term_row("CRI ~ diabetes + age", diabetes_subset, "CRI ~ Group + Age", "Group"))
    rows.append(ols_term_row("CRI ~ age * diabetes", diabetes_subset, "CRI ~ Age * Group", "Group"))
    rows.append(ols_term_row("CRI ~ age * diabetes", diabetes_subset, "CRI ~ Age * Group", "Age:Group"))

    adjusted = diabetes_subset.dropna(subset=["SYS_BP", "DIA_BP", "BMI"]).copy()
    rows.append(
        ols_term_row(
            "CRI ~ diabetes + age + BP + BMI",
            adjusted,
            "CRI ~ Group + Age + SYS_BP + DIA_BP + BMI",
            "Group",
        )
    )
    return pd.DataFrame(rows)


def plot_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    colors = np.where(metrics["Diabetes"], "#F58518", np.where(metrics["is_healthy"], "#4C78A8", "#7f7f7f"))
    ax.scatter(metrics["CRI_log_offset_1"], metrics["CRI_log_offset_10"], c=colors, s=42, alpha=0.85)
    lims = [
        min(metrics["CRI_log_offset_1"].min(), metrics["CRI_log_offset_10"].min()) - 0.1,
        max(metrics["CRI_log_offset_1"].max(), metrics["CRI_log_offset_10"].max()) + 0.1,
    ]
    ax.plot(lims, lims, color="#6b7280", linewidth=1, linestyle="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("CRI from log(velocity + 1)")
    ax.set_ylabel("CRI from log(velocity + 10)")
    ax.set_title("CRI sensitivity to log offset", loc="left", fontsize=12, weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#e5e7eb", linewidth=0.8)
    fig.savefig(output_dir / "cri_log_offset_1_vs_10.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "cri_log_offset_1_vs_10.svg", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    root = Path(args.cap_flow_path or PATHS["cap_flow"])
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(root)
    offset_1 = calculate_offset_cri(raw, 1)
    offset_10 = calculate_offset_cri(raw, 10)
    metrics = offset_1.merge(
        offset_10[["Participant", "CRI_log_offset_10"]], on="Participant", how="outer"
    )
    metrics["delta_offset_10_minus_1"] = metrics["CRI_log_offset_10"] - metrics["CRI_log_offset_1"]
    metrics.to_csv(output_dir / "cri_log_offset_participant_metrics.csv", index=False)

    rows = []
    for offset, cri_col in [(1, "CRI_log_offset_1"), (10, "CRI_log_offset_10")]:
        tested = run_tests(metrics, cri_col)
        tested.insert(0, "log_offset", offset)
        rows.append(tested)
    tests = pd.concat(rows, ignore_index=True)
    tests["significant_0.05"] = tests["p_value"] < 0.05
    tests.to_csv(output_dir / "cri_log_offset_significance_tests.csv", index=False)

    wide = tests.pivot_table(
        index=["analysis", "test", "term"],
        columns="log_offset",
        values=["p_value", "significant_0.05"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_offset_{offset:g}" for metric, offset in wide.columns]
    wide = wide.reset_index()
    if "significant_0.05_offset_1" in wide.columns and "significant_0.05_offset_10" in wide.columns:
        wide["significance_changed"] = wide["significant_0.05_offset_1"] != wide["significant_0.05_offset_10"]
    wide.to_csv(output_dir / "cri_log_offset_significance_changes.csv", index=False)

    rho, rho_p = stats.spearmanr(metrics["CRI_log_offset_1"], metrics["CRI_log_offset_10"], nan_policy="omit")
    pearson_r, pearson_p = stats.pearsonr(
        metrics.dropna(subset=["CRI_log_offset_1", "CRI_log_offset_10"])["CRI_log_offset_1"],
        metrics.dropna(subset=["CRI_log_offset_1", "CRI_log_offset_10"])["CRI_log_offset_10"],
    )
    plot_comparison(metrics, output_dir)

    changed = wide[wide.get("significance_changed", False) == True] if "significance_changed" in wide else pd.DataFrame()
    report = [
        "CRI Log Offset Sensitivity",
        "==========================",
        "",
        "Exploratory only. Existing CRI code and manuscript replacement text were not changed.",
        "Definition: CRI is AUC from 0.2-1.2 psi of averaged up/down log-transformed video median velocity.",
        "Comparison: log(velocity + 1) versus log(velocity + 10).",
        "",
        f"Participants with both CRI variants: {int(metrics[['CRI_log_offset_1', 'CRI_log_offset_10']].dropna().shape[0])}",
        f"Spearman rho between CRI variants: {rho:.4f} (p={rho_p:.3g})",
        f"Pearson r between CRI variants: {pearson_r:.4f} (p={pearson_p:.3g})",
        "",
        "Significance changes at alpha=0.05:",
        changed.to_string(index=False) if len(changed) else "None",
        "",
        "All test rows:",
        tests.to_string(index=False),
    ]
    (output_dir / "cri_log_offset_sensitivity_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote CRI log-offset sensitivity outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-flow-path", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
