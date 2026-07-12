"""
Calculate control-group CRI summaries and export linear CRI ANOVA tables.

Inputs:
    results/Stiffness/stiffness_coefficients_log.csv

Outputs:
    results/CRI_Analysis/cri_control_age_group_summary.csv
    results/CRI_Analysis/cri_control_age_group_summary.txt
    results/CRI_Analysis/cri_linear_anova_latex.txt
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.config import PATHS


CRI_SOURCE_COL = "SI_logvel_averaged_02_12"
CRI_COL = "CRI"
CONTROL_SET = "set01"
LINEAR_FORMULA = "CRI ~ Age + C(Sex) + SYS_BP"


def get_cap_flow_path() -> str:
    """Return configured capillary-flow path, falling back to the repo root."""
    configured = PATHS.get("cap_flow")
    if configured and os.path.exists(configured):
        return configured
    return str(Path(__file__).resolve().parents[2])


def load_control_cri_data(cap_flow_path: str) -> pd.DataFrame:
    """Load participant-level control CRI data."""
    data_path = os.path.join(
        cap_flow_path, "results", "Stiffness", "stiffness_coefficients_log.csv"
    )
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Missing stiffness coefficient file: {data_path}. "
            "Run src/analysis/stiffness_coeff.py first."
        )

    df = pd.read_csv(data_path)
    required = ["Participant", CRI_SOURCE_COL, "Age", "Sex", "SYS_BP", "SET"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    df = df[df["SET"] == CONTROL_SET].copy()
    df[CRI_COL] = df[CRI_SOURCE_COL]

    for col in [CRI_COL, "Age", "SYS_BP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Sex"] = df["Sex"].astype(str).str.strip()
    df = df.replace({"Sex": {"": np.nan, "nan": np.nan, "None": np.nan}})
    df = df.dropna(subset=[CRI_COL, "Age", "Sex", "SYS_BP"])
    return df.reset_index(drop=True)


def iqr(values: pd.Series) -> float:
    """Return Q3 - Q1 using pandas default quantile interpolation."""
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return float(q3 - q1)


def calculate_age_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate requested median and IQR summaries for controls."""
    group_specs = [
        ("controls_under_30", "Controls under 30", df["Age"] < 30),
        ("controls_over_30", "Controls over 30", df["Age"] > 30),
        ("controls_under_60", "Controls under 60", df["Age"] < 60),
        ("controls_over_60", "Controls over 60", df["Age"] > 60),
    ]

    rows = []
    for group_id, label, mask in group_specs:
        group = df.loc[mask].copy()
        cri = group[CRI_COL].dropna()
        rows.append(
            {
                "group_id": group_id,
                "group": label,
                "age_filter": label.replace("Controls ", "Age "),
                "n": int(len(cri)),
                "median_cri": float(cri.median()) if len(cri) else np.nan,
                "q1_cri": float(cri.quantile(0.25)) if len(cri) else np.nan,
                "q3_cri": float(cri.quantile(0.75)) if len(cri) else np.nan,
                "iqr_cri": iqr(cri) if len(cri) else np.nan,
                "min_age": float(group["Age"].min()) if len(group) else np.nan,
                "max_age": float(group["Age"].max()) if len(group) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def fit_linear_anova(df: pd.DataFrame):
    """Fit the linear CRI model and Type-II ANOVA."""
    model = smf.ols(LINEAR_FORMULA, data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    return model, anova


def format_p(value: float) -> str:
    """Format p-values for compact LaTeX tables."""
    if pd.isna(value):
        return ""
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def write_summary_text(summary_df: pd.DataFrame, output_path: str) -> None:
    """Write a human-readable summary of the requested age groups."""
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("Control CRI Age-Group Summary\n")
        handle.write("=" * 30 + "\n\n")
        handle.write("Controls are defined as SET == 'set01'.\n")
        handle.write("Age groups use strict cutoffs: under means <, over means >.\n\n")
        for _, row in summary_df.iterrows():
            handle.write(f"{row['group']} ({row['age_filter']}):\n")
            handle.write(f"  n = {int(row['n'])}\n")
            handle.write(f"  median CRI = {row['median_cri']:.3f}\n")
            handle.write(f"  IQR = {row['iqr_cri']:.3f}\n")
            handle.write(f"  Q1-Q3 = {row['q1_cri']:.3f}-{row['q3_cri']:.3f}\n")
            handle.write(f"  age range = {row['min_age']:.0f}-{row['max_age']:.0f}\n\n")


def latex_table(df: pd.DataFrame, caption: str, label: str, column_format: str) -> str:
    """Return a LaTeX table string from a display-ready dataframe."""
    return df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format=column_format,
    )


def write_latex_anova_report(
    df: pd.DataFrame,
    model,
    anova: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Write LaTeX-exportable text for the linear CRI ANOVA."""
    anova_export = anova.reset_index(names="Factor").copy()
    anova_export = anova_export.rename(
        columns={
            "sum_sq": "Sum of squares",
            "df": "df",
            "F": "F",
            "PR(>F)": "p",
        }
    )
    anova_export["Sum of squares"] = anova_export["Sum of squares"].map(
        lambda value: f"{value:.3f}"
    )
    anova_export["df"] = anova_export["df"].map(lambda value: f"{value:.0f}")
    anova_export["F"] = anova_export["F"].map(
        lambda value: "" if pd.isna(value) else f"{value:.3f}"
    )
    anova_export["p"] = anova_export["p"].map(format_p)

    conf = model.conf_int()
    coef_rows = []
    for term in model.params.index:
        coef_rows.append(
            {
                "Term": term,
                "Estimate": f"{model.params[term]:.3f}",
                "SE": f"{model.bse[term]:.3f}",
                "t": f"{model.tvalues[term]:.3f}",
                "p": format_p(model.pvalues[term]),
                "95\\% CI": f"{conf.loc[term, 0]:.3f} to {conf.loc[term, 1]:.3f}",
            }
        )
    coef_export = pd.DataFrame(coef_rows)

    summary_export = summary_df[
        ["group", "n", "median_cri", "q1_cri", "q3_cri", "iqr_cri"]
    ].copy()
    summary_export = summary_export.rename(
        columns={
            "group": "Group",
            "n": "n",
            "median_cri": "Median CRI",
            "q1_cri": "Q1",
            "q3_cri": "Q3",
            "iqr_cri": "IQR",
        }
    )
    for col in ["Median CRI", "Q1", "Q3", "IQR"]:
        summary_export[col] = summary_export[col].map(lambda value: f"{value:.3f}")

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("% LaTeX-exportable CRI linear ANOVA report\n")
        handle.write("% Data source: results/Stiffness/stiffness_coefficients_log.csv\n")
        handle.write("% Cohort: controls only (SET == 'set01')\n")
        handle.write(f"% Formula: {LINEAR_FORMULA}\n")
        handle.write(f"% n = {int(model.nobs)}\n")
        handle.write(f"% R-squared = {model.rsquared:.5f}\n")
        handle.write(f"% Adjusted R-squared = {model.rsquared_adj:.5f}\n\n")

        handle.write(
            latex_table(
                summary_export,
                "Control CRI medians and interquartile ranges by age group.",
                "tab:control-cri-age-summary",
                "lrrrrr",
            )
        )
        handle.write("\n\n")
        handle.write(
            latex_table(
                anova_export,
                "Type-II ANOVA for the linear CRI model in controls.",
                "tab:cri-linear-anova",
                "lrrrr",
            )
        )
        handle.write("\n\n")
        handle.write(
            latex_table(
                coef_export,
                "Linear CRI model coefficients in controls.",
                "tab:cri-linear-coefficients",
                "lrrrrl",
            )
        )


def main() -> int:
    cap_flow_path = get_cap_flow_path()
    output_dir = os.path.join(cap_flow_path, "results", "CRI_Analysis")
    os.makedirs(output_dir, exist_ok=True)

    df = load_control_cri_data(cap_flow_path)
    summary_df = calculate_age_group_summary(df)
    model, anova = fit_linear_anova(df)

    summary_csv = os.path.join(output_dir, "cri_control_age_group_summary.csv")
    summary_txt = os.path.join(output_dir, "cri_control_age_group_summary.txt")
    latex_txt = os.path.join(output_dir, "cri_linear_anova_latex.txt")

    summary_df.to_csv(summary_csv, index=False)
    write_summary_text(summary_df, summary_txt)
    write_latex_anova_report(df, model, anova, summary_df, latex_txt)

    print(f"Loaded {len(df)} controls.")
    print(summary_df[["group", "n", "median_cri", "iqr_cri"]].to_string(index=False))
    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved summary TXT: {summary_txt}")
    print(f"Saved LaTeX ANOVA TXT: {latex_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
