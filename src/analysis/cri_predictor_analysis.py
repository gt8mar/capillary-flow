"""
Filename: src/analysis/cri_predictor_analysis.py

Analyze how demographic and blood-pressure variables affect the
Compression Resistance Index (CRI), and compare those predictors against
systolic blood pressure as an outcome.

CRI is defined here as SI_logvel_averaged_02_12: the area under the averaged
up/down log-velocity curve from 0.2 to 1.2 psi.
"""

import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.config import PATHS


CRI_SOURCE_COL = "SI_logvel_averaged_02_12"
CRI_COL = "CRI"
CONTROL_SET = "set01"


def get_cap_flow_path() -> str:
    """Return the configured capillary-flow path, falling back to repo root."""
    configured = PATHS.get("cap_flow")
    if configured and os.path.exists(configured):
        return configured

    return str(Path(__file__).resolve().parents[2])


cap_flow_path = get_cap_flow_path()


def setup_plotting() -> None:
    """Set publication-friendly plotting defaults."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 0.8,
        }
    )


def load_cri_data() -> pd.DataFrame:
    """Load controls-only participant-level CRI data."""
    data_path = os.path.join(
        cap_flow_path, "results", "Stiffness", "stiffness_coefficients_log.csv"
    )
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Missing stiffness coefficient file: {data_path}. "
            "Run src/analysis/stiffness_coeff.py first."
        )

    df = pd.read_csv(data_path)
    required_cols = ["Participant", CRI_SOURCE_COL, "Age", "Sex", "SYS_BP", "SET"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    df = df[df["SET"] == CONTROL_SET].copy()
    df = df.rename(columns={CRI_SOURCE_COL: CRI_COL})

    for col in [CRI_COL, "Age", "SYS_BP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Sex"] = df["Sex"].astype(str).str.strip()
    df = df.replace({"Sex": {"": np.nan, "nan": np.nan, "None": np.nan}})
    df = df.dropna(subset=[CRI_COL, "Age", "Sex", "SYS_BP"])
    df = df.reset_index(drop=True)

    if len(df) < 5:
        raise ValueError(f"Insufficient complete control records for analysis: n={len(df)}")

    print(f"Loaded {len(df)} controls with complete CRI/Age/Sex/SBP data.")
    print(f"CRI source column: {CRI_SOURCE_COL}")
    return df


def fit_ols_anova(df: pd.DataFrame, formula: str) -> Dict:
    """Fit OLS and return a Type-II ANOVA table."""
    model = smf.ols(formula, data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    return {"formula": formula, "model": model, "anova": anova}


def run_models(df: pd.DataFrame) -> Dict[str, Dict]:
    """Run CRI and SBP predictor models."""
    models = {
        "cri_main": fit_ols_anova(df, "CRI ~ Age + C(Sex) + SYS_BP"),
        "cri_interactions": fit_ols_anova(
            df,
            "CRI ~ Age + C(Sex) + SYS_BP + Age:C(Sex) + Age:SYS_BP + C(Sex):SYS_BP",
        ),
        "sbp_main": fit_ols_anova(df, "SYS_BP ~ Age + C(Sex)"),
        "sbp_interactions": fit_ols_anova(df, "SYS_BP ~ Age + C(Sex) + Age:C(Sex)"),
    }

    for name, result in models.items():
        print("\n" + "=" * 72)
        print(name)
        print("Formula:", result["formula"])
        print(result["anova"])

    return models


def save_anova_tables(models: Dict[str, Dict], output_dir: str) -> None:
    """Save ANOVA and coefficient tables."""
    table_names = {
        "cri_main": "cri_anova_main.csv",
        "cri_interactions": "cri_anova_interactions.csv",
        "sbp_main": "sbp_anova_main.csv",
        "sbp_interactions": "sbp_anova_interactions.csv",
    }

    for model_name, filename in table_names.items():
        out_path = os.path.join(output_dir, filename)
        models[model_name]["anova"].reset_index(names="Factor").to_csv(out_path, index=False)

    coefficient_rows = []
    for model_name, result in models.items():
        model = result["model"]
        conf = model.conf_int()
        for term in model.params.index:
            coefficient_rows.append(
                {
                    "model": model_name,
                    "formula": result["formula"],
                    "term": term,
                    "coef": model.params[term],
                    "std_err": model.bse[term],
                    "t": model.tvalues[term],
                    "p_value": model.pvalues[term],
                    "ci_low": conf.loc[term, 0],
                    "ci_high": conf.loc[term, 1],
                    "n": int(model.nobs),
                    "r_squared": model.rsquared,
                    "adj_r_squared": model.rsquared_adj,
                    "aic": model.aic,
                    "bic": model.bic,
                }
            )

    pd.DataFrame(coefficient_rows).to_csv(
        os.path.join(output_dir, "model_coefficients.csv"), index=False
    )


def plot_diagnostics(df: pd.DataFrame, models: Dict[str, Dict], output_dir: str) -> None:
    """Create residual, Q-Q, and observed-vs-predicted plots for main models."""
    diagnostic_specs = {
        "cri_main": ("CRI", "Compression Resistance Index (CRI)"),
        "sbp_main": ("SYS_BP", "Systolic Blood Pressure (mmHg)"),
    }

    for model_name, (outcome_col, outcome_label) in diagnostic_specs.items():
        model = models[model_name]["model"]
        residuals = model.resid
        fitted = model.fittedvalues
        observed = df[outcome_col]

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        fig.suptitle(f"Diagnostics: {model_name}")

        sns.scatterplot(x=fitted, y=residuals, ax=axes[0])
        axes[0].axhline(0, color="red", linestyle="--", linewidth=0.8)
        axes[0].set_xlabel("Fitted")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Fitted")

        sm.graphics.qqplot(residuals, line="45", fit=True, ax=axes[1])
        axes[1].set_title("Q-Q Plot")

        sns.scatterplot(x=observed, y=fitted, ax=axes[2])
        min_val = min(observed.min(), fitted.min())
        max_val = max(observed.max(), fitted.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=0.8)
        axes[2].set_xlabel(f"Observed {outcome_label}")
        axes[2].set_ylabel(f"Predicted {outcome_label}")
        axes[2].set_title("Observed vs Predicted")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_diagnostics.png"), dpi=300)
        plt.close()


def plot_predictor_relationships(df: pd.DataFrame, output_dir: str) -> None:
    """Create predictor/outcome relationship plots."""
    plot_specs = [
        ("Age", "CRI", "CRI vs Age", "cri_vs_age.png"),
        ("SYS_BP", "CRI", "CRI vs Systolic Blood Pressure", "cri_vs_sbp.png"),
        ("Age", "SYS_BP", "Systolic Blood Pressure vs Age", "sbp_vs_age.png"),
    ]

    for x_col, y_col, title, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue="Sex", s=45, ax=ax)
        sns.regplot(data=df, x=x_col, y=y_col, scatter=False, color="black", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Systolic Blood Pressure (mmHg)" if x_col == "SYS_BP" else x_col)
        ax.set_ylabel(
            "Compression Resistance Index (CRI)" if y_col == "CRI" else "Systolic Blood Pressure (mmHg)"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()


def write_summary_report(df: pd.DataFrame, models: Dict[str, Dict], output_dir: str) -> None:
    """Write a concise text report for model formulas and key statistics."""
    report_path = os.path.join(output_dir, "cri_predictor_analysis_summary.txt")
    with open(report_path, "w") as f:
        f.write("CRI Predictor Analysis Summary\n")
        f.write("=" * 32 + "\n\n")
        f.write(f"Data source: results/Stiffness/stiffness_coefficients_log.csv\n")
        f.write(f"CRI definition: {CRI_SOURCE_COL}\n")
        f.write("Cohort: controls only (SET == 'set01')\n")
        f.write(f"Complete participant records: {len(df)}\n\n")
        f.write("Mixed-effects note:\n")
        f.write(
            "CRI and SYS_BP are participant-level outcomes in this dataset, so each "
            "participant contributes one complete row. Mixed-effects models are not "
            "fit because there are no repeated observations for these outcomes.\n\n"
        )

        f.write("Descriptive statistics:\n")
        f.write(df[[CRI_COL, "Age", "SYS_BP"]].describe().to_string())
        f.write("\n\nSex counts:\n")
        f.write(df["Sex"].value_counts(dropna=False).to_string())
        f.write("\n\n")

        for model_name, result in models.items():
            model = result["model"]
            f.write("=" * 72 + "\n")
            f.write(f"{model_name}\n")
            f.write(f"Formula: {result['formula']}\n")
            f.write(f"n = {int(model.nobs)}\n")
            f.write(f"R-squared = {model.rsquared:.5f}\n")
            f.write(f"Adjusted R-squared = {model.rsquared_adj:.5f}\n")
            f.write(f"AIC = {model.aic:.5f}\n")
            f.write(f"BIC = {model.bic:.5f}\n\n")
            f.write("ANOVA:\n")
            f.write(result["anova"].to_string())
            f.write("\n\nCoefficients:\n")
            coef_table = pd.DataFrame(
                {
                    "coef": model.params,
                    "std_err": model.bse,
                    "t": model.tvalues,
                    "p_value": model.pvalues,
                }
            )
            f.write(coef_table.to_string())
            f.write("\n\n")

    print(f"Summary report written to: {report_path}")


def main() -> int:
    """Run the CRI and SBP predictor analysis."""
    print("\nRunning CRI predictor analysis...")
    setup_plotting()

    output_dir = os.path.join(cap_flow_path, "results", "CRI_Analysis")
    os.makedirs(output_dir, exist_ok=True)

    df = load_cri_data()
    df.to_csv(os.path.join(output_dir, "cri_analysis_dataset.csv"), index=False)

    models = run_models(df)
    save_anova_tables(models, output_dir)
    plot_diagnostics(df, models, output_dir)
    plot_predictor_relationships(df, output_dir)
    write_summary_report(df, models, output_dir)

    print(f"\nCRI predictor analysis complete. Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
