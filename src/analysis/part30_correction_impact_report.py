"""Summarize manuscript-facing number changes after correcting part30.

Outputs contain participant IDs only where needed for auditability and no direct
participant names, emails, or phone numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu
import statsmodels.formula.api as smf
import statsmodels.api as sm


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "part30_diabetes_correction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FP = REPO_ROOT / "summary_df_nhp_video_stats.csv"
HW_FP = REPO_ROOT / "height_weight.csv"
CRI_FP = REPO_ROOT / "results" / "Stiffness" / "cri_metrics.csv"
TABLE1_FP = REPO_ROOT / "docs" / "current_draft" / "table1_median_iqr_latex.txt"


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().eq("TRUE")


def fmt1(value: float) -> str:
    return f"{value:.1f}"


def fmt0(value: float) -> str:
    return f"{value:.0f}"


def median_iqr(values: pd.Series, decimals: int = 1) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return "--"
    q1, med, q3 = clean.quantile([0.25, 0.5, 0.75])
    formatter = fmt0 if decimals == 0 else fmt1
    return f"{formatter(med)} ({formatter(q1)}--{formatter(q3)})"


def stats_tuple(values: pd.Series) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    q1, med, q3 = clean.quantile([0.25, 0.5, 0.75])
    return float(med), float(q1), float(q3)


def p_mwu(a: pd.Series, b: pd.Series) -> float:
    return float(mannwhitneyu(pd.to_numeric(a, errors="coerce").dropna(),
                              pd.to_numeric(b, errors="coerce").dropna(),
                              alternative="two-sided").pvalue)


def load_video_df() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_FP)
    if HW_FP.exists():
        hw = pd.read_csv(HW_FP)
        df = df.merge(hw, on="Participant", how="outer", suffixes=("_x", "_y"))
        for col in ["Height", "Weight"]:
            x_col = f"{col}_x"
            y_col = f"{col}_y"
            if x_col in df.columns and y_col in df.columns:
                df[col] = df[x_col].combine_first(df[y_col])
                df = df.drop(columns=[x_col, y_col])

    for col in ["Age", "Height", "Weight", "SYS_BP", "DIA_BP", "Pressure", "Video_Median_Velocity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Diabetes"] = as_bool(df["Diabetes"])
    df["Hypertension"] = as_bool(df["Hypertension"])
    return df


def build_participant_df(df: pd.DataFrame) -> pd.DataFrame:
    table_df = df.dropna(subset=["SET"]).copy()
    participant_df = table_df.sort_values(["Participant", "Date", "Location", "Video"]).drop_duplicates("Participant").set_index("Participant")

    pressure_stats = table_df.pivot_table(
        index="Participant",
        columns="Pressure",
        values="Video_Median_Velocity",
        aggfunc="mean",
    )
    pressure_stats.columns = [f"velocity_at_{float(p):.1f}psi" for p in pressure_stats.columns]
    participant_df = participant_df.join(pressure_stats)

    intercepts = {}
    mean_vel = table_df.groupby(["Participant", "Pressure"], as_index=False)["Video_Median_Velocity"].mean()
    for pid, sub in mean_vel.groupby("Participant"):
        sub = sub.dropna(subset=["Pressure", "Video_Median_Velocity"])
        if len(sub) < 2:
            intercepts[pid] = np.nan
        else:
            _, intercept = np.polyfit(sub["Pressure"], sub["Video_Median_Velocity"], 1)
            intercepts[pid] = intercept
    participant_df["velocity_at_0.0psi"] = pd.Series(intercepts)

    height_m = participant_df["Height"] * 0.0254
    weight_kg = participant_df["Weight"] * 0.453592
    participant_df["BMI"] = weight_kg / (height_m ** 2)
    return participant_df.reset_index()


def latex_table1(participant_df: pd.DataFrame) -> str:
    groups = [
        ("set01", "Control"),
        ("set02", "Hypertension"),
        ("set03", "Diabetes"),
        ("Overall", "Overall"),
    ]

    def sub_for(group):
        return participant_df if group == "Overall" else participant_df[participant_df["SET"].eq(group)]

    def cells_for(var: str, decimals: int = 1) -> list[str]:
        return [median_iqr(sub_for(group)[var], decimals) for group, _ in groups]

    def cat_cells(var: str, value) -> list[str]:
        cells = []
        for group, _ in groups:
            sub = sub_for(group)
            n = int((sub[var] == value).sum())
            pct = int(round((n / len(sub) * 100))) if len(sub) else 0
            cells.append(f"{n} ({pct} \\%)")
        return cells

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small                               % drop to \footnotesize if space is tight",
        r"\setlength{\tabcolsep}{4pt}          % default 6 pt -> 4 pt",
        "",
        r"\caption{Participant demographics and flow-velocity measures",
        r"         (median; interquartile range in parentheses).}",
        "",
        r"\begin{tabularx}{\textwidth}{@{}l*{4}{>{\centering\arraybackslash}X}@{}}",
        r"\toprule",
        r"              & \textbf{Control} & \textbf{Hypertension} & \textbf{Diabetes} & \textbf{Overall} \\",
        r"\midrule",
        "N (participants)                     & "
        + " & ".join(str(len(sub_for(group))) for group, _ in groups)
        + r" \\",
        "",
        "Age, yr                               & " + " & ".join(cells_for("Age")) + r" \\",
        "Height, in                            & " + " & ".join(cells_for("Height")) + r" \\",
        "Weight, lb                            & " + " & ".join(cells_for("Weight")) + r" \\",
        r"BMI, kg m\(^{-2}\)                    & " + " & ".join(cells_for("BMI")) + r" \\",
        "",
        r"Sys.\ BP, mmHg                        & " + " & ".join(cells_for("SYS_BP")) + r" \\",
        r"Dia.\ BP, mmHg                        & " + " & ".join(cells_for("DIA_BP")) + r" \\",
        r"\bottomrule",
    ]

    for pressure, label in [
        ("0.0", r"Vel.\@ 0 psi, um s\(^{-1}\)$\dagger$          "),
        ("0.2", r"Vel.\@ 0.2 psi                       "),
        ("0.4", r"Vel.\@ 0.4 psi                       "),
        ("0.6", r"Vel.\@ 0.6 psi                       "),
        ("0.8", r"Vel.\@ 0.8 psi                       "),
        ("1.0", r"Vel.\@ 1.0 psi                       "),
        ("1.2", r"Vel.\@ 1.2 psi                       "),
    ]:
        col = f"velocity_at_{pressure}psi"
        lines.append(label + " & " + " & ".join(cells_for(col, decimals=0)) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"Male, n (\%)                          & " + " & ".join(cat_cells("Sex", "M")) + r" \\",
        r"Female, n (\%)                        & " + " & ".join(cat_cells("Sex", "F")) + r" \\",
        r"Hypertension, n (\%)                 & " + " & ".join(cat_cells("Hypertension", True)) + r" \\",
        r"Diabetes, n (\%)                     & " + " & ".join(cat_cells("Diabetes", True)) + r" \\",
        "",
        r"\bottomrule",
        r"% \multicolumn{5}{@{}l}{\footnotesize One participant excluded for poor video quality (ID part24).}",
        r"\multicolumn{5}{@{}l}{\footnotesize Height, weight, and BMI summarize available values; one hypertension participant is missing height and weight.}\\",
        r"\multicolumn{5}{@{}l}{\footnotesize $\dagger$Calculated, not measured directly}",
        "",
        r"\end{tabularx}",
        r"\label{tab:set_comparison}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def load_cri() -> pd.DataFrame:
    df = pd.read_csv(CRI_FP)
    df["Diabetes"] = as_bool(df["Diabetes"])
    df["Hypertension"] = as_bool(df["Hypertension"])
    for col in ["Age", "SYS_BP", "DIA_BP", "CRI", "SI_logvel_averaged_02_12"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def comparison_rows(cri: pd.DataFrame) -> list[dict]:
    rows = []

    comparisons = [
        ("Diabetes vs healthy controls", cri[cri["SET"].eq("set01")], cri[cri["SET"].eq("set03")], "Control", "Diabetes"),
        ("Hypertension vs healthy controls", cri[cri["SET"].eq("set01")], cri[cri["SET"].eq("set02")], "Control", "Hypertension"),
    ]
    for name, a, b, a_label, b_label in comparisons:
        for var in ["SYS_BP", "DIA_BP", "CRI"]:
            rows.append({
                "analysis": name,
                "variable": var,
                "group1": a_label,
                "n1": len(a),
                "median_iqr1": median_iqr(a[var], 3 if var == "CRI" else 1),
                "group2": b_label,
                "n2": len(b),
                "median_iqr2": median_iqr(b[var], 3 if var == "CRI" else 1),
                "mann_whitney_p": p_mwu(a[var], b[var]),
            })

    cd = cri[cri["SET"].isin(["set01", "set03"])].copy()
    cd["DiabetesGroup"] = cd["SET"].eq("set03").astype(int)
    model = smf.ols("CRI ~ DiabetesGroup + Age", data=cd.dropna(subset=["CRI", "Age"])).fit()
    rows.append({
        "analysis": "Diabetes vs healthy controls age-adjusted OLS",
        "variable": "CRI",
        "group1": "DiabetesGroup coefficient",
        "n1": int(model.nobs),
        "median_iqr1": f"{model.params['DiabetesGroup']:.4f}",
        "group2": "p-value",
        "n2": "",
        "median_iqr2": f"{model.pvalues['DiabetesGroup']:.6g}",
        "mann_whitney_p": np.nan,
    })

    older = cd[cd["Age"] >= 54]
    rows.append({
        "analysis": "Diabetes vs healthy controls age >= 54",
        "variable": "CRI",
        "group1": "Control",
        "n1": int((older["SET"] == "set01").sum()),
        "median_iqr1": median_iqr(older.loc[older["SET"].eq("set01"), "CRI"], 3),
        "group2": "Diabetes",
        "n2": int((older["SET"] == "set03").sum()),
        "median_iqr2": median_iqr(older.loc[older["SET"].eq("set03"), "CRI"], 3),
        "mann_whitney_p": p_mwu(older.loc[older["SET"].eq("set01"), "CRI"], older.loc[older["SET"].eq("set03"), "CRI"]),
    })

    return rows


def age_control_rows(cri: pd.DataFrame) -> list[dict]:
    controls = cri[cri["SET"].eq("set01")].copy()
    rows = []
    for threshold in [30, 60]:
        low = controls[controls["Age"] < threshold]
        high = controls[controls["Age"] >= threshold]
        for var in ["SYS_BP", "CRI"]:
            rows.append({
                "threshold": threshold,
                "variable": var,
                "low_n": len(low),
                "low_median_iqr": median_iqr(low[var], 3 if var == "CRI" else 1),
                "high_n": len(high),
                "high_median_iqr": median_iqr(high[var], 3 if var == "CRI" else 1),
                "mann_whitney_p": p_mwu(low[var], high[var]),
            })
    model_df = controls.dropna(subset=["CRI", "Age", "Sex", "SYS_BP"]).copy()
    anova_model = smf.ols("CRI ~ Age + C(Sex) + SYS_BP", data=model_df).fit()
    anova = sm.stats.anova_lm(anova_model, typ=2)
    anova.to_csv(OUT_DIR / "corrected_control_cri_anova_age_sex_sbp.csv")
    return rows


def ks_tables(video_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pressure_df = video_df[video_df["Pressure"].between(0.2, 1.2)].copy()
    tests = [
        ("Age29", pressure_df[pressure_df["SET"].eq("set01")], lambda d: d["Age"] <= 29, lambda d: d["Age"] > 29),
        ("Age59", pressure_df[pressure_df["SET"].eq("set01")], lambda d: d["Age"] <= 59, lambda d: d["Age"] > 59),
        ("Sex", pressure_df[pressure_df["SET"].eq("set01")], lambda d: d["Sex"].eq("M"), lambda d: d["Sex"].eq("F")),
        ("SYS_BP", pressure_df[pressure_df["SET"].eq("set01")], lambda d: d["SYS_BP"] < 120, lambda d: d["SYS_BP"] >= 120),
        ("Diabetes", pressure_df[pressure_df["SET"].isin(["set01", "set03"])], lambda d: d["SET"].eq("set01"), lambda d: d["SET"].eq("set03")),
        ("Hypertension", pressure_df[pressure_df["SET"].isin(["set01", "set02"])], lambda d: d["SET"].eq("set01"), lambda d: d["SET"].eq("set02")),
    ]
    for name, df, left_fn, right_fn in tests:
        for pressure in sorted(df["Pressure"].dropna().unique()):
            sub = df[df["Pressure"].eq(pressure)]
            left = sub.loc[left_fn(sub), "Video_Median_Velocity"].dropna()
            right = sub.loc[right_fn(sub), "Video_Median_Velocity"].dropna()
            if len(left) and len(right):
                stat, p = ks_2samp(left, right)
                rows.append({
                    "test": name,
                    "pressure": pressure,
                    "ks": stat,
                    "p_value": p,
                    "n_left": len(left),
                    "n_right": len(right),
                })
    return pd.DataFrame(rows)


def main() -> None:
    video_df = load_video_df()
    participant_df = build_participant_df(video_df)
    table_latex = latex_table1(participant_df)
    TABLE1_FP.write_text(table_latex, encoding="utf-8")
    (OUT_DIR / "table1_median_iqr_latex_part30_corrected.txt").write_text(table_latex, encoding="utf-8")

    cri = load_cri()
    pd.DataFrame(comparison_rows(cri)).to_csv(OUT_DIR / "corrected_condition_comparisons.csv", index=False)
    pd.DataFrame(age_control_rows(cri)).to_csv(OUT_DIR / "corrected_control_age_cri_bp.csv", index=False)
    ks_tables(video_df).to_csv(OUT_DIR / "corrected_ks_tables.csv", index=False)

    counts = {
        "table1_set_counts": participant_df["SET"].value_counts(dropna=False).to_dict(),
        "cri_set_counts": cri["SET"].value_counts(dropna=False).to_dict(),
        "table1_sex_by_set": participant_df.groupby("SET")["Sex"].value_counts().to_dict(),
    }
    (OUT_DIR / "corrected_counts.txt").write_text(str(counts), encoding="utf-8")
    print(f"Wrote corrected manuscript number report to {OUT_DIR}")


if __name__ == "__main__":
    main()
