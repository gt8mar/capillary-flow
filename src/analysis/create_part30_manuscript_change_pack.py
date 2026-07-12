"""Create manuscript change pack after the part30 diabetes correction."""

from __future__ import annotations

from html import escape
from pathlib import Path
import re

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DRAFT_DIR = REPO_ROOT / "docs" / "current_draft"
OUT_DIR = DRAFT_DIR / "part30_correction_updates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_TEX = DRAFT_DIR / "atvb_2026_simple.tex"
UPDATED_TEX = DRAFT_DIR / "atvb_2026_simple_part30_corrected_numbers.tex"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def table1() -> str:
    table = read(DRAFT_DIR / "table1_median_iqr_latex.txt")
    grouping_note = (
        r"\multicolumn{5}{@{}l}{\footnotesize Participants with diabetes were analyzed in the diabetes cohort "
        r"regardless of comorbid hypertension; the hypertension cohort contains participants with hypertension "
        r"without diabetes.}\\"
    )
    if grouping_note not in table:
        table = table.replace(
            r"\multicolumn{5}{@{}l}{\footnotesize Height, weight, and BMI summarize available values; one hypertension participant is missing height and weight.}\\",
            grouping_note
            + "\n"
            + r"\multicolumn{5}{@{}l}{\footnotesize Height, weight, and BMI summarize available values; one hypertension participant is missing height and weight.}\\",
        )
    return table


def anova_rows() -> list[dict[str, str]]:
    anova = pd.read_csv(
        REPO_ROOT
        / "results"
        / "part30_diabetes_correction"
        / "corrected_control_cri_anova_age_sex_sbp.csv",
        index_col=0,
    )
    factor_names = {
        "C(Sex)": "C(Sex)",
        "Age": "Age",
        "SYS_BP": "Systolic BP",
        "Residual": "Residual",
    }
    rows = []
    for factor in ["C(Sex)", "Age", "SYS_BP", "Residual"]:
        row = anova.loc[factor]
        rows.append(
            {
                "factor": factor_names[factor],
                "sum_sq": f"{row['sum_sq']:.3f}",
                "df": f"{int(row['df'])}",
                "f_value": "-"
                if pd.isna(row["F"])
                else f"{row['F']:.3f}",
                "p_value": "-"
                if pd.isna(row["PR(>F)"])
                else f"{row['PR(>F)']:.2e}"
                if row["PR(>F)"] < 0.001
                else f"{row['PR(>F)']:.3f}",
            }
        )
    return rows


def anova_table() -> str:
    rows = anova_rows()
    body = "\n".join(
        f"{row['factor']} & {row['sum_sq']} & {row['df']} & {row['f_value']} & {row['p_value']} \\\\"
        for row in rows
    )
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{ANOVA results showing the effects of sex, age, and systolic blood pressure on CRI.}}
\label{{tab:anova_results}}
\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\toprule
Factor & Sum of Squares & df & F-Value & P-Value \\ \midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def _ks_table(csv_path: Path, label: str, caption: str, include_sysbp: bool) -> str:
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        p = f"{row['Pressure_psi']:.1f}"
        age29 = f"{row['Age29_KS_Stat']:.3f} & {row['Age29_p_value']:.4f}"
        age59 = f"{row['Age_KS_Stat']:.3f} & {row['Age_p_value']:.4f}"
        if include_sysbp:
            extra1 = f"{row['Sex_KS_Stat']:.3f} & {row['Sex_p_value']:.4f}"
            extra2 = f"{row['SYS_BP_KS_Stat']:.3f} & {row['SYS_BP_p_value']:.4f}"
        else:
            extra1 = f"{row['Diabetes_KS_Stat']:.3f} & {row['Diabetes_p_value']:.4f}"
            extra2 = f"{row['Hypertension_KS_Stat']:.3f} & {row['Hypertension_p_value']:.4f}"
        rows.append(f"{p} & {age29} & {age59} & {extra1} & {extra2} \\\\")

    if include_sysbp:
        header = r"Pressure & \multicolumn{2}{c|}{Age ($\leq$29)} & \multicolumn{2}{c|}{Age ($\leq$59)} & \multicolumn{2}{c|}{Sex} & \multicolumn{2}{c|}{SYS\_BP} \\"
    else:
        header = r"Pressure & \multicolumn{2}{c|}{Age ($\leq$29)} & \multicolumn{2}{c|}{Age ($\leq$59)} & \multicolumn{2}{c|}{Diabetes} & \multicolumn{2}{c|}{Hypertension} \\"

    return "\n".join([
        r"\begin{table}[h!]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{|c|cc|cc|cc|cc|}",
        r"\hline",
        header,
        r"(psi) & KS & p-value & KS & p-value & KS & p-value & KS & p-value \\",
        r"\hline",
        *rows,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def pressure_dist_table() -> str:
    return _ks_table(
        REPO_ROOT / "results" / "ks_statistics_tables" / "Demographics_KS_Comparison_Table_Modified.csv",
        "tab:pressure_dist",
        "Kolmogorov-Smirnov (KS) test for velocity distribution dissimilarity at increasing external pressure",
        include_sysbp=True,
    )


def health_ks_table() -> str:
    return _ks_table(
        REPO_ROOT / "results" / "ks_statistics_tables" / "Health_and_Age_KS_Comparison_Table_Modified.csv",
        "tab:health_ks",
        "Kolmogorov-Smirnov (KS) test for velocity distribution dissimilarity - Health Conditions",
        include_sysbp=False,
    )


def mixed_model_table() -> str:
    return r"""\begin{sidewaystable}[htbp]
\centering
\caption{Mixed Effects Model Results for Blood Flow Analysis with Height and Weight}
\label{tab:hw_mixed_model_results}
\begin{tabular}{lrrrrr}
\toprule
Parameter & Basic Hw & Age Pressure Hw & Random Slopes Hw & Log Velocity Hw & Log Random Slopes Hw \\
\midrule
\multicolumn{6}{l}{\textbf{Fixed Effects}} \\
Age & 6.7254* (3.3290) & 5.6134 (4.2122) & 7.2358* (3.0836) & 0.0535*** (0.0126) & 0.0517*** (0.0127) \\
Age:Pressure & -- & 1.6758 (3.8929) & -- & -- & -- \\
Height & 13.3563 (21.0616) & 13.3934 (21.0500) & 15.1090 (19.4836) & 0.0292 (0.0795) & 0.0191 (0.0801) \\
Intercept & -909.9125 (1319.3931) & -864.4679 (1322.8774) & -674.6533 (1226.4984) & -0.0730 (4.9806) & 0.2730 (4.9715) \\
Pressure & -600.4826*** (80.4276) & -675.3906*** (191.7199) & -621.5047*** (150.6260) & -2.4874*** (0.2096) & -2.6052*** (0.3984) \\
Weight & 4.6748 (2.6945) & 4.6835 (2.6931) & 2.2278 (2.7703) & 0.0118 (0.0102) & 0.0150 (0.0108) \\
\midrule
\multicolumn{6}{l}{\textbf{Standardized Effects: Log Random Slopes Model}} \\
Age & \multicolumn{2}{r}{Standardized $\beta$ = 0.3822} & \multicolumn{3}{r}{Range effect = 3.3077} \\
Pressure & \multicolumn{2}{r}{Standardized $\beta$ = -0.3096} & \multicolumn{3}{r}{Range effect = -2.6052} \\
Height & \multicolumn{2}{r}{Standardized $\beta$ = 0.0240} & \multicolumn{3}{r}{Range effect = 0.2478} \\
Weight & \multicolumn{2}{r}{Standardized $\beta$ = 0.1428} & \multicolumn{3}{r}{Range effect = 2.0041} \\
\midrule
\multicolumn{6}{l}{\textbf{Random Effects (SD)}} \\
RE1 & 363.9364 & 363.6618 & 752.8351 & 1.4379 & 1.7965 \\
RE2 & -- & -- & 753.6941 & -- & 2.0050 \\
Residual SD & 778.8128 & 779.2129 & 739.0172 & 2.0287 & 1.9196 \\
\midrule
\multicolumn{6}{l}{\textbf{Model Statistics}} \\
Log-Likelihood & -6964.84 & -6962.47 & -6941.27 & -1882.43 & -1859.31 \\
\bottomrule
\end{tabular}
\end{sidewaystable}
"""


def _load_demo_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "results" / "demographics" / name, index_col=0)


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().eq("TRUE")


def _participant_demo_df() -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / "summary_df_nhp_video_stats.csv")
    hw_path = REPO_ROOT / "height_weight.csv"
    if hw_path.exists():
        hw = pd.read_csv(hw_path)
        df = df.merge(hw, on="Participant", how="outer", suffixes=("_x", "_y"))
        for col in ["Height", "Weight"]:
            x_col = f"{col}_x"
            y_col = f"{col}_y"
            if x_col in df.columns and y_col in df.columns:
                df[col] = df[x_col].combine_first(df[y_col])
                df = df.drop(columns=[x_col, y_col])

    df = df.dropna(subset=["SET"]).copy()
    for col in ["Age", "Height", "Weight", "SYS_BP", "DIA_BP", "Pressure", "Video_Median_Velocity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Hypertension"] = _as_bool(df["Hypertension"])
    df["Diabetes"] = _as_bool(df["Diabetes"])

    participant_df = (
        df.sort_values(["Participant", "Date", "Location", "Video"])
        .drop_duplicates("Participant")
        .set_index("Participant")
    )
    pressure_stats = df.pivot_table(
        index="Participant",
        columns="Pressure",
        values="Video_Median_Velocity",
        aggfunc="mean",
    )
    pressure_stats.columns = [f"velocity_at_{float(p):.1f}psi" for p in pressure_stats.columns]
    participant_df = participant_df.join(pressure_stats)

    intercepts = {}
    mean_vel = df.groupby(["Participant", "Pressure"], as_index=False)["Video_Median_Velocity"].mean()
    for participant, sub in mean_vel.groupby("Participant"):
        sub = sub.dropna(subset=["Pressure", "Video_Median_Velocity"])
        if len(sub) < 2:
            intercepts[participant] = pd.NA
        else:
            _, intercept = np.polyfit(sub["Pressure"], sub["Video_Median_Velocity"], 1)
            intercepts[participant] = intercept
    participant_df["velocity_at_0.0psi"] = pd.Series(intercepts)

    height_m = participant_df["Height"] * 0.0254
    weight_kg = participant_df["Weight"] * 0.453592
    participant_df["BMI"] = weight_kg / (height_m ** 2)
    return participant_df.reset_index()


def _mean_sd_range(values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return "--"
    return f"{clean.mean():.1f} $\\pm$ {clean.std():.1f} ({clean.min():.1f}--{clean.max():.1f})"


def _cat_count(sub: pd.DataFrame, var: str, value) -> str:
    if sub.empty:
        return "--"
    n = int((sub[var] == value).sum())
    pct = int(round(n / len(sub) * 100))
    return f"{n} ({pct}\\%)"


def _demo_table(name: str, label: str, caption: str, columns: list[str], headings: list[str]) -> str:
    df = _load_demo_csv(name)
    df = df.loc[~df.index.to_series().astype(str).str.startswith("Note:")]
    df = df.replace({"Â±": r"$\pm$"}, regex=True)
    col_spec = "@{}l" + "*{" + str(len(columns)) + r"}{>{\centering\arraybackslash}X}@{}"
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        rf"\begin{{tabularx}}{{\textwidth}}{{{col_spec}}}",
        r"\toprule",
        "              & " + " & ".join(rf"\textbf{{{h}}}" for h in headings) + r" \\",
        r"\midrule",
    ]
    row_map = {
        "N": "N (participants)",
        "Age (years)": "Age, yr",
        "Height (inches)": "Height, in",
        "Weight (lbs)": "Weight, lb",
        "BMI (kg/m^2)": r"BMI, kg m\(^{-2}\)",
        "SYS_BP (mmHg)": r"Sys.\ BP, mmHg",
        "DIA_BP (mmHg)": r"Dia.\ BP, mmHg",
        "velocity_at_0psi (calculated, mm/s)": r"Vel.\@ 0 psi, um s\(^{-1}\)$\dagger$",
        "velocity_at_0.2psi (mm/s)": r"Vel.\@ 0.2 psi",
        "velocity_at_0.4psi (mm/s)": r"Vel.\@ 0.4 psi",
        "velocity_at_0.6psi (mm/s)": r"Vel.\@ 0.6 psi",
        "velocity_at_0.8psi (mm/s)": r"Vel.\@ 0.8 psi",
        "velocity_at_1.0psi (mm/s)": r"Vel.\@ 1.0 psi",
        "velocity_at_1.2psi (mm/s)": r"Vel.\@ 1.2 psi",
        "Male": r"Male, n (\%)",
        "Female": r"Female, n (\%)",
        "Hypertension = True": r"Hypertension, n (\%)",
        "Diabetes = True": r"Diabetes, n (\%)",
    }
    for idx, display in row_map.items():
        if idx not in df.index:
            continue
        if idx in ["velocity_at_0psi (calculated, mm/s)", "Male"]:
            lines.append(r"\bottomrule")
        values = [str(df.loc[idx, col]).replace("%", r"\%").replace("â€“", "--").replace("Â±", r"$\pm$") for col in columns]
        lines.append(display + " & " + " & ".join(values) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\multicolumn{" + str(len(columns) + 1) + r"}{@{}l}{\footnotesize $\dagger$Calculated, not measured directly}",
        r"\end{tabularx}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def _cohort_demo_table(label: str, caption: str, groups: list[tuple[str, pd.DataFrame]]) -> str:
    col_spec = "@{}l" + "*{" + str(len(groups)) + r"}{>{\centering\arraybackslash}X}@{}"
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        rf"\begin{{tabularx}}{{\textwidth}}{{{col_spec}}}",
        r"\toprule",
        "              & " + " & ".join(rf"\textbf{{{name}}}" for name, _ in groups) + r" \\",
        r"\midrule",
    ]
    rows = [
        ("N (participants)", lambda sub: str(len(sub))),
        ("Age, yr", lambda sub: _mean_sd_range(sub["Age"])),
        ("Height, in", lambda sub: _mean_sd_range(sub["Height"])),
        ("Weight, lb", lambda sub: _mean_sd_range(sub["Weight"])),
        (r"BMI, kg m\(^{-2}\)", lambda sub: _mean_sd_range(sub["BMI"])),
        (r"Sys.\ BP, mmHg", lambda sub: _mean_sd_range(sub["SYS_BP"])),
        (r"Dia.\ BP, mmHg", lambda sub: _mean_sd_range(sub["DIA_BP"])),
        (r"Vel.\@ 0 psi, um s\(^{-1}\)$\dagger$", lambda sub: _mean_sd_range(sub["velocity_at_0.0psi"])),
        (r"Vel.\@ 0.2 psi", lambda sub: _mean_sd_range(sub["velocity_at_0.2psi"])),
        (r"Vel.\@ 0.4 psi", lambda sub: _mean_sd_range(sub["velocity_at_0.4psi"])),
        (r"Vel.\@ 0.6 psi", lambda sub: _mean_sd_range(sub["velocity_at_0.6psi"])),
        (r"Vel.\@ 0.8 psi", lambda sub: _mean_sd_range(sub["velocity_at_0.8psi"])),
        (r"Vel.\@ 1.0 psi", lambda sub: _mean_sd_range(sub["velocity_at_1.0psi"])),
        (r"Vel.\@ 1.2 psi", lambda sub: _mean_sd_range(sub["velocity_at_1.2psi"])),
        (r"Male, n (\%)", lambda sub: _cat_count(sub, "Sex", "M")),
        (r"Female, n (\%)", lambda sub: _cat_count(sub, "Sex", "F")),
        (r"Hypertension, n (\%)", lambda sub: _cat_count(sub, "Hypertension", True)),
        (r"Diabetes, n (\%)", lambda sub: _cat_count(sub, "Diabetes", True)),
    ]
    for display, getter in rows:
        if display in [r"Vel.\@ 0 psi, um s\(^{-1}\)$\dagger$", r"Male, n (\%)"]:
            lines.append(r"\bottomrule")
        lines.append(display + " & " + " & ".join(getter(sub) for _, sub in groups) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\multicolumn{" + str(len(groups) + 1) + r"}{@{}l}{\footnotesize This table summarizes the non-diabetic hypertension cohort; diabetes participants with comorbid hypertension are summarized in the diabetes cohort.}\\",
        r"\multicolumn{" + str(len(groups) + 1) + r"}{@{}l}{\footnotesize Height, weight, and BMI summarize available values; one hypertension participant is missing height and weight.}\\",
        r"\multicolumn{" + str(len(groups) + 1) + r"}{@{}l}{\footnotesize $\dagger$Calculated, not measured directly}",
        r"\end{tabularx}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def diabetes_demo_table() -> str:
    return _demo_table(
        "demographics_by_diabetes.csv",
        "tab:diabetes_demographics",
        "Participants with diabetes demographics and flow-velocity measures (mean $\\pm$ SD; range in parentheses).",
        ["False", "True", "Overall"],
        ["No Diabetes", "Diabetes", "Overall"],
    )


def hypertension_demo_table() -> str:
    participant_df = _participant_demo_df()
    control = participant_df[participant_df["SET"].eq("set01")].copy()
    hypertension = participant_df[participant_df["SET"].eq("set02")].copy()
    overall = participant_df[participant_df["SET"].isin(["set01", "set02"])].copy()
    return _cohort_demo_table(
        "tab:hypertension_demographics",
        "Healthy control and hypertension cohort demographics and flow-velocity measures (mean $\\pm$ SD; range in parentheses).",
        [
            ("Control", control),
            ("Hypertension", hypertension),
            ("Overall", overall),
        ],
    )


def replace_table(tex: str, label: str, new_table: str) -> str:
    label_token = r"\label{" + label + "}"
    label_idx = tex.find(label_token)
    if label_idx == -1:
        raise RuntimeError(f"Could not find table label {label}")

    table_begin = tex.rfind(r"\begin{table}", 0, label_idx)
    sidewaystable_begin = tex.rfind(r"\begin{sidewaystable}", 0, label_idx)
    if table_begin == -1 and sidewaystable_begin == -1:
        raise RuntimeError(f"Could not find table start for {label}")

    if sidewaystable_begin > table_begin:
        begin_idx = sidewaystable_begin
        end_token = r"\end{sidewaystable}"
    else:
        begin_idx = table_begin
        end_token = r"\end{table}"

    end_idx = tex.find(end_token, label_idx)
    if end_idx == -1:
        raise RuntimeError(f"Could not find table end for {label}")
    end_idx += len(end_token)
    return tex[:begin_idx] + new_table.strip() + tex[end_idx:]


def add_todo_before(text: str, needle: str, todo: str) -> str:
    idx = text.find(needle)
    if idx == -1:
        return text
    return text[:idx] + f"% TODO(part30): {todo}\n" + text[idx:]


def update_tex() -> str:
    tex = read(ORIGINAL_TEX)
    for label, table in [
        ("tab:set_comparison", table1()),
        ("tab:anova_results", anova_table()),
        ("tab:pressure_dist", pressure_dist_table()),
        ("tab:health_ks", health_ks_table()),
        ("tab:diabetes_demographics", diabetes_demo_table()),
        ("tab:hypertension_demographics", hypertension_demo_table()),
        ("tab:hw_mixed_model_results", mixed_model_table()),
    ]:
        tex = replace_table(tex, label, table)

    replacements = {
        "relative to healthy controls (n=35), diabetic subjects (n=17)": "relative to healthy controls (n=34), diabetic subjects (n=18)",
        "diabetic subjects (n=18) with pharmacologically well-controlled blood pressure had significantly increased CRI, even when adjusted for age.": "diabetic subjects (n=18) had significantly increased unadjusted CRI, but this association was not significant after age adjustment.",
        "subjects with pharmacologically well-controlled hypertension (n=19)": "non-diabetic subjects with pharmacologically well-controlled hypertension (n=20)",
        "healthy (n=35; 17 male, 18 female), diabetic (n=17), and hypertensive (n=19)": "healthy (n=34; 16 male, 18 female), diabetic (n=18), and non-diabetic hypertensive (n=20)",
        "We enrolled 72 participants, including participants with diabetes (n=17), hypertension (n=20), and healthy controls (n=35) (Fig. \\ref{overview}). One participant with hypertension was excluded because of poor video quality": "We analyzed 72 participants, including participants with diabetes (n=18), non-diabetic hypertension cohort participants (n=20), and healthy controls (n=34), after excluding one participant with hypertension because of poor video quality (Fig. \\ref{overview}). Participants with both diabetes and hypertension (n=7) were assigned to the diabetes group for disease-group analyses; the hypertension cohort therefore refers to participants with hypertension without diabetes",
        "Systolic blood pressure measurements (p=0.848)": "Systolic blood pressure measurements (p=0.759)",
        "whose CRI was 5.193 (IQR: 4.441-5.801)": "whose CRI was 5.346 (IQR: 4.651-5.842)",
        "(p=0.0165) lower CRI": "(p=0.0120) lower CRI",
        "participants sixty or above (5.127 IQR: 4.919-5.703)": "participants sixty or above (5.313 IQR: 4.985-5.793)",
        "blood pressure (p=0.915": "blood pressure (p=0.970",
        "we conducted a two-way ANOVA analysis (Table \\ref{tab:anova_results}) with interactions to analyze the effects of age, systolic blood pressure, and sex on  CRI": "we used an ANOVA model (Table \\ref{tab:anova_results}) to analyze the effects of age, systolic blood pressure, and sex on CRI",
        "f=18.35, p=0.0001": "F=21.664, p=6.17e-05",
        "blood pressure (f=0.062, p=0.804) and sex (f=0.020, p=0.888)": "sex (F=0.023, p=0.880) and systolic blood pressure (F=0.050, p=0.824)",
        "weight (effect size 0.148, p=0.1548)": "weight (standardized effect size 0.143, p=0.165)",
        "height (effect size 0.043, p=0.6755)": "height (standardized effect size 0.024, p=0.812)",
        "slopes of 0.0480 and –2.5009": "slopes of 0.0517 and -2.6052",
        "participants with diabetes (n=17) to our control group (n=35)": "participants with diabetes (n=18) to our control group (n=34)",
        "revealed significantly elevated velocities at all pressures, even when accounting for age": "revealed significantly elevated velocities at most pressures",
        "Blood pressure was close to significantly (p=0.0485) distinguishing": "Blood pressure was significantly higher (p=0.0485) in",
        "Blood pressure was close to significantly (p=0.0545) distinguishing": "Blood pressure was significantly higher (p=0.0485) in",
        "Blood pressure was significantly higher (p=0.0485) in our diabetic participants": "Blood pressure was significantly higher (p=0.0485) in diabetic participants",
        "mmHg) from controls": "mmHg) than controls",
        "(p=0.0545)": "(p=0.0485)",
        "median 129.0, IQR: 111.0-142.0": "median 128.5, IQR: 113.5-141.2",
        "median 114.0, IQR: 107.5-128.0": "median 113.5, IQR: 107.2-128.5",
        "(p=0.0112)": "(p=0.0205)",
        "median 5.982, IQR: 4.769-6.415": "median 5.941, IQR: 4.630-6.372",
        "median 4.574, IQR: 3.198-5.529": "median 4.729, IQR: 3.183-5.531",
        "p=0.0367": "p=0.211",
        "Diabetes remained significantly associated with higher CRI after adjustment for age (p=0.211)": "Diabetes was not significantly associated with higher CRI after adjustment for age (p=0.211)",
        "p=0.4669": "p=0.740",
        "participants with hypertension (n=19) to our control group (n=35)": "participants in the non-diabetic hypertension cohort (n=20) to our control group (n=34)",
        "(p=0.0115)": "(p=0.00841)",
        "median 127.0 mmHg, IQR: 118.0-138.5": "median 128.5 mmHg, IQR: 120.0-137.8",
        "(p=0.704)": "(p=0.320)",
        "median 5.365 IQR: 3.215-5.996": "median 5.322 IQR: 3.011-5.973",
        "median 114.0 mmHg, IQR: 107.5-128.0": "median 113.5 mmHg, IQR: 107.2-128.5",
        "the hypertensive group had significantly higher": "the non-diabetic hypertension cohort had significantly higher",
        "between the hypertensive group": "between the non-diabetic hypertension cohort",
        "Velocity curves for participants with and without Hypertension.": "Velocity curves for the non-diabetic hypertension cohort and controls.",
        "participants with pharmacologically-managed hypertension versus controls": "the non-diabetic hypertension cohort versus controls",
        "whereas participants with treated hypertension did not differ significantly from age-matched controls": "whereas the non-diabetic treated hypertension cohort did not differ significantly from age-matched controls",
        "treated hypertensive participants would be expected to show similar significant differences": "the non-diabetic treated hypertension cohort would be expected to show similar significant differences",
        r"Fig. \ref{fig:classifier} d": r"Fig. \ref{fig:condition} d",
        "blood-pressure controlled participants with diabetes": "participants with diabetes",
        "The cohort was limited in size (n=71)": "The cohort was limited in size (n=72)",
        "not captured by measuring blood pressure": "not fully captured by measuring blood pressure alone",
        "not explained by measured blood pressure": "not fully explained by measured blood pressure alone",
        "with insignificant KS-Tests": "with mostly non-significant KS-Tests",
        "with mostly insignificant KS-Tests": "with mostly non-significant KS-Tests",
        "showed no difference in velocities after controlling for age": "showed little consistent difference in velocities",
        "suggesting that hypertension medicine successfully brings microvascular function back to an age-appropriate baseline": "suggesting that treated hypertension was not associated with a detectable CRI increase relative to controls",
        "Participants with diabetes maintained higher velocities under external pressure and had significantly increased CRI after age adjustment": "Participants with diabetes maintained higher velocities under external pressure and had significantly increased unadjusted CRI, although the age-adjusted association was not significant after the part30 correction",
        "modest cohort size (n=17)": "modest cohort size (n=18)",
    }
    for old, new in replacements.items():
        tex = tex.replace(old, new)

    tex = add_todo_before(
        tex,
        "Diabetes remained significantly associated with higher CRI after adjustment for age",
        "Age-adjusted diabetes association is no longer significant after part30 correction; revise interpretation rather than leaving this sentence as-is.",
    )
    tex = add_todo_before(
        tex,
        r"\abstract{",
        "Abstract revised numerically, but the diabetes conclusion should be softened because age-adjusted CRI is no longer significant and classifier metrics were not rerun.",
    )
    tex = add_todo_before(
        tex,
        "We enrolled 72 participants",
        "Cohort accounting needs one final wording pass: current analysis data contain 72 analyzed participants after excluding part24, with 34 controls, 20 hypertension, and 18 diabetes.",
    )
    tex = add_todo_before(
        tex,
        "Together, these results suggest that diabetes is associated with residual microvascular dysfunction",
        "Diabetes CRI remains significant unadjusted, but age-adjusted OLS is not significant; soften this conclusion or frame as unadjusted/preliminary.",
    )
    tex = add_todo_before(
        tex,
        "This contrast suggests that pressure-modulated capillaroscopy may be sensitive to residual microvascular remodeling in diabetes",
        "This discussion claim should be softened; corrected data support unadjusted diabetes differences but not a significant age-adjusted CRI association.",
    )
    tex = add_todo_before(
        tex,
        "These features were strongly associated with age and diabetes but not with measured blood pressure or sex",
        "This sentence overstates the corrected diabetes result; diabetes CRI is unadjusted-significant but not age-adjusted-significant, and diabetes SBP differs from controls.",
    )
    tex = add_todo_before(
        tex,
        "See table below (Table \\ref{tab:age_only_classification})",
        "Classifier supplemental tables and AUC should be regenerated after part30 correction before relying on these numbers.",
    )
    tex = add_todo_before(
        tex,
        "Together, these findings establish a proof of principle for a sensitive and non-invasive method",
        "The preceding independence claim should distinguish age effects from diabetes effects; diabetes is not cleanly independent of BP/age after correction.",
    )
    tex = add_todo_before(
        tex,
        "The classifier results reinforce this interpretation",
        "Full classifier rerun timed out in this session; keep the old AUC only if you rerun/confirm classifier metrics after part30 correction.",
    )
    tex = add_todo_before(
        tex,
        "suggesting that hypertension medicine successfully brings microvascular function back",
        "This causal medication interpretation remains too strong; data show no CRI difference in treated/recorded hypertension group, not medication normalization.",
    )
    return tex


def markdown_report() -> str:
    condition = pd.read_csv(REPO_ROOT / "results" / "part30_diabetes_correction" / "corrected_condition_comparisons.csv")
    age = pd.read_csv(REPO_ROOT / "results" / "part30_diabetes_correction" / "corrected_control_age_cri_bp.csv")
    ks_health = pd.read_csv(REPO_ROOT / "results" / "ks_statistics_tables" / "Health_and_Age_KS_Comparison_Table_Modified.csv")
    ks_demo = pd.read_csv(REPO_ROOT / "results" / "ks_statistics_tables" / "Demographics_KS_Comparison_Table_Modified.csv")
    effect = pd.read_csv(REPO_ROOT / "results" / "mixed_model" / "height_weight" / "fixed_effect_sizes.csv")
    log_rs = effect[effect["model"].eq("log_random_slopes_hw")]

    def df_to_md(df: pd.DataFrame, columns: list[str] | None = None) -> str:
        if columns is not None:
            df = df[columns]
        headers = [str(c) for c in df.columns]
        rows = [[str(v) for v in row] for row in df.to_numpy()]
        out = ["| " + " | ".join(headers) + " |"]
        out.append("| " + " | ".join(["---"] * len(headers)) + " |")
        out.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(out)

    anova_md = "\n".join(
        [
            "| Factor | Sum Sq | df | F | p |",
            "|---|---:|---:|---:|---:|",
            *[
                (
                    f"| {row['factor']} | {row['sum_sq']} | {row['df']} | "
                    f"{row['f_value']} | {row['p_value']} |"
                )
                for row in anova_rows()
            ],
        ]
    )

    return f"""# Part30 Diabetes Correction Manuscript Change Pack

Generated from corrected derived data where `part30` is `SET=set03`, `Diabetes=True`, and `is_healthy=False`.

## Files Generated

- Corrected TeX copy: `{UPDATED_TEX.relative_to(REPO_ROOT)}`
- This Markdown report: `{(OUT_DIR / 'part30_correction_change_report.md').relative_to(REPO_ROOT)}`
- HTML report: `{(OUT_DIR / 'part30_correction_change_report.html').relative_to(REPO_ROOT)}`
- Pasteable corrected tables: `{OUT_DIR.relative_to(REPO_ROOT)}`
- Main Table 1 pasteable file also updated: `docs/current_draft/table1_median_iqr_latex.txt`

## High Priority Text Changes

- Main analysis cohort: Control `n=34`, Hypertension `n=20`, Diabetes `n=18`, Overall `n=72`.
- One hypertension participant is missing height/weight, so height, weight, and BMI summaries use available values for that row while N still remains `20`.
- The hypertension supplementary demographics table is cohort-based (`set01` controls vs `set02` hypertension), so it reports `34`, `20`, and `54`; the `27` total hypertension metadata count only appears where all comorbid diabetes participants are included.
- CRI analysis cohort: Control `n=34`, Hypertension `n=20`, Diabetes `n=18`, Overall `n=72`.
- Control sex count: `16 male, 18 female`.
- Diabetes unadjusted CRI remains significant: `p=0.0205`.
- Diabetes age-adjusted OLS is no longer significant: strict control-vs-diabetes `p=0.211`; all non-diabetes-vs-diabetes `p=0.0774`.
- The generated TeX now states that the diabetes age-adjusted CRI association is not significant; remaining discussion/abstract interpretation still needs softening.
- The classifier AUC sentence needs confirmation; the full classifier rerun timed out after 180 seconds.
- The supplemental classifier tables are still marked with a TODO because they were not regenerated after the part30 correction.

## Condition Comparisons

{df_to_md(condition)}

## Control Age Comparisons

{df_to_md(age)}

## ANOVA Table Used In Paper

Use `CRI ~ Age + C(Sex) + SYS_BP` for the paper table.

{anova_md}

## Height/Weight Effects

Log random-slope model fixed effects and standardized effects:

{df_to_md(log_rs, ['term','estimate','std_error','p_value','standardized_beta','range_effect'])}

Interpretation: raw height coefficients can look larger because height and weight use different units. Standardized beta and range effect both support weight having the larger body-size association.

## Updated Health KS Table

{df_to_md(ks_health)}

## Updated Demographic KS Table

{df_to_md(ks_demo)}

## Tables Recreated

- `table1_set_comparison_corrected.tex`
- `anova_results_corrected.tex`
- `pressure_dist_corrected.tex`
- `health_ks_corrected.tex`
- `diabetes_demographics_corrected.tex`
- `hypertension_demographics_corrected.tex`
- `height_weight_mixed_model_corrected.tex`

## Interpretation TODOs Inserted In TeX Copy

- Age-adjusted diabetes claim is no longer significant and needs softening.
- Diabetes conclusion should distinguish unadjusted CRI significance from age-adjusted uncertainty.
- Classifier AUC needs a completed rerun before retaining exact AUC text.
- Hypertension medication-normalization language is causal and should be softened.
"""


def html_report(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            lines.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("- "):
            lines.append(f"<p>&bull; {escape(line[2:])}</p>")
        elif line.startswith("|"):
            lines.append(f"<pre>{escape(line)}</pre>")
        elif line.strip() == "":
            lines.append("")
        else:
            lines.append(f"<p>{escape(line)}</p>")
    return "\n".join([
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Part30 Correction Change Report</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;line-height:1.45}pre{margin:0;font-family:Consolas,monospace;white-space:pre-wrap}.warn{background:#fff3cd;padding:8px}</style>",
        "</head><body>",
        *lines,
        "</body></html>",
    ])


def main() -> None:
    tables = {
        "table1_set_comparison_corrected.tex": table1(),
        "anova_results_corrected.tex": anova_table(),
        "pressure_dist_corrected.tex": pressure_dist_table(),
        "health_ks_corrected.tex": health_ks_table(),
        "diabetes_demographics_corrected.tex": diabetes_demo_table(),
        "hypertension_demographics_corrected.tex": hypertension_demo_table(),
        "height_weight_mixed_model_corrected.tex": mixed_model_table(),
    }
    for name, content in tables.items():
        write(OUT_DIR / name, content)

    updated = update_tex()
    write(UPDATED_TEX, updated)

    md = markdown_report()
    write(OUT_DIR / "part30_correction_change_report.md", md)
    write(OUT_DIR / "part30_correction_change_report.html", html_report(md))
    print(f"Wrote change pack to {OUT_DIR}")
    print(f"Wrote corrected TeX copy to {UPDATED_TEX}")


if __name__ == "__main__":
    main()
