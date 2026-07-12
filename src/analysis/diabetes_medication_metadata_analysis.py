"""
Extract diabetes diagnosis metadata and summarize medication-related notes.

This script intentionally drops direct identifiers from the new metadata/extra
CSV before writing any results. It maps rows to participant IDs using the
project convention specified for the extra CSV: the first data row is part09,
and subsequent rows increment by one participant number.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - script still produces descriptive output
    stats = None


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRA_PATH = REPO_ROOT / "metadata" / "extra" / "Participant listserv - Sheet1.csv"
SUMMARY_PATH = REPO_ROOT / "summary_df_nhp_video_stats.csv"
CRI_PATH = REPO_ROOT / "results" / "Stiffness" / "cri_metrics.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "diabetes_medication_metadata_analysis"

DIRECT_IDENTIFIER_PATTERNS = ("name", "email", "phone", "tel")

DIABETES_MED_PATTERNS = {
    "metformin": r"\bmetformin\b|\bmetaformin\b",
    "sulfonylurea": r"\bglipizide\b",
    "insulin": r"\binsulin\b|\bhumalog\b|\bhumaolg\b",
    "glp1_or_gip_glp1": r"\bozempic\b|\bmounjaro\b",
}

ANTIHYPERTENSIVE_PATTERNS = {
    "ace_inhibitor": r"\blisinopril\b",
    "arb": r"\blosartan\b|\bvalsartan\b",
    "calcium_channel_blocker": r"\bamlodipine\b",
    "beta_blocker": r"\bmetoprolol\b|\bmetopolol\b",
    "generic_bp_med": r"\bbp meds?\b|\bbp medication\b|\bhypertension\)",
}

OTHER_MED_PATTERNS = {
    "unspecified_medication": r"\bmeds?\b|\bmedication\b",
    "statin_or_lipid": r"\batorvastatin\b|\bstatin\b|\bcholesterol\)",
    "aspirin": r"\baspirin\b",
    "anticoagulant": r"\beliquis\b",
    "immunosuppressant": r"\bimmunosuppressant\b",
}


def _participant_id_for_extra_row(row_index: int) -> str:
    """Return participant ID for a zero-based extra CSV data row."""
    return f"part{row_index + 9:02d}"


def _drop_direct_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns whose names suggest direct identifiers."""
    keep_cols = [
        col
        for col in df.columns
        if not any(pattern in col.lower() for pattern in DIRECT_IDENTIFIER_PATTERNS)
    ]
    return df.loc[:, keep_cols].copy()


def _parse_numeric(value) -> float:
    """Extract the first numeric value from a mixed free-text cell."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else np.nan


def _parse_visit_year(value) -> float:
    """Parse YYMMDD-like visit dates into a four-digit year."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    text = text.split(".")[0]
    if len(text) >= 2 and text[:2].isdigit():
        year = int(text[:2])
        return float(2000 + year if year < 80 else 1900 + year)
    return np.nan


def _boolish(value) -> bool:
    """Convert mixed boolean/string disease flags to bool."""
    return str(value).strip().upper() in {"TRUE", "1", "TYPE 1", "TYPE 2"}


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def load_extra_diabetes_metadata() -> pd.DataFrame:
    """Load extra CSV and return a sanitized participant-level dataframe."""
    extra = pd.read_csv(EXTRA_PATH)
    extra.insert(0, "Participant", [_participant_id_for_extra_row(i) for i in range(len(extra))])
    extra = _drop_direct_identifiers(extra)

    rename_map = {
        "date": "extra_date",
        "set": "extra_set",
        "height": "extra_height",
        "weight": "extra_weight",
        "year diabetes diagnosed": "diabetes_diagnosis_year_raw",
        "age diabetes diagnosed": "diabetes_diagnosis_age_raw",
        "diabetes type": "diabetes_type_extra",
    }
    extra = extra.rename(columns={old: new for old, new in rename_map.items() if old in extra.columns})

    extra["diabetes_diagnosis_year"] = pd.to_numeric(
        extra.get("diabetes_diagnosis_year_raw"), errors="coerce"
    )
    extra.loc[
        ~extra["diabetes_diagnosis_year"].between(1900, 2026, inclusive="both"),
        "diabetes_diagnosis_year",
    ] = np.nan

    extra["diabetes_diagnosis_age"] = extra.get("diabetes_diagnosis_age_raw").apply(_parse_numeric)
    extra.loc[
        ~extra["diabetes_diagnosis_age"].between(0, 120, inclusive="both"),
        "diabetes_diagnosis_age",
    ] = np.nan

    extra["extra_visit_year"] = extra.get("extra_date").apply(_parse_visit_year)
    return extra


def load_existing_participant_metadata() -> pd.DataFrame:
    """Load existing participant-level analysis metadata."""
    df = pd.read_csv(SUMMARY_PATH)
    participant = df.sort_values(["Participant", "Date"]).drop_duplicates("Participant").copy()

    keep_cols = [
        "Participant",
        "SET",
        "Date",
        "Age",
        "SYS_BP",
        "DIA_BP",
        "Hypertension",
        "Diabetes",
        "Raynauds",
        "HeartDisease",
        "Other",
    ]
    participant = participant[[col for col in keep_cols if col in participant.columns]].copy()
    participant["existing_visit_year"] = participant["Date"].apply(_parse_visit_year)
    participant["diabetes_flag"] = participant["Diabetes"].apply(_boolish)
    participant["hypertension_flag"] = participant["Hypertension"].apply(_boolish)
    participant["heart_disease_flag"] = participant["HeartDisease"].apply(_boolish)
    return participant


def add_medication_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add medication class flags from the existing free-text Other field."""
    out = df.copy()
    out["other_text"] = out["Other"].fillna("").astype(str)

    for med_class, pattern in DIABETES_MED_PATTERNS.items():
        out[f"med_diabetes_{med_class}"] = out["other_text"].str.contains(
            pattern, case=False, regex=True, na=False
        )
    for med_class, pattern in ANTIHYPERTENSIVE_PATTERNS.items():
        out[f"med_antihypertensive_{med_class}"] = out["other_text"].str.contains(
            pattern, case=False, regex=True, na=False
        )
    for med_class, pattern in OTHER_MED_PATTERNS.items():
        out[f"med_other_{med_class}"] = out["other_text"].str.contains(
            pattern, case=False, regex=True, na=False
        )

    diabetes_med_cols = [col for col in out.columns if col.startswith("med_diabetes_")]
    antihypertensive_cols = [
        col for col in out.columns if col.startswith("med_antihypertensive_")
    ]
    other_med_cols = [col for col in out.columns if col.startswith("med_other_")]
    out["has_diabetes_med_note"] = out[diabetes_med_cols].any(axis=1)
    out["has_antihypertensive_med_note"] = out[antihypertensive_cols].any(axis=1)
    out["has_any_med_note"] = out[diabetes_med_cols + antihypertensive_cols + other_med_cols].any(axis=1)
    out["has_unspecified_med_note"] = out.get("med_other_unspecified_medication", False)
    out["has_other_nonfalse_note"] = out["Other"].notna() & ~out["Other"].astype(str).str.upper().eq("FALSE")
    return out


def add_cri(df: pd.DataFrame) -> pd.DataFrame:
    """Merge participant-level CRI if available."""
    if not CRI_PATH.exists():
        df["CRI"] = np.nan
        return df
    cri = pd.read_csv(CRI_PATH)
    if "CRI" not in cri.columns:
        df["CRI"] = np.nan
        return df
    return df.merge(cri[["Participant", "CRI"]].drop_duplicates("Participant"), on="Participant", how="left")


def existing_diagnosis_metadata_scan() -> pd.DataFrame:
    """Scan existing metadata files for diagnosis/date/duration columns."""
    files = [
        SUMMARY_PATH,
        REPO_ROOT / "summary_df_nhp_video_medians.csv",
        REPO_ROOT / "metadata" / "merged" / "merged_metadata.csv",
        REPO_ROOT / "metadata" / "merged" / "merged_metadata_240730.csv",
    ]
    keywords = re.compile(r"diagnos|duration|a1c|hba1c|glyc|diabetes.*year|diabetes.*age", re.I)
    rows = []
    for path in files:
        if not path.exists():
            continue
        data = pd.read_csv(path, nrows=5)
        for column in data.columns:
            if keywords.search(column):
                rows.append({"source_file": str(path.relative_to(REPO_ROOT)), "column": column})
    return pd.DataFrame(rows)


def add_diagnosis_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate years since diabetes diagnosis when enough information exists."""
    out = df.copy()
    out["visit_year"] = out["existing_visit_year"].fillna(out["extra_visit_year"])
    out["years_since_diabetes_diagnosis"] = np.nan

    has_year = out["diabetes_diagnosis_year"].notna() & out["visit_year"].notna()
    out.loc[has_year, "years_since_diabetes_diagnosis"] = (
        out.loc[has_year, "visit_year"] - out.loc[has_year, "diabetes_diagnosis_year"]
    )

    has_age = (
        out["years_since_diabetes_diagnosis"].isna()
        & out["diabetes_diagnosis_age"].notna()
        & pd.to_numeric(out["Age"], errors="coerce").notna()
    )
    out.loc[has_age, "years_since_diabetes_diagnosis"] = (
        pd.to_numeric(out.loc[has_age, "Age"], errors="coerce")
        - out.loc[has_age, "diabetes_diagnosis_age"]
    )

    out.loc[out["years_since_diabetes_diagnosis"] < 0, "years_since_diabetes_diagnosis"] = np.nan
    return out


def summarize_trends(merged: pd.DataFrame, diagnosis_scan: pd.DataFrame) -> str:
    """Create a small text report."""
    included = merged[merged["SET"].isin(["set01", "set02", "set03"])].copy()
    diabetes = included[included["diabetes_flag"]].copy()

    lines = [
        "Diabetes Diagnosis and Medication Metadata Analysis",
        "=================================================",
        "",
        "Direct identifiers from metadata/extra were not written to outputs.",
        "The extra CSV was mapped by row order: first data row -> part09, then incrementing by participant number.",
        "",
        "Existing metadata diagnosis scan:",
    ]
    if diagnosis_scan.empty:
        lines.append("- No existing structured diabetes diagnosis year/age/duration/A1c columns were found outside metadata/extra.")
    else:
        for _, row in diagnosis_scan.iterrows():
            lines.append(f"- {row['source_file']}: {row['column']}")

    lines.extend(
        [
            "",
            f"Included participants in analysis dataframe: {len(included)}",
            f"Participants flagged with diabetes: {len(diabetes)}",
            f"Diabetes participants with diagnosis year or age in extra CSV: {int(diabetes[['diabetes_diagnosis_year', 'diabetes_diagnosis_age']].notna().any(axis=1).sum())}",
            f"Diabetes participants with calculable years since diagnosis: {int(diabetes['years_since_diabetes_diagnosis'].notna().sum())}",
            "",
            "Medication note coverage by group:",
        ]
    )

    group_summary = included.groupby("SET").agg(
        n=("Participant", "count"),
        any_med_note=("has_any_med_note", "sum"),
        unspecified_med_note=("has_unspecified_med_note", "sum"),
        diabetes_med_note=("has_diabetes_med_note", "sum"),
        antihypertensive_med_note=("has_antihypertensive_med_note", "sum"),
        other_nonfalse_note=("has_other_nonfalse_note", "sum"),
    )
    set_names = {"set01": "control", "set02": "hypertension", "set03": "diabetes"}
    for set_id, row in group_summary.iterrows():
        lines.append(
            f"- {set_names.get(set_id, set_id)} ({set_id}): n={int(row['n'])}, "
            f"any medication note={int(row['any_med_note'])}, "
            f"unspecified medication note={int(row['unspecified_med_note'])}, "
            f"diabetes-med note={int(row['diabetes_med_note'])}, "
            f"antihypertensive-med note={int(row['antihypertensive_med_note'])}, "
            f"any non-FALSE Other note={int(row['other_nonfalse_note'])}"
        )

    lines.extend(["", "Diagnosis-duration trend checks:"])
    trend_df = diabetes.dropna(subset=["years_since_diabetes_diagnosis", "CRI"])
    if len(trend_df) >= 3 and stats is not None:
        rho, p_value = stats.spearmanr(trend_df["years_since_diabetes_diagnosis"], trend_df["CRI"])
        lines.append(
            f"- Spearman correlation between years since diabetes diagnosis and CRI: "
            f"rho={rho:.3f}, p={p_value:.3g}, n={len(trend_df)}"
        )
    else:
        lines.append(
            f"- Too few diabetic participants with both diagnosis duration and CRI for a reliable correlation "
            f"(n={len(trend_df)})."
        )

    med_df = diabetes.dropna(subset=["CRI"])
    if len(med_df):
        for flag in ["has_diabetes_med_note", "has_antihypertensive_med_note", "has_any_med_note"]:
            grouped = med_df.groupby(flag)["CRI"].agg(["count", "median", "mean"])
            lines.append(f"- CRI by {flag}:")
            for value, row in grouped.iterrows():
                lines.append(
                    f"  {flag}={bool(value)}: n={int(row['count'])}, "
                    f"median={row['median']:.3f}, mean={row['mean']:.3f}"
                )

    extra_set_numeric = pd.to_numeric(included["extra_set"], errors="coerce")
    extra_set_label = "set" + extra_set_numeric.astype("Int64").astype(str).str.zfill(2)
    set_mismatch = included[
        extra_set_numeric.notna()
        & included["SET"].notna()
        & (extra_set_label != included["SET"])
    ]
    if len(set_mismatch):
        lines.extend(["", "Data-quality flags:"])
        lines.append(
            f"- {len(set_mismatch)} participant(s) had a mismatch between extra CSV set and analysis SET. "
            "See sanitized_participant_metadata.csv."
        )

    invalid_age = included[
        included["diabetes_diagnosis_age_raw"].notna()
        & included["diabetes_diagnosis_age"].isna()
    ]
    if len(invalid_age):
        if "Data-quality flags:" not in lines:
            lines.extend(["", "Data-quality flags:"])
        lines.append(
            f"- {len(invalid_age)} participant(s) had non-age text in the age diabetes diagnosed field. "
            "See sanitized_participant_metadata.csv."
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- Medication information is sparse and captured as free text, not as a structured medication inventory.",
            "- Diagnosis timing is available for only a small subset of diabetes participants.",
            "- These fields are useful for descriptive checks and limitations text, but should not be treated as fully adjusted covariates.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extra = load_extra_diabetes_metadata()
    existing = load_existing_participant_metadata()
    merged = existing.merge(extra, on="Participant", how="left")
    merged = add_medication_flags(merged)
    merged = add_cri(merged)
    merged = add_diagnosis_duration(merged)

    diagnosis_scan = existing_diagnosis_metadata_scan()

    # Do not save direct identifiers. Also drop raw free text from the main CSV;
    # medication flags preserve analyzable information without exporting notes.
    sanitized = merged.drop(columns=["Other", "other_text"], errors="ignore")
    sanitized.to_csv(OUTPUT_DIR / "sanitized_participant_metadata.csv", index=False)

    diabetes_subset_cols = [
        "Participant",
        "SET",
        "Age",
        "SYS_BP",
        "DIA_BP",
        "diabetes_flag",
        "hypertension_flag",
        "CRI",
        "diabetes_diagnosis_year",
        "diabetes_diagnosis_age",
        "years_since_diabetes_diagnosis",
        "diabetes_type_extra",
        "has_diabetes_med_note",
        "has_antihypertensive_med_note",
        "has_any_med_note",
        "has_unspecified_med_note",
    ]
    diabetes_subset = sanitized.loc[
        sanitized["diabetes_flag"], [col for col in diabetes_subset_cols if col in sanitized.columns]
    ].copy()
    diabetes_subset.to_csv(OUTPUT_DIR / "diabetes_participant_summary.csv", index=False)

    diagnosis_scan.to_csv(OUTPUT_DIR / "existing_diagnosis_metadata_scan.csv", index=False)

    group_summary = sanitized[sanitized["SET"].isin(["set01", "set02", "set03"])].groupby("SET").agg(
        n=("Participant", "count"),
        diabetes_n=("diabetes_flag", "sum"),
        hypertension_n=("hypertension_flag", "sum"),
        any_med_note=("has_any_med_note", "sum"),
        unspecified_med_note=("has_unspecified_med_note", "sum"),
        diabetes_med_note=("has_diabetes_med_note", "sum"),
        antihypertensive_med_note=("has_antihypertensive_med_note", "sum"),
        diagnosis_timing_available=("years_since_diabetes_diagnosis", lambda s: int(s.notna().sum())),
    )
    group_summary.to_csv(OUTPUT_DIR / "medication_diagnosis_summary_by_set.csv")

    trend_rows = []
    diabetes_subset_for_trends = sanitized[sanitized["diabetes_flag"]].copy()
    for flag in ["has_diabetes_med_note", "has_antihypertensive_med_note", "has_any_med_note"]:
        for value, group in diabetes_subset_for_trends.groupby(flag):
            cri = group["CRI"].dropna()
            trend_rows.append({
                "analysis": f"CRI_by_{flag}",
                "group_value": bool(value),
                "n": int(len(cri)),
                "median_CRI": float(cri.median()) if len(cri) else np.nan,
                "mean_CRI": float(cri.mean()) if len(cri) else np.nan,
            })
    trend_df = diabetes_subset_for_trends.dropna(subset=["years_since_diabetes_diagnosis", "CRI"])
    if len(trend_df) >= 3 and stats is not None:
        rho, p_value = stats.spearmanr(trend_df["years_since_diabetes_diagnosis"], trend_df["CRI"])
        trend_rows.append({
            "analysis": "Spearman_CRI_vs_years_since_diabetes_diagnosis",
            "group_value": "",
            "n": int(len(trend_df)),
            "median_CRI": np.nan,
            "mean_CRI": np.nan,
            "spearman_rho": float(rho),
            "p_value": float(p_value),
        })
    pd.DataFrame(trend_rows).to_csv(OUTPUT_DIR / "small_trend_checks.csv", index=False)

    report = summarize_trends(merged, diagnosis_scan)
    (OUTPUT_DIR / "diabetes_medication_metadata_analysis.txt").write_text(report, encoding="utf-8")

    print(f"Wrote sanitized analysis outputs to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
