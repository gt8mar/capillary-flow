"""Apply the verified part30 diabetes metadata correction to derived CSV files.

The raw participant metadata marks part30 as a diabetes participant in set03.
Some derived analysis tables were generated before that correction and still
classify part30 as a control. This script updates only analysis-facing metadata
fields and does not read or write direct participant identifiers.
"""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PARTICIPANT = "part30"
CORRECTIONS = {
    "SET": "set03",
    "Diabetes": True,
    "Hypertension": False,
    "is_healthy": False,
}

CSV_PATHS = [
    REPO_ROOT / "summary_df_nhp_video_stats.csv",
    REPO_ROOT / "summary_df_nhp_video_medians.csv",
    REPO_ROOT / "summary_df_with_stiffness.csv",
    REPO_ROOT / "results" / "Stiffness" / "cri_metrics.csv",
    REPO_ROOT / "results" / "Stiffness" / "stiffness_coefficients.csv",
    REPO_ROOT / "results" / "Stiffness" / "stiffness_coefficients_log.csv",
    REPO_ROOT / "results" / "stiffness_metrics.csv",
    REPO_ROOT / "results" / "stiffness_metrics_log.csv",
]


def _coerce_bool_value(series: pd.Series, value: bool):
    """Return a boolean value formatted like the existing column values."""
    non_null = series.dropna()
    if non_null.empty:
        return value

    sample = non_null.iloc[0]
    if isinstance(sample, str):
        upper_values = {str(v).upper() for v in non_null.unique()}
        if upper_values <= {"TRUE", "FALSE"}:
            return "TRUE" if value else "FALSE"
        return str(value)
    return value


def apply_corrections(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"path": str(csv_path), "status": "missing"}

    df = pd.read_csv(csv_path)
    if "Participant" not in df.columns:
        return {"path": str(csv_path), "status": "skipped_no_participant_col"}

    mask = df["Participant"].astype(str).eq(PARTICIPANT)
    if not mask.any():
        return {"path": str(csv_path), "status": "skipped_no_part30"}

    before = {}
    after = {}
    for col, value in CORRECTIONS.items():
        if col not in df.columns:
            continue
        before[col] = sorted({str(v) for v in df.loc[mask, col].dropna().unique()})
        if isinstance(value, bool):
            df.loc[mask, col] = _coerce_bool_value(df[col], value)
        else:
            df.loc[mask, col] = value
        after[col] = sorted({str(v) for v in df.loc[mask, col].dropna().unique()})

    df.to_csv(csv_path, index=False)
    return {
        "path": str(csv_path.relative_to(REPO_ROOT)),
        "status": "updated",
        "rows": int(mask.sum()),
        "before": before,
        "after": after,
    }


def main() -> None:
    for csv_path in CSV_PATHS:
        result = apply_corrections(csv_path)
        print(result)


if __name__ == "__main__":
    main()
