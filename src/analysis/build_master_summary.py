"""Build a master summary of statistical analysis outputs for manuscript work.

This module orchestrates analysis runs, collects p-values/statistics from known
output files, normalizes them into one long-form table, applies FDR correction,
and writes paper-friendly summary artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import PATHS


@dataclass(frozen=True)
class AnalysisSpec:
    """Execution and output contract for an analysis script."""

    key: str
    family: str
    module: str
    scope: str  # "core" or "supplement"
    expected_outputs: Tuple[str, ...]


P_CANDIDATES = (
    "p_value",
    "pvalue",
    "P_Value",
    "P-value",
    "PR(>F)",
    "p",
)

STAT_CANDIDATES = (
    "statistic",
    "KS_Statistic",
    "F",
    "U",
    "r",
)

ANALYSIS_CANDIDATES = (
    "analysis",
    "Comparison",
    "comparison",
    "grouping_factor",
    "factor",
)

TEST_CANDIDATES = (
    "test",
    "test_type",
    "Test",
)

GROUP1_CANDIDATES = ("group1", "Group_1", "group_1")
GROUP2_CANDIDATES = ("group2", "Group_2", "group_2")
N_TOTAL_CANDIDATES = ("n", "N", "n_total")
N1_CANDIDATES = ("n1", "n_group_1", "n_control")
N2_CANDIDATES = ("n2", "n_group_2", "n_diabetic")


CORE_ANALYSES: Tuple[AnalysisSpec, ...] = (
    AnalysisSpec(
        key="stiffness_coeff",
        family="stiffness",
        module="src.analysis.stiffness_coeff",
        scope="core",
        expected_outputs=(
            "results/Stiffness/stiffness_coefficients.csv",
            "results/Stiffness/stiffness_coefficients_log.csv",
        ),
    ),
    AnalysisSpec(
        key="plot_stiffness",
        family="stiffness",
        module="src.analysis.plot_stiffness",
        scope="core",
        expected_outputs=(
            "results/Stiffness/plots/stiffness_significance.csv",
            "results/Stiffness/plots/stiffness_correlations.csv",
            "results/Stiffness/plots/age_group_analysis_significance.csv",
            "results/Stiffness/stiffness_metric_comparison.csv",
        ),
    ),
    AnalysisSpec(
        key="hysteresis",
        family="hysteresis",
        module="src.analysis.hysteresis",
        scope="core",
        expected_outputs=("results/Hysteresis/hysteresis_pvalues.csv",),
    ),
    AnalysisSpec(
        key="anova",
        family="anova",
        module="src.analysis.anova",
        scope="core",
        expected_outputs=(
            "results/ANOVA_Analysis/anova_main_effects.tex",
            "results/ANOVA_Analysis/anova_interaction_effects.tex",
        ),
    ),
    AnalysisSpec(
        key="create_ks_tables",
        family="ks",
        module="src.analysis.create_ks_tables",
        scope="core",
        expected_outputs=(
            "results/ks_statistics_tables/All_Variables_KS_Statistics_Modified.csv",
            "results/ks_statistics_tables/KS_Statistics_Summary_Modified.csv",
        ),
    ),
    AnalysisSpec(
        key="age_threshold",
        family="thresholds",
        module="src.analysis.age_threshold",
        scope="core",
        expected_outputs=("results/AgeThreshold",),
    ),
    AnalysisSpec(
        key="bp_threshold",
        family="thresholds",
        module="src.analysis.bp_threshold",
        scope="core",
        expected_outputs=("results/SYSBPThreshold",),
    ),
    AnalysisSpec(
        key="sex_analysis",
        family="demographics",
        module="src.analysis.sex_analysis",
        scope="core",
        expected_outputs=("results/SexAnalysis",),
    ),
)


SUPPLEMENT_ANALYSES: Tuple[AnalysisSpec, ...] = (
    AnalysisSpec(
        key="com_distance_analysis",
        family="supplement_com_distance",
        module="src.analysis.com_distance_analysis",
        scope="supplement",
        expected_outputs=("results/cap_com_distance.csv",),
    ),
    AnalysisSpec(
        key="com_distance_age_analysis",
        family="supplement_com_distance",
        module="src.analysis.com_distance_age_analysis",
        scope="supplement",
        expected_outputs=(
            "results/com_distance_age_summary.csv",
            "results/com_distance_age_scatter.png",
        ),
    ),
    AnalysisSpec(
        key="health_classifier",
        family="supplement_classifier",
        module="src.analysis.health_classifier",
        scope="supplement",
        expected_outputs=("results/Classifier",),
    ),
    AnalysisSpec(
        key="shear_analysis",
        family="supplement_shear",
        module="src.analysis.shear_analysis",
        scope="supplement",
        expected_outputs=("results/shear",),
    ),
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def _pick_first(series: pd.Series, candidates: Iterable[str]) -> Optional[Any]:
    for c in candidates:
        if c in series.index:
            return series[c]
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> Optional[int]:
    f = _to_float(value)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg correction for a vector of p-values."""
    vals = p_values.astype(float).to_numpy()
    n = len(vals)
    if n == 0:
        return pd.Series(dtype=float)
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return pd.Series(out, index=p_values.index)


def _run_module(module: str, root: Path) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", module]
    started = _now_iso()
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    ended = _now_iso()
    return {
        "module": module,
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
        "started_at": started,
        "ended_at": ended,
        "status": "ok" if proc.returncode == 0 else "failed",
    }


def _output_exists(root: Path, relpath: str) -> bool:
    return (root / relpath).exists()


def _find_existing_outputs(root: Path, expected: Tuple[str, ...]) -> List[str]:
    return [p for p in expected if _output_exists(root, p)]


def _normalize_from_dataframe(
    df: pd.DataFrame,
    *,
    family: str,
    script: str,
    source_file: str,
    run_timestamp: str,
    git_commit: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return rows

    p_col = next((c for c in P_CANDIDATES if c in df.columns), None)
    if p_col is None:
        return rows

    for _, r in df.iterrows():
        p_value = _to_float(r.get(p_col))
        if p_value is None:
            continue

        analysis_name = _pick_first(r, ANALYSIS_CANDIDATES) or Path(source_file).stem
        test_name = _pick_first(r, TEST_CANDIDATES) or "Unknown"
        stat_val = _to_float(_pick_first(r, STAT_CANDIDATES))
        group1 = _pick_first(r, GROUP1_CANDIDATES)
        group2 = _pick_first(r, GROUP2_CANDIDATES)
        n_total = _to_int(_pick_first(r, N_TOTAL_CANDIDATES))
        n1 = _to_int(_pick_first(r, N1_CANDIDATES))
        n2 = _to_int(_pick_first(r, N2_CANDIDATES))
        significant = bool(p_value < 0.05)

        rows.append(
            {
                "analysis_family": family,
                "script": script,
                "source_file": source_file,
                "analysis_name": str(analysis_name),
                "test_name": str(test_name),
                "outcome_variable": None,
                "grouping_variable": str(r.get("grouping_factor")) if "grouping_factor" in df.columns else None,
                "group_1": None if group1 is None else str(group1),
                "group_2": None if group2 is None else str(group2),
                "statistic_name": "statistic",
                "statistic_value": stat_val,
                "p_value_raw": p_value,
                "q_value_fdr_bh": None,
                "significant_raw_0_05": significant,
                "significant_fdr_0_05": None,
                "effect_size_name": None,
                "effect_size_value": None,
                "n_total": n_total,
                "n_group_1": n1,
                "n_group_2": n2,
                "covariates": None,
                "model_formula": None,
                "pressure_range": None,
                "velocity_transform": None,
                "notes": None,
                "status": "ok",
                "error_message": None,
                "run_timestamp": run_timestamp,
                "git_commit": git_commit,
            }
        )

    return rows


def _collect_anova_rows(
    root: Path,
    input_csv: Path,
    run_timestamp: str,
    git_commit: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not input_csv.exists():
        return rows

    try:
        from src.analysis import anova as anova_mod
    except Exception:
        return rows

    try:
        raw_df = pd.read_csv(input_csv)
        participant_df = anova_mod.summarize_participants(raw_df, age_threshold=50)
        result = anova_mod.perform_anova_analysis(participant_df, log_transform=False)
        table_map = {
            "anova_main": "ANOVA main effects",
            "anova_full": "ANOVA with interactions",
        }
        for key, label in table_map.items():
            if key not in result:
                continue
            tdf = result[key].reset_index().rename(columns={"index": "factor"})
            if "PR(>F)" not in tdf.columns:
                continue
            for _, rr in tdf.iterrows():
                factor = str(rr.get("factor", "unknown"))
                if factor.lower() == "residual":
                    continue
                p_value = _to_float(rr.get("PR(>F)"))
                if p_value is None:
                    continue
                rows.append(
                    {
                        "analysis_family": "anova",
                        "script": "src.analysis.anova",
                        "source_file": "in_memory:anova_model_table",
                        "analysis_name": f"{label}: {factor}",
                        "test_name": "ANOVA F-test",
                        "outcome_variable": "Participant_Median_Velocity",
                        "grouping_variable": factor,
                        "group_1": None,
                        "group_2": None,
                        "statistic_name": "F",
                        "statistic_value": _to_float(rr.get("F")),
                        "p_value_raw": p_value,
                        "q_value_fdr_bh": None,
                        "significant_raw_0_05": bool(p_value < 0.05),
                        "significant_fdr_0_05": None,
                        "effect_size_name": None,
                        "effect_size_value": None,
                        "n_total": _to_int(len(participant_df)),
                        "n_group_1": None,
                        "n_group_2": None,
                        "covariates": "Age, Sex, SYS_BP, Diabetes, Hypertension",
                        "model_formula": None,
                        "pressure_range": None,
                        "velocity_transform": "raw",
                        "notes": None,
                        "status": "ok",
                        "error_message": None,
                        "run_timestamp": run_timestamp,
                        "git_commit": git_commit,
                    }
                )
    except Exception as exc:
        rows.append(
            {
                "analysis_family": "anova",
                "script": "src.analysis.anova",
                "source_file": "in_memory:anova_model_table",
                "analysis_name": "anova_collection",
                "test_name": None,
                "outcome_variable": None,
                "grouping_variable": None,
                "group_1": None,
                "group_2": None,
                "statistic_name": None,
                "statistic_value": None,
                "p_value_raw": None,
                "q_value_fdr_bh": None,
                "significant_raw_0_05": None,
                "significant_fdr_0_05": None,
                "effect_size_name": None,
                "effect_size_value": None,
                "n_total": None,
                "n_group_1": None,
                "n_group_2": None,
                "covariates": None,
                "model_formula": None,
                "pressure_range": None,
                "velocity_transform": None,
                "notes": "Failed to collect ANOVA tables directly",
                "status": "failed",
                "error_message": str(exc),
                "run_timestamp": run_timestamp,
                "git_commit": git_commit,
            }
        )
    return rows


def _collect_rows_from_known_outputs(
    root: Path,
    run_timestamp: str,
    git_commit: str,
) -> List[Dict[str, Any]]:
    known = (
        ("stiffness", "src.analysis.plot_stiffness", "results/Stiffness/plots/stiffness_significance.csv"),
        ("stiffness", "src.analysis.plot_stiffness", "results/Stiffness/plots/age_group_analysis_significance.csv"),
        ("hysteresis", "src.analysis.hysteresis", "results/Hysteresis/hysteresis_pvalues.csv"),
        ("ks", "src.analysis.create_ks_tables", "results/ks_statistics_tables/All_Variables_KS_Statistics_Modified.csv"),
        ("ks", "src.analysis.create_ks_tables", "results/ks_statistics_tables/All_Variables_KS_Statistics.csv"),
    )
    out_rows: List[Dict[str, Any]] = []
    for family, script, rel in known:
        fp = root / rel
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        out_rows.extend(
            _normalize_from_dataframe(
                df,
                family=family,
                script=script,
                source_file=rel,
                run_timestamp=run_timestamp,
                git_commit=git_commit,
            )
        )
    return out_rows


def _build_failure_row(
    spec: AnalysisSpec,
    *,
    run_timestamp: str,
    git_commit: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "analysis_family": spec.family,
        "script": spec.module,
        "source_file": None,
        "analysis_name": spec.key,
        "test_name": None,
        "outcome_variable": None,
        "grouping_variable": None,
        "group_1": None,
        "group_2": None,
        "statistic_name": None,
        "statistic_value": None,
        "p_value_raw": None,
        "q_value_fdr_bh": None,
        "significant_raw_0_05": None,
        "significant_fdr_0_05": None,
        "effect_size_name": None,
        "effect_size_value": None,
        "n_total": None,
        "n_group_1": None,
        "n_group_2": None,
        "covariates": None,
        "model_formula": None,
        "pressure_range": None,
        "velocity_transform": None,
        "notes": None,
        "status": "failed",
        "error_message": reason,
        "run_timestamp": run_timestamp,
        "git_commit": git_commit,
    }


def _apply_fdr(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["q_value_fdr_bh"] = np.nan
    ok_mask = df["status"].eq("ok") & df["p_value_raw"].notna()
    for fam in sorted(df.loc[ok_mask, "analysis_family"].dropna().unique()):
        fam_mask = ok_mask & df["analysis_family"].eq(fam)
        if fam_mask.sum() == 0:
            continue
        df.loc[fam_mask, "q_value_fdr_bh"] = _bh_fdr(df.loc[fam_mask, "p_value_raw"])
    df["significant_fdr_0_05"] = np.where(
        df["q_value_fdr_bh"].notna(),
        df["q_value_fdr_bh"] < 0.05,
        np.nan,
    )
    return df


def _markdown_from_table(df: pd.DataFrame, scope_label: str, manifest: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# {scope_label} Analysis Summary")
    lines.append("")
    lines.append(f"- Generated at: `{manifest.get('generated_at')}`")
    lines.append(f"- Git commit: `{manifest.get('git_commit')}`")
    lines.append(f"- Scope: `{scope_label}`")
    lines.append("")

    total_rows = len(df)
    ok_rows = int(df["status"].eq("ok").sum()) if "status" in df.columns else 0
    failed_rows = int(df["status"].eq("failed").sum()) if "status" in df.columns else 0
    sig_raw = int((df["p_value_raw"].notna() & (df["p_value_raw"] < 0.05)).sum()) if "p_value_raw" in df.columns else 0
    sig_fdr = int((df["q_value_fdr_bh"].notna() & (df["q_value_fdr_bh"] < 0.05)).sum()) if "q_value_fdr_bh" in df.columns else 0
    lines.append("## Snapshot")
    lines.append(f"- Total rows: `{total_rows}`")
    lines.append(f"- OK rows: `{ok_rows}`")
    lines.append(f"- Failed rows: `{failed_rows}`")
    lines.append(f"- Significant (raw p<0.05): `{sig_raw}`")
    lines.append(f"- Significant (FDR q<0.05): `{sig_fdr}`")
    lines.append("")

    lines.append("## Top Findings (FDR)")
    cols = [
        "analysis_family",
        "analysis_name",
        "test_name",
        "statistic_value",
        "p_value_raw",
        "q_value_fdr_bh",
        "n_total",
        "group_1",
        "group_2",
        "source_file",
    ]
    keep = [c for c in cols if c in df.columns]
    top = df[df["q_value_fdr_bh"].notna()].sort_values("q_value_fdr_bh").head(40)
    if top.empty:
        lines.append("No FDR-eligible results were found.")
    else:
        lines.append(top[keep].to_markdown(index=False))
    lines.append("")

    lines.append("## Failed Or Missing")
    failed = df[df["status"] != "ok"]
    if failed.empty:
        lines.append("No failed analyses recorded.")
    else:
        fail_cols = [c for c in ["analysis_family", "script", "analysis_name", "status", "error_message"] if c in failed.columns]
        lines.append(failed[fail_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Script Execution Status")
    status_rows = manifest.get("execution", [])
    if status_rows:
        status_df = pd.DataFrame(status_rows)
        status_cols = [c for c in ["analysis_key", "module", "status", "returncode", "found_outputs"] if c in status_df.columns]
        lines.append(status_df[status_cols].to_markdown(index=False))
    else:
        lines.append("No execution records.")
    lines.append("")
    return "\n".join(lines)


def _ensure_outdir(root: Path) -> Path:
    outdir = root / "results" / "PaperSummary"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _choose_specs(mode: str) -> Tuple[AnalysisSpec, ...]:
    if mode == "core":
        return CORE_ANALYSES
    if mode == "supplement":
        return SUPPLEMENT_ANALYSES
    return CORE_ANALYSES + SUPPLEMENT_ANALYSES


def _collect_for_scope(
    *,
    root: Path,
    mode: str,
    input_csv: Path,
    run_timestamp: str,
    git_commit: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    specs = _choose_specs(mode)
    execution_log: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for spec in specs:
        run_res = _run_module(spec.module, root)
        found_outputs = _find_existing_outputs(root, spec.expected_outputs)
        effective_status = run_res["status"]
        if run_res["status"] == "ok" and len(found_outputs) == 0:
            effective_status = "missing_output"
        execution_log.append(
            {
                "analysis_key": spec.key,
                "module": spec.module,
                "status": effective_status,
                "returncode": run_res["returncode"],
                "found_outputs": found_outputs,
                "stdout_tail": run_res["stdout_tail"],
                "stderr_tail": run_res["stderr_tail"],
                "started_at": run_res["started_at"],
                "ended_at": run_res["ended_at"],
            }
        )

        if effective_status != "ok":
            if effective_status == "missing_output":
                reason = "Execution completed but expected outputs were not created."
            else:
                reason = run_res["stderr_tail"] or run_res["stdout_tail"] or "Module execution failed."
            rows.append(
                _build_failure_row(
                    spec,
                    run_timestamp=run_timestamp,
                    git_commit=git_commit,
                    reason=reason,
                )
            )

    rows.extend(_collect_rows_from_known_outputs(root, run_timestamp, git_commit))
    if mode in ("core", "all"):
        rows.extend(_collect_anova_rows(root, input_csv, run_timestamp, git_commit))

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        out_df = pd.DataFrame(
            columns=[
                "analysis_family",
                "script",
                "source_file",
                "analysis_name",
                "test_name",
                "outcome_variable",
                "grouping_variable",
                "group_1",
                "group_2",
                "statistic_name",
                "statistic_value",
                "p_value_raw",
                "q_value_fdr_bh",
                "significant_raw_0_05",
                "significant_fdr_0_05",
                "effect_size_name",
                "effect_size_value",
                "n_total",
                "n_group_1",
                "n_group_2",
                "covariates",
                "model_formula",
                "pressure_range",
                "velocity_transform",
                "notes",
                "status",
                "error_message",
                "run_timestamp",
                "git_commit",
            ]
        )

    out_df = _apply_fdr(out_df)
    manifest = {
        "generated_at": run_timestamp,
        "git_commit": git_commit,
        "mode": mode,
        "cap_flow_root": str(root),
        "input_csv": str(input_csv),
        "execution": execution_log,
    }
    return out_df, manifest


def _write_outputs(
    root: Path,
    df: pd.DataFrame,
    manifest: Dict[str, Any],
    *,
    csv_name: str,
    md_name: str,
) -> None:
    outdir = _ensure_outdir(root)
    csv_path = outdir / csv_name
    md_path = outdir / md_name
    manifest_path = outdir / "run_manifest.json"

    df.to_csv(csv_path, index=False)
    md = _markdown_from_table(df, scope_label=csv_name.replace("_analysis_summary.csv", ""), manifest=manifest)
    md_path.write_text(md, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved summary Markdown: {md_path}")
    print(f"Saved run manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build master statistical summary files for manuscript analyses.")
    parser.add_argument(
        "--mode",
        choices=("core", "supplement", "all"),
        default="all",
        help="Which scope to run and summarize.",
    )
    parser.add_argument(
        "--cap-flow-root",
        default=PATHS.get("cap_flow", str(Path.cwd())),
        help="Project root containing data/results folders.",
    )
    parser.add_argument(
        "--input-csv",
        default="summary_df_nhp_video_stats.csv",
        help="Primary input CSV filename expected under cap-flow-root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.cap_flow_root).resolve()
    run_timestamp = _now_iso()
    commit = _git_commit(root)
    input_csv = root / args.input_csv

    print(f"Using root: {root}")
    print(f"Mode: {args.mode}")
    print(f"Input CSV: {input_csv}")

    if not root.exists():
        print(f"Error: cap-flow root does not exist: {root}")
        return 1

    df, manifest = _collect_for_scope(
        root=root,
        mode=args.mode,
        input_csv=input_csv,
        run_timestamp=run_timestamp,
        git_commit=commit,
    )

    if args.mode == "core":
        _write_outputs(
            root,
            df,
            manifest,
            csv_name="master_analysis_summary.csv",
            md_name="master_analysis_summary.md",
        )
    elif args.mode == "supplement":
        _write_outputs(
            root,
            df,
            manifest,
            csv_name="supplement_analysis_summary.csv",
            md_name="supplement_analysis_summary.md",
        )
    else:
        core_df = df[df["analysis_family"].str.startswith("supplement", na=False) == False].copy()
        supp_df = df[df["analysis_family"].str.startswith("supplement", na=False)].copy()

        _write_outputs(
            root,
            core_df,
            manifest,
            csv_name="master_analysis_summary.csv",
            md_name="master_analysis_summary.md",
        )
        _write_outputs(
            root,
            supp_df,
            manifest,
            csv_name="supplement_analysis_summary.csv",
            md_name="supplement_analysis_summary.md",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
