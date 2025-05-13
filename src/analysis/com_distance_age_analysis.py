"""
Filename: com_distance_age_analysis.py
------------------------------------------------------
Merge the inter-capillary centre-of-mass (COM) distance statistics with
participant metadata and explore age-related trends.

Steps
-----
1. Load *video*-level COM distance table produced by
   ``com_distance_analysis.py``.
2. Load the master statistics table (default:
   ``summary_df_nhp_video_stats.csv``) that contains *Age* for every video.
3. Merge on [Participant, Date, Location, Video].
4. Aggregate to *Participant × Location* level (average COM distance and
   capillary count across videos).
5. Perform exploratory analysis:
   • scatter + regression line of Age vs Avg_COM_Distance
   • Pearson correlation (r, p-value)
6. Write the aggregated table to ``results/com_distance_age_summary.csv`` and
   save the plot to ``results/com_distance_age_scatter.png``.

Run
---
>>> python -m src.analysis.com_distance_age_analysis
"""

# Standard library imports
import argparse
import os
import time

# Third-party imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Local imports
from src.config import PATHS

################################################################################
# Helper functions
################################################################################

def _load_video_stats(path: str) -> pd.DataFrame:
    """Read statistics table expected to contain Age per video."""
    df = pd.read_csv(path)
    required = {"Participant", "Date", "Location", "Video", "Age"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Video-stats table is missing columns: {missing}")
    return df[list(required)]

################################################################################
# Main
################################################################################

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assess relation between COM distance and age")
    p.add_argument(
        "--dist",
        default=os.path.join(PATHS["cap_flow"], "results", "cap_com_distance.csv"),
        help="Path to per-video COM distance CSV (default: results/cap_com_distance.csv)",
    )
    p.add_argument(
        "--video-stats",
        default=os.path.join(PATHS["cap_flow"], "summary_df_nhp_video_stats.csv"),
        help="Path to master video statistics file that contains Age (default: summary_df_nhp_video_stats.csv)",
    )
    p.add_argument(
        "--out-csv",
        default=os.path.join(PATHS["cap_flow"], "results", "com_distance_age_summary.csv"),
        help="Where to write aggregated table (default: results/com_distance_age_summary.csv)",
    )
    p.add_argument(
        "--out-fig",
        default=os.path.join(PATHS["cap_flow"], "results", "com_distance_age_scatter.png"),
        help="Where to save scatter plot (default: results/com_distance_age_scatter.png)",
    )
    return p.parse_args()


def main() -> None:  # noqa: D401
    args = _parse_args()
    t0 = time.time()

    print("Loading COM-distance table…")
    dist_df = pd.read_csv(args.dist)

    print("Loading video statistics (incl. Age)…")
    stats_df = _load_video_stats(args.video_stats)

    # Merge
    print("Merging…")
    merged = pd.merge(
        dist_df,
        stats_df,
        on=["Participant", "Date", "Location", "Video"],
        how="left",
    )

    missing_age = merged["Age"].isna().sum()
    if missing_age:
        print(f"Warning: {missing_age} merged rows have missing Age; they will be dropped from analysis.")
    merged = merged.dropna(subset=["Age", "Mean_COM_Distance"])

    # Aggregate to Participant × Location
    group_cols = ["Participant", "Location"]
    summary = (
        merged.groupby(group_cols)
        .agg(
            Avg_COM_Distance=("Mean_COM_Distance", "mean"),
            Avg_N_Capillaries=("N_Capillaries", "mean"),
            Videos=("Video", "nunique"),
            Age=("Age", "first"),  # Age is constant per participant
        )
        .reset_index()
    )

    # Correlation analysis
    print("Computing Pearson correlation (Age vs Avg_COM_Distance)…")
    r, p_val = stats.pearsonr(summary["Age"], summary["Avg_COM_Distance"])
    print(f"r = {r:.3f},  p = {p_val:.4g}")

    # Scatter plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(3, 2.5))
    sns.regplot(x="Age", y="Avg_COM_Distance", data=summary, ci=95, scatter_kws={"s": 20, "alpha": 0.7})
    plt.xlabel("Age (years)")
    plt.ylabel("Mean inter-capillary distance (pixels)")
    plt.title(f"Age vs COM distance  (r={r:.2f}, p={'<0.001' if p_val < 0.001 else f'{p_val:.3f}'})", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    plt.savefig(args.out_fig, dpi=300)
    plt.close()
    print(f"Scatter plot saved to: {args.out_fig}")

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    print(f"Aggregated table written to: {args.out_csv}")

    print(f"Done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main() 