"""
Filename: com_distance_analysis.py
------------------------------------------------------
Compute spatial statistics (average inter-capillary distance) for each
participant video using the centre-of-mass table produced by
``make_center_of_mass.py``.

Output
------
A tidy DataFrame with one row per (Participant, Date, Location, Video) and
columns:
    Participant  Date  Location  Video  N_Capillaries  Mean_COM_Distance
The table is written to ``<cap_flow>/results/cap_com_distance.csv`` so that it
can later be merged into the master `diameter_analysis_df.csv`.

Usage
-----
>>> python -m src.analysis.com_distance_analysis        # writes the CSV
>>> python -m src.analysis.com_distance_analysis --no-write   # just preview
"""

from __future__ import annotations

# Standard library imports
import argparse
import os
import time
from typing import List

# Third-party imports
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

# Local imports
from src.config import PATHS

###############################################################################
# Core computation
###############################################################################

def average_com_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean pairwise COM distance for each (participant, date, location, video)."""
    required_cols = {"Participant", "Date", "Location", "Video", "COM_X", "COM_Y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {missing}")

    records: List[dict] = []
    group_cols = ["Participant", "Date", "Location", "Video"]

    for keys, grp in df.groupby(group_cols):
        n_caps = len(grp)
        if n_caps < 2:
            mean_dist = np.nan
        else:
            coords = grp[["COM_X", "COM_Y"]].to_numpy(float)
            mean_dist = pdist(coords).mean()  # average of all pairwise distances

        records.append({
            **dict(zip(group_cols, keys)),
            "N_Capillaries": n_caps,
            "Mean_COM_Distance": mean_dist,
        })

    return pd.DataFrame(records)

###############################################################################
# CLI
###############################################################################

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute average COM distances per video")
    parser.add_argument(
        "--input",
        default=os.path.join(PATHS["cap_flow"], "results", "cap_center_of_mass.csv"),
        help="Path to centre-of-mass CSV (default: results/cap_center_of_mass.csv)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PATHS["cap_flow"], "results", "cap_com_distance.csv"),
        help="Where to write the summary CSV (default: results/cap_com_distance.csv)",
    )
    parser.add_argument("--no-write", action="store_true", help="Do not write CSV to disk")
    return parser.parse_args()

###############################################################################
# Main
###############################################################################

def main() -> None:
    args = _parse_args()

    t0 = time.time()
    print(f"Loading centre-of-mass table: {args.input}")
    com_df = pd.read_csv(args.input)

    dist_df = average_com_distance(com_df)
    print(dist_df.head())

    if not args.no_write:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        dist_df.to_csv(args.output, index=False)
        print(f"Distance summary written to: {args.output}")

    print(f"Completed in {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main() 