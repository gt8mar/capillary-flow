"""
Filename: make_center_of_mass.py
------------------------------------------------------
Utility script to compute the geometric centre (centre of mass) of each
capillary centre-line file.

Each centre-line CSV contains the x (row) and y (column) pixel coordinates
in its first two columns.  The centre of mass is simply the mean x and mean y
values of those pixels.  The script extracts metadata (Participant, Date,
Location, Video, Capillary) from the filename using ``parse_filename`` and
returns / writes a tidy DataFrame with the following columns::

    Participant  Date  Location  Video  Capillary  COM_X  COM_Y

Example
-------
>>> python -m src.analysis.make_center_of_mass  # writes results/center_of_mass.csv

By: Marcus Forst (original code structure)
Edited: <today by ChatGPT>
"""

# Standard library imports
import os
import time
import platform
from typing import List

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from src.tools.parse_filename import parse_filename
from src.config import PATHS  # centralised paths dictionary

###############################################################################
# Helper functions
###############################################################################

def _log_progress(index: int, total: int, prefix: str = "Processing") -> None:
    """Lightweight progress bar to stdout."""
    pct = (index + 1) / total * 100
    print(f"\r{prefix}: {index + 1}/{total} [{pct:5.1f}%]", end="")


def center_of_mass_summary(directory: str) -> pd.DataFrame:
    """Compute centre of mass for every centre-line CSV in *directory*.

    Parameters
    ----------
    directory : str
        Path containing centre-line ``*.csv`` files.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with one row per capillary and columns
        [Participant, Date, Location, Video, Capillary, COM_X, COM_Y].
    """
    csv_files: List[str] = [f for f in os.listdir(directory) if f.lower().endswith(".csv")]
    total = len(csv_files)
    print(f"Found {total} centre-line files to process in '{directory}'.")

    records = []
    for i, filename in enumerate(csv_files):
        _log_progress(i, total, prefix="Computing COM")
        file_path = os.path.join(directory, filename)

        # Read centre-line CSV; first two columns are x (row) and y (col)
        try:
            df = pd.read_csv(file_path, header=None, usecols=[0, 1])
        except Exception as exc:
            print(f"\nWarning: Could not read {file_path}: {exc}")
            continue

        # Numerical centre of mass
        com_x: float = df.iloc[:, 0].mean()
        com_y: float = df.iloc[:, 1].mean()

        # Parse metadata from filename
        try:
            participant, date, location, video, _ = parse_filename(filename)
        except Exception as exc:
            print(f"\nWarning: Could not parse metadata from {filename}: {exc}")
            continue

        capillary = filename.split(".")[0].split("_")[-1]

        records.append({
            "Participant": participant,
            "Date": date,
            "Location": location,
            "Video": video.replace("bp", ""),  # keep consistency with other tables
            "Capillary": capillary,
            "COM_X": com_x,
            "COM_Y": com_y,
        })

    print("\nCentre-of-mass computation complete!\n")
    return pd.DataFrame(records)

###############################################################################
# Main entry-point
###############################################################################

def main(write: bool = True) -> pd.DataFrame:  # noqa: D401
    """Create the centre-of-mass table and (optionally) persist it to *results*.

    The output file is written to
    ``<cap_flow>/results/cap_center_of_mass.csv``.
    """
    centerlines_dir: str
    if platform.system() == "Windows":
        centerlines_dir = os.path.join(PATHS["cap_flow"], "results", "centerlines")
    else:
        centerlines_dir = os.path.join(PATHS["cap_flow"], "results", "centerlines")

    start_t = time.time()
    com_df = center_of_mass_summary(centerlines_dir)

    if write:
        out_path = os.path.join(PATHS["cap_flow"], "results", "cap_center_of_mass.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        com_df.to_csv(out_path, index=False)
        print(f"Centre-of-mass table written to: {out_path}")

    print(f"Total run-time: {time.time() - start_t:.2f} s")
    return com_df


if __name__ == "__main__":  # pragma: no cover
    main(write=True) 