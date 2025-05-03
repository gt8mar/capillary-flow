0:0
+"""
+finger_multi_finger_stats.py
+=================================
+
+Utility functions to identify participants who have capillary–velocity measurements
+recorded on more than one finger **and** to compare their velocities across those
+fingers.
+
+This is complementary to the existing `finger_usage_groups.py` (which merely groups
+participants by the set of fingers they used) and to the heavy-weight analysis
+performed in `finger_stats.py`.
+
+Typical usage
+-------------
+
+```python
+from src.analysis.finger_multi_finger_stats import (
+    get_multifinger_participants,
+    build_multifinger_summary,
+    run_paired_finger_tests,
+)
+
+# `merged_df` should already contain the cleaned velocity data merged with finger
+# circumference metrics (see `finger_stats.main`).
+
+multi_participants = get_multifinger_participants(merged_df)
+summary_df, wide_df, stats_df = build_multifinger_summary(merged_df, multi_participants)
+print(stats_df)
+```
+
+Functions
+~~~~~~~~~
+* ``get_multifinger_participants`` – returns a list of participants who have more than
+  one unique finger in the dataset.
+* ``build_multifinger_summary`` – creates long- and wide-form DataFrames with median
+  velocities for each participant/finger and, optionally, runs paired t-tests across
+  finger pairs.
+* ``run_paired_finger_tests`` – helper that executes matched-pairs (within-subject)
+  t-tests for every specified finger pair.
+
+All statistical tests are **within-subject** (paired) because the same participant is
+measured with two different fingers.  The output is a tidy ``stats_df`` suitable for
+reporting.
+"""
+
+from __future__ import annotations
+
+from typing import Iterable, List, Tuple
+
+import pandas as pd
+import numpy as np
+from scipy import stats
+
+# -----------------------------------------------------------------------------
+# Participant selection helpers
+# -----------------------------------------------------------------------------
+
+
+def get_multifinger_participants(df: pd.DataFrame, *, finger_col: str = "Finger", participant_col: str = "Participant") -> List[str]:
+    """Return a list of participants who appear with **more than one** finger.
+
+    Parameters
+    ----------
+    df : pandas.DataFrame
+        Input DataFrame containing at least ``participant_col`` and ``finger_col``.
+    finger_col : str, default "Finger"
+        Column indicating which finger was used for the measurement.
+    participant_col : str, default "Participant"
+        Column identifying the participant.
+
+    Returns
+    -------
+    list[str]
+        Participant identifiers that have *n_unique_fingers > 1*.
+    """
+
+    finger_counts = df.groupby(participant_col)[finger_col].nunique()
+    multi_participants = finger_counts[finger_counts > 1].index.tolist()
+    return multi_participants
+
+
+# -----------------------------------------------------------------------------
+# Velocity-comparison helpers
+# -----------------------------------------------------------------------------
+
+
+def build_multifinger_summary(
+    df: pd.DataFrame,
+    participants: Iterable[str] | None = None,
+    *,
+    velocity_col: str = "Video_Median_Velocity",
+    finger_col: str = "Finger",
+    participant_col: str = "Participant",
+) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
+    """Summarise velocities for participants with multiple fingers and run pairwise tests.
+
+    Parameters
+    ----------
+    df : pandas.DataFrame
+        DataFrame containing velocity measurements.
+    participants : Iterable[str] | None, optional
+        Restrict analysis to this subset of participants.  If *None*, the function
+        automatically selects participants with >1 finger via
+        ``get_multifinger_participants``.
+    velocity_col, finger_col, participant_col : str
+        Column names to use.
+
+    Returns
+    -------
+    summary_df : pandas.DataFrame (long form)
+        Median velocity for every (participant, finger) pair.
+    wide_df : pandas.DataFrame (wide form)
+        ``summary_df`` pivoted so each column is a finger and each row is a
+        participant.
+    stats_df : pandas.DataFrame
+        Results of paired t-tests across finger pairs (one row per comparison).
+    """
+
+    # ---------- Participant selection ----------
+    if participants is None:
+        participants = get_multifinger_participants(df, finger_col=finger_col, participant_col=participant_col)
+
+    participants = set(participants)
+    multi_df = df[df[participant_col].isin(participants)].copy()
+
+    if multi_df.empty:
+        raise ValueError("No participants with multiple finger measurements found.")
+
+    # ---------- Compute per-participant per-finger median velocity ----------
+    summary_df = (
+        multi_df.groupby([participant_col, finger_col])[velocity_col]
+        .median()
+        .reset_index()
+        .rename(columns={velocity_col: "Median_Velocity"})
+    )
+
+    # Wide-form
+    wide_df = summary_df.pivot(index=participant_col, columns=finger_col, values="Median_Velocity")
+
+    # ---------- Run paired tests across finger pairs ----------
+    stats_df = run_paired_finger_tests(wide_df)
+
+    return summary_df, wide_df, stats_df
+
+
+def run_paired_finger_tests(
+    wide_df: pd.DataFrame,
+    *,
+    finger_pairs: Iterable[Tuple[str, str]] | None = None,
+) -> pd.DataFrame:
+    """Run paired t-tests between specified finger pairs.
+
+    Parameters
+    ----------
+    wide_df : pandas.DataFrame
+        DataFrame where each *column* is a finger and each *row* a participant.
+    finger_pairs : Iterable[tuple[str, str]] | None, optional
+        Explicit list of pairs to test.  If *None*, all combinations present in
+        ``wide_df`` will be tested.
+
+    Returns
+    -------
+    pandas.DataFrame
+        DataFrame with columns: ``Comparison``, ``n``, ``Mean_Diff``, ``SD_Diff``,
+        ``t_stat``, ``p_value``.
+    """
+
+    # Use all column pairs if none provided
+    if finger_pairs is None:
+        fingers = list(wide_df.columns)
+        finger_pairs = [(f1, f2) for i, f1 in enumerate(fingers) for f2 in fingers[i + 1 :]]
+
+    results: list[dict[str, float | str | int]] = []
+
+    for f1, f2 in finger_pairs:
+        if f1 not in wide_df.columns or f2 not in wide_df.columns:
+            continue  # skip if one finger unavailable
+
+        paired_df = wide_df[[f1, f2]].dropna()
+        if len(paired_df) < 2:
+            continue  # need at least 2 participants for a paired t-test
+
+        t_stat, p_val = stats.ttest_rel(paired_df[f1], paired_df[f2])
+        diff = paired_df[f1] - paired_df[f2]
+
+        results.append(
+            {
+                "Comparison": f"{f1} – {f2}",
+                "n": len(paired_df),
+                "Mean_Diff": diff.mean(),
+                "SD_Diff": diff.std(ddof=1),
+                "t_stat": t_stat,
+                "p_value": p_val,
+            }
+        )
+
+    stats_df = pd.DataFrame(results)
+    return stats_df 