"""Generic video quality utilities shared across repositories.

These helpers intentionally avoid domain-specific language so they can be
reused for any frame-sequence quality workflow.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

DEFAULT_NUMERIC_FIELDS = (
    "mean_intensity_avg",
    "mean_intensity_sd",
    "contrast_avg",
    "contrast_sd",
    "edge_energy_avg",
    "edge_energy_sd",
    "frame_diff_avg",
    "frame_diff_sd",
    "n_frames",
    "readable_frames",
    "unreadable_frames",
)


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def read_video_index_csv(
    path: Path | str,
    numeric_fields: Sequence[str] = DEFAULT_NUMERIC_FIELDS,
    int_fields: Sequence[str] = ("global_seq", "run_id", "pulses", "pulse_width_us", "delta_t_us"),
) -> List[Dict[str, object]]:
    """Read a CSV index and coerce standard numeric fields."""
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    parsed: List[Dict[str, object]] = []
    for row in rows:
        out: Dict[str, object] = dict(row)
        for field in numeric_fields:
            out[field] = safe_float(row.get(field, 0))
        for field in int_fields:
            out[field] = int(safe_float(row.get(field, 0)))
        parsed.append(out)
    return parsed


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / len(values)
    sd = math.sqrt(var)
    return mu, (sd if sd > 1e-12 else 1.0)


def enrich_quality_scores(
    rows: List[MutableMapping[str, object]],
    *,
    edge_weight: float = 0.45,
    contrast_weight: float = 0.25,
    intensity_stability_weight: float = 0.20,
    motion_stability_weight: float = 0.10,
    unreadable_penalty_scale: float = 6.0,
) -> None:
    """Add z-score components and a composite `quality_score` to each row."""
    edge_vals = [safe_float(r.get("edge_energy_avg", 0.0)) for r in rows]
    contrast_vals = [safe_float(r.get("contrast_avg", 0.0)) for r in rows]
    intensity_sd_vals = [safe_float(r.get("mean_intensity_sd", 0.0)) for r in rows]
    frame_diff_sd_vals = [safe_float(r.get("frame_diff_sd", 0.0)) for r in rows]

    edge_mu, edge_sd = mean_std(edge_vals)
    contrast_mu, contrast_sd = mean_std(contrast_vals)
    intensity_sd_mu, intensity_sd_sd = mean_std(intensity_sd_vals)
    frame_diff_sd_mu, frame_diff_sd_sd = mean_std(frame_diff_sd_vals)

    for row in rows:
        edge_z = (safe_float(row.get("edge_energy_avg", 0.0)) - edge_mu) / edge_sd
        contrast_z = (safe_float(row.get("contrast_avg", 0.0)) - contrast_mu) / contrast_sd
        intensity_stability_z = -(
            (safe_float(row.get("mean_intensity_sd", 0.0)) - intensity_sd_mu) / intensity_sd_sd
        )
        motion_stability_z = -(
            (safe_float(row.get("frame_diff_sd", 0.0)) - frame_diff_sd_mu) / frame_diff_sd_sd
        )
        n_frames = max(1.0, safe_float(row.get("n_frames", 0.0)))
        unreadable_ratio = safe_float(row.get("unreadable_frames", 0.0)) / n_frames
        unreadable_penalty = unreadable_penalty_scale * unreadable_ratio

        quality_score = (
            edge_weight * edge_z
            + contrast_weight * contrast_z
            + intensity_stability_weight * intensity_stability_z
            + motion_stability_weight * motion_stability_z
            - unreadable_penalty
        )

        row["edge_z"] = edge_z
        row["contrast_z"] = contrast_z
        row["intensity_stability_z"] = intensity_stability_z
        row["motion_stability_z"] = motion_stability_z
        row["unreadable_ratio"] = unreadable_ratio
        row["quality_score"] = quality_score


def rank_rows_by_group(
    rows: Sequence[Mapping[str, object]],
    *,
    group_keys: Sequence[str] = ("pulses", "pulse_width_us", "delta_t_us"),
    score_key: str = "quality_score",
    sequence_key: str = "global_seq",
) -> List[Dict[str, object]]:
    """Rank rows by `score_key` descending within each group."""
    grouped: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in group_keys)].append(row)

    ranked_rows: List[Dict[str, object]] = []
    for key in sorted(grouped.keys()):
        group_rows = list(grouped[key])
        group_rows.sort(
            key=lambda r: (-safe_float(r.get(score_key, 0.0)), int(safe_float(r.get(sequence_key, 0))))
        )
        condition_id = "p{}_pw{}_dt{}".format(key[0], key[1], key[2])
        for rank, row in enumerate(group_rows, start=1):
            out = dict(row)
            out["condition_rank"] = rank
            out["condition_size"] = len(group_rows)
            out["condition_id"] = condition_id
            ranked_rows.append(out)
    return ranked_rows


def compute_frame_metrics(
    frame_paths: Sequence[Path | str],
    *,
    downsample: Tuple[int, int] = (360, 270),
) -> Dict[str, float]:
    """Compute quality metrics for a frame sequence.

    Returns means and standard deviations for intensity, contrast, edges, and
    frame-to-frame differences. Unreadable frames are skipped and counted.
    """
    import numpy as np
    from PIL import Image, ImageChops, ImageFilter

    means: List[float] = []
    contrasts: List[float] = []
    edges: List[float] = []
    diffs: List[float] = []
    unreadable = 0
    prev = None

    for frame_path in frame_paths:
        try:
            img = Image.open(frame_path).convert("L").resize(downsample)
        except Exception:
            unreadable += 1
            continue

        arr = np.asarray(img, dtype=np.float32)
        means.append(float(arr.mean()))
        contrasts.append(float(arr.std()))

        edge_img = img.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.asarray(edge_img, dtype=np.float32)
        edges.append(float(edge_arr.mean()))

        if prev is not None:
            diff_img = ImageChops.difference(img, prev)
            diff_arr = np.asarray(diff_img, dtype=np.float32)
            diffs.append(float(diff_arr.mean()))
        prev = img

    def fmean(values: Sequence[float]) -> float:
        return float(statistics.fmean(values)) if values else 0.0

    def pstdev(values: Sequence[float]) -> float:
        return float(statistics.pstdev(values)) if len(values) > 1 else 0.0

    return {
        "readable_frames": len(means),
        "unreadable_frames": unreadable,
        "mean_intensity_avg": fmean(means),
        "mean_intensity_sd": pstdev(means),
        "contrast_avg": fmean(contrasts),
        "contrast_sd": pstdev(contrasts),
        "edge_energy_avg": fmean(edges),
        "edge_energy_sd": pstdev(edges),
        "frame_diff_avg": fmean(diffs),
        "frame_diff_sd": pstdev(diffs),
    }


def adjacent_parameter_comparisons(
    rows: Sequence[Mapping[str, object]],
    *,
    seq_key: str = "global_seq",
    timestamp_key: str = "timestamp",
    score_key: str = "quality_score",
    parameter_keys: Sequence[str] = ("pulses", "pulse_width_us", "delta_t_us"),
) -> List[Dict[str, object]]:
    """Build adjacent pair comparisons where at least one parameter changes."""
    sorted_rows = sorted(rows, key=lambda r: int(safe_float(r.get(seq_key, 0))))
    out: List[Dict[str, object]] = []
    for left, right in zip(sorted_rows, sorted_rows[1:]):
        left_seq = int(safe_float(left.get(seq_key, 0)))
        right_seq = int(safe_float(right.get(seq_key, 0)))
        if right_seq != left_seq + 1:
            continue

        changes = []
        for key in parameter_keys:
            if left.get(key) != right.get(key):
                changes.append(f"{key}:{left.get(key)}->{right.get(key)}")
        if not changes:
            continue

        gap_seconds = None
        left_ts = left.get(timestamp_key)
        right_ts = right.get(timestamp_key)
        try:
            if left_ts and right_ts:
                t0 = datetime.strptime(str(left_ts), "%Y%m%d_%H%M%S%f")
                t1 = datetime.strptime(str(right_ts), "%Y%m%d_%H%M%S%f")
                gap_seconds = (t1 - t0).total_seconds()
        except Exception:
            gap_seconds = None

        delta_score = safe_float(right.get(score_key, 0.0)) - safe_float(left.get(score_key, 0.0))
        out.append(
            {
                "left_seq": left_seq,
                "right_seq": right_seq,
                "time_gap_s": gap_seconds,
                "delta_score": delta_score,
                "abs_delta_score": abs(delta_score),
                "changes": ", ".join(changes),
            }
        )
    return out
