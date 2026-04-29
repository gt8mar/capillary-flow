"""
Filename: diagnose_basler.py
-----------------------------
Generates stabilization diagnostics for a Basler video folder: temporal mean
images (before/after), sharpness metrics, frame-to-frame stability, and shift
summary plots.  Designed for rapid parameter iteration — no contrast step.

Two correction modes:
  fixed      — register every frame to a single template (legacy moco behavior)
  sequential — register each frame to its predecessor, accumulate shifts.
               Handles sudden jumps/discontinuities that break fixed-template.

Usage:
    python scripts/diagnose_basler.py vid08 --frames 30
    python scripts/diagnose_basler.py vid08 --frames 30 --mode sequential
    python scripts/diagnose_basler.py vid08 --registration-crop 0.75
    python scripts/diagnose_basler.py vid08 --template-frame 10
    python scripts/diagnose_basler.py vid08  # full run

By: Marcus Forst
"""

import argparse
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add moco-py to path
MOCO_PY_PATH = "C:\\Users\\gt8ma\\moco-py"
if MOCO_PY_PATH not in sys.path:
    sys.path.insert(0, MOCO_PY_PATH)

from moco_py import MotionCorrector
from moco_py.io import shifts_to_results
from moco_py.plotting import plot_correction_summary

# Reuse the frame loader from preprocess_basler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.preprocess_basler import load_tiff_stack

DEFAULT_BASLER_DIR = "C:\\Users\\gt8ma\\Basler"


def laplacian_variance(image):
    """Compute the Laplacian variance (sharpness metric) of a grayscale image.

    Args:
        image: 2D numpy array (uint8 or float).

    Returns:
        float: Variance of the Laplacian. Higher = sharper.
    """
    lap = cv2.Laplacian(image.astype(np.float64), cv2.CV_64F)
    return float(lap.var())


def mean_frame_to_frame_diff(stack):
    """Compute the mean absolute difference between consecutive frames.

    Args:
        stack: 3D numpy array (N, H, W).

    Returns:
        float: Mean of |frame[i] - frame[i-1]| across all consecutive pairs.
    """
    if stack.shape[0] < 2:
        return 0.0
    diffs = np.abs(stack[1:].astype(np.float64) - stack[:-1].astype(np.float64))
    return float(diffs.mean())


def sequential_correct(stack, mc):
    """Register each frame to its predecessor and accumulate shifts.

    Unlike fixed-template correction, this handles sudden displacements
    (e.g. finger twitches) because each frame-to-frame shift is small
    even when the total drift from frame 0 is large.

    Args:
        stack: 3D numpy array (N, H, W) of raw frames.
        mc: MotionCorrector instance (uses its max_shift / registration_crop).

    Returns:
        tuple: (corrected_stack, cumulative_shifts, rms_errors)
            - corrected_stack: (N, H, W) array aligned to frame 0
            - cumulative_shifts: List of [x, y] total shifts per frame
            - rms_errors: List of per-frame RMS errors (frame-to-frame)
    """
    n = stack.shape[0]

    # Frame-to-frame shifts
    pair_shifts = [[0, 0]]
    rms_errors = [0.0]
    for i in range(1, n):
        _, shifts, rms = mc.correct_stack(
            stack[i:i + 1], stack[i - 1], return_rms=True
        )
        pair_shifts.append(shifts[0])
        rms_errors.append(rms[0])

    # Accumulate to get position relative to frame 0
    cumulative_shifts = [[0, 0]]
    for i in range(1, n):
        cx = cumulative_shifts[-1][0] + pair_shifts[i][0]
        cy = cumulative_shifts[-1][1] + pair_shifts[i][1]
        cumulative_shifts.append([cx, cy])

    # Apply cumulative shifts (need a separate mc with crop_edges=False)
    mc_apply = MotionCorrector(
        max_shift=max(abs(s) for pair in cumulative_shifts for s in pair) + 1,
        crop_edges=False
    )
    corrected_stack = mc_apply.apply_shifts(stack, cumulative_shifts)

    return corrected_stack, cumulative_shifts, rms_errors


def save_temporal_mean_comparison(mean_before, mean_after, save_path,
                                 sharp_before, sharp_after):
    """Save a side-by-side comparison of temporal mean images.

    Args:
        mean_before: 2D float array, temporal mean before stabilization.
        mean_after: 2D float array, temporal mean after stabilization.
        save_path: Output PNG path.
        sharp_before: Sharpness value for the before image.
        sharp_after: Sharpness value for the after image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(mean_before, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Before (sharpness: {sharp_before:.1f})")
    axes[0].axis("off")

    axes[1].imshow(mean_after, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"After (sharpness: {sharp_after:.1f})")
    axes[1].axis("off")

    ratio = sharp_after / sharp_before if sharp_before > 0 else float("inf")
    fig.suptitle(f"Temporal Mean — Sharpness improvement: {ratio:.2f}x",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Stabilization diagnostics for Basler video folders"
    )
    parser.add_argument(
        "video",
        help="Video folder name (e.g. vid08)"
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help="Only process first N frames (default: all)"
    )
    parser.add_argument(
        "--registration-crop", type=float, default=0.5,
        help="Fraction of image center for shift computation (default: 0.5)"
    )
    parser.add_argument(
        "--max-shift", type=int, default=50,
        help="Maximum allowed shift in pixels (default: 50)"
    )
    parser.add_argument(
        "--template-frame", type=int, default=0,
        help="Index of the frame to use as the registration template (default: 0)"
    )
    parser.add_argument(
        "--mode", choices=["fixed", "sequential"], default="fixed",
        help="Correction mode: 'fixed' (single template) or 'sequential' "
             "(frame-to-frame, handles jumps) (default: fixed)"
    )
    parser.add_argument(
        "--crop-edges", action="store_true",
        help="Crop corrected frames to the valid overlap region"
    )
    parser.add_argument(
        "--basler-dir", default=DEFAULT_BASLER_DIR,
        help=f"Path to Basler directory (default: {DEFAULT_BASLER_DIR})"
    )
    args = parser.parse_args()

    video_folder = os.path.join(args.basler_dir, args.video)
    if not os.path.isdir(video_folder):
        print(f"Error: folder not found: {video_folder}")
        sys.exit(1)

    diag_dir = os.path.join(video_folder, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    # --- Load frames ---
    t0 = time.time()
    print(f"[{args.video}] Loading frames...")
    stack, filenames = load_tiff_stack(video_folder)

    if args.frames is not None and args.frames < stack.shape[0]:
        stack = stack[:args.frames]
        filenames = filenames[:args.frames]
        print(f"  Truncated to first {args.frames} frames")

    n_frames = stack.shape[0]
    template_idx = args.template_frame
    if template_idx >= n_frames:
        print(f"  Warning: template-frame {template_idx} >= {n_frames} frames, "
              f"using frame 0")
        template_idx = 0

    template = stack[template_idx]

    # --- Run motion correction ---
    print(f"  Running motion correction "
          f"(mode={args.mode}, crop={args.registration_crop}, "
          f"max_shift={args.max_shift}, "
          f"template=frame {template_idx}, crop_edges={args.crop_edges})...")
    t_moco = time.time()
    mc = MotionCorrector(
        max_shift=args.max_shift,
        crop_edges=args.crop_edges,
        registration_crop=args.registration_crop
    )

    if args.mode == "sequential":
        corrected_stack, shifts, rms_errors = sequential_correct(stack, mc)
    else:
        corrected_stack, shifts, rms_errors = mc.correct_stack(
            stack, template, return_rms=True
        )
    moco_time = time.time() - t_moco
    print(f"  Motion correction: {moco_time:.1f}s")

    # --- Crop raw stack to match if crop_edges was used ---
    if args.crop_edges and corrected_stack.shape != stack.shape:
        # Crop raw frames to the same valid region for fair comparison
        h_corr, w_corr = corrected_stack.shape[1], corrected_stack.shape[2]
        h_raw, w_raw = stack.shape[1], stack.shape[2]
        y_off = (h_raw - h_corr) // 2
        x_off = (w_raw - w_corr) // 2
        stack = stack[:, y_off:y_off + h_corr, x_off:x_off + w_corr]
        print(f"  Cropped both stacks to {stack.shape[1]}x{stack.shape[2]} "
              f"for fair comparison")

    # --- Before-stabilization metrics ---
    print("  Computing before-stabilization metrics...")
    mean_before = stack.astype(np.float64).mean(axis=0)
    sharp_before = laplacian_variance(mean_before)
    ftf_before = mean_frame_to_frame_diff(stack)

    # --- After-stabilization metrics ---
    print("  Computing after-stabilization metrics...")
    mean_after = corrected_stack.astype(np.float64).mean(axis=0)
    sharp_after = laplacian_variance(mean_after)
    ftf_after = mean_frame_to_frame_diff(corrected_stack)

    # --- Shift statistics ---
    shifts_arr = np.array(shifts)  # (N, 2) — [x, y] per frame
    shift_x = shifts_arr[:, 0]
    shift_y = shifts_arr[:, 1]
    magnitudes = np.sqrt(shift_x.astype(float)**2 + shift_y.astype(float)**2)

    # --- Save temporal mean comparison ---
    mean_path = os.path.join(diag_dir, "temporal_mean.png")
    save_temporal_mean_comparison(mean_before, mean_after, mean_path,
                                 sharp_before, sharp_after)
    print(f"  Saved {mean_path}")

    # --- Save shift summary plot ---
    shift_path = os.path.join(diag_dir, "shift_summary.png")
    results = shifts_to_results(shifts, rms_errors)
    plot_correction_summary(results, save_path=shift_path, show=False,
                            title=f"{args.video} — Shift Summary")
    plt.close("all")
    print(f"  Saved {shift_path}")

    # --- Console summary ---
    total_time = time.time() - t0
    sharp_ratio = sharp_after / sharp_before if sharp_before > 0 else float("inf")

    print()
    print("=" * 60)
    print(f"  DIAGNOSTICS: {args.video}")
    print("=" * 60)
    print(f"  Frames:            {n_frames}")
    print(f"  Mode:              {args.mode}")
    print(f"  Template frame:    {template_idx}")
    print(f"  Registration crop: {args.registration_crop}")
    print(f"  Max shift:         {args.max_shift}")
    print(f"  Crop edges:        {args.crop_edges}")
    print("-" * 60)
    print(f"  Sharpness (before):  {sharp_before:.1f}")
    print(f"  Sharpness (after):   {sharp_after:.1f}")
    print(f"  Sharpness ratio:     {sharp_ratio:.2f}x")
    print("-" * 60)
    print(f"  Frame-to-frame diff (before): {ftf_before:.2f}")
    print(f"  Frame-to-frame diff (after):  {ftf_after:.2f}")
    ftf_reduction = (1 - ftf_after / ftf_before) * 100 if ftf_before > 0 else 0
    print(f"  Reduction:                    {ftf_reduction:.1f}%")
    print("-" * 60)
    print(f"  Shift X range:     [{shift_x.min()}, {shift_x.max()}]")
    print(f"  Shift Y range:     [{shift_y.min()}, {shift_y.max()}]")
    print(f"  Mean magnitude:    {magnitudes.mean():.1f} px")
    print(f"  Max magnitude:     {magnitudes.max():.1f} px")
    print(f"  Mean RMS error:    {np.mean(rms_errors):.2f}")
    print("-" * 60)
    print(f"  Moco time:         {moco_time:.1f}s")
    print(f"  Total time:        {total_time:.1f}s")
    print(f"  Output dir:        {diag_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
