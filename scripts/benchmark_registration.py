"""
Filename: benchmark_registration.py
-------------------------------------
Runs multiple registration algorithms on ground truth frame pairs and
reports accuracy.  Reads the CSV produced by annotate_shifts.py.

Algorithms benchmarked:
    1. moco-py fixed       -- MotionCorrector.correct_stack with fixed template
    2. moco-py sequential  -- frame-to-frame chained registration
    3. cv2.phaseCorrelate  -- FFT-based, sub-pixel
    4. cv2.findTransformECC (TRANSLATION) -- iterative, sub-pixel
    5. cv2.findTransformECC (EUCLIDEAN)   -- adds rotation detection
    6. ORB+RANSAC          -- CLAHE + ORB feature matching + RANSAC translation
    7. CLAHE+phaseCorr     -- CLAHE + Hanning window + phase correlate
    8. masked phaseCorr    -- CLAHE + adaptive threshold mask + phase correlate

Usage:
    python scripts/benchmark_registration.py vid08
    python scripts/benchmark_registration.py vid01 --basler-dir D:\\other\\path

By: Marcus Forst
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

# Add moco-py to path
MOCO_PY_PATH = "C:\\Users\\gt8ma\\moco-py"
if MOCO_PY_PATH not in sys.path:
    sys.path.insert(0, MOCO_PY_PATH)

from moco_py import MotionCorrector

# Reuse the frame loader from preprocess_basler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.preprocess_basler import load_tiff_stack

DEFAULT_BASLER_DIR = "C:\\Users\\gt8ma\\Basler"


def load_ground_truth(csv_path):
    """Load ground truth shifts from CSV.

    Args:
        csv_path: Path to shifts.csv produced by annotate_shifts.py.

    Returns:
        list of dicts with keys: frame_a, frame_b, shift_x, shift_y, skipped.
    """
    pairs = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['skipped'].strip().lower() == 'true':
                continue
            pairs.append({
                'frame_a': int(row['frame_a']),
                'frame_b': int(row['frame_b']),
                'shift_x': int(row['shift_x']),
                'shift_y': int(row['shift_y']),
            })
    return pairs


# ---------- Algorithm implementations ----------

def run_moco_fixed(frame_a, frame_b, mc):
    """Run moco-py in fixed-template mode.

    Returns:
        (shift_x, shift_y) as integers.
    """
    _, shifts, _ = mc.correct_stack(
        frame_b[np.newaxis, ...], frame_a, return_rms=True
    )
    return shifts[0][0], shifts[0][1]


def run_moco_sequential(frame_a, frame_b, mc):
    """Run moco-py frame-to-frame (same as fixed for a single pair).

    For a pair (A, B), sequential is identical to fixed with A as template.
    Included for API completeness — the difference matters for chains.

    Returns:
        (shift_x, shift_y) as integers.
    """
    return run_moco_fixed(frame_a, frame_b, mc)


def run_phase_correlate(frame_a, frame_b):
    """Run cv2.phaseCorrelate (FFT-based, sub-pixel).

    Returns:
        (shift_x, shift_y) as floats.
    """
    a = frame_a.astype(np.float64)
    b = frame_b.astype(np.float64)
    (dx, dy), response = cv2.phaseCorrelate(a, b)
    return dx, dy


def run_ecc_translation(frame_a, frame_b):
    """Run cv2.findTransformECC with MOTION_TRANSLATION.

    Returns:
        (shift_x, shift_y) as floats, or (nan, nan) on failure.
    """
    a = frame_a.astype(np.float32)
    b = frame_b.astype(np.float32)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, warp_matrix = cv2.findTransformECC(
            a, b, warp_matrix, cv2.MOTION_TRANSLATION, criteria
        )
        return float(warp_matrix[0, 2]), float(warp_matrix[1, 2])
    except cv2.error:
        return float('nan'), float('nan')


def run_ecc_euclidean(frame_a, frame_b):
    """Run cv2.findTransformECC with MOTION_EUCLIDEAN (adds rotation).

    Returns:
        (shift_x, shift_y, angle_deg) as floats, or (nan, nan, nan) on failure.
    """
    a = frame_a.astype(np.float32)
    b = frame_b.astype(np.float32)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, warp_matrix = cv2.findTransformECC(
            a, b, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
        )
        dx = float(warp_matrix[0, 2])
        dy = float(warp_matrix[1, 2])
        angle = float(np.degrees(np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])))
        return dx, dy, angle
    except cv2.error:
        return float('nan'), float('nan'), float('nan')


# ---------- Capillary-aware algorithms ----------

def enhance_clahe(image, clip_limit=3.0, grid_size=(8, 8)):
    """Apply CLAHE contrast enhancement.

    Args:
        image: uint8 grayscale image.
        clip_limit: CLAHE clip limit.
        grid_size: Tile grid size for CLAHE.

    Returns:
        CLAHE-enhanced uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def run_orb_match(frame_a, frame_b):
    """ORB feature matching with RANSAC translation estimation.

    Detects keypoints on CLAHE-enhanced images, matches them with a ratio
    test, and estimates a rigid translation via RANSAC.

    Returns:
        (shift_x, shift_y) as floats, or (nan, nan) on failure.
    """
    a = enhance_clahe(frame_a)
    b = enhance_clahe(frame_b)

    orb = cv2.ORB_create(nfeatures=2000)
    kp_a, des_a = orb.detectAndCompute(a, None)
    kp_b, des_b = orb.detectAndCompute(b, None)

    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return float('nan'), float('nan')

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_a, des_b, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        return float('nan'), float('nan')

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(pts_b, pts_a, method=cv2.RANSAC)

    if M is None:
        return float('nan'), float('nan')

    return float(M[0, 2]), float(M[1, 2])


def run_clahe_phase_correlate(frame_a, frame_b):
    """Phase correlate on CLAHE-enhanced + Hanning-windowed images.

    Boosts capillary contrast before running FFT cross-correlation,
    with a Hanning window to reduce edge effects.

    Returns:
        (shift_x, shift_y) as floats.
    """
    a = enhance_clahe(frame_a).astype(np.float64)
    b = enhance_clahe(frame_b).astype(np.float64)
    window = cv2.createHanningWindow(a.shape[::-1], cv2.CV_64F)
    (dx, dy), _ = cv2.phaseCorrelate(a * window, b * window)
    return dx, dy


def run_masked_phase_correlate(frame_a, frame_b):
    """Phase correlate on masked (capillary-only) images.

    Zeros out background using adaptive thresholding so only capillary
    regions contribute to the FFT correlation.

    Returns:
        (shift_x, shift_y) as floats.
    """
    a_enh = enhance_clahe(frame_a)
    b_enh = enhance_clahe(frame_b)

    mask = cv2.adaptiveThreshold(
        a_enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 5
    )

    a_masked = a_enh.astype(np.float64) * (mask / 255.0)
    b_masked = b_enh.astype(np.float64) * (mask / 255.0)

    (dx, dy), _ = cv2.phaseCorrelate(a_masked, b_masked)
    return dx, dy


# ---------- Main benchmark ----------

def benchmark(stack, gt_pairs, registration_crop=0.5):
    """Run all algorithms on ground truth pairs and collect results.

    Args:
        stack: (N, H, W) uint8 image stack.
        gt_pairs: list of dicts from load_ground_truth().
        registration_crop: Fraction for moco-py center crop.

    Returns:
        dict mapping algorithm name to list of per-pair result dicts.
    """
    mc = MotionCorrector(max_shift=50, crop_edges=False,
                         registration_crop=registration_crop)

    algorithms = {
        'moco-py fixed': lambda a, b: run_moco_fixed(a, b, mc),
        'moco-py sequential': lambda a, b: run_moco_sequential(a, b, mc),
        'phaseCorrelate': run_phase_correlate,
        'ECC translation': run_ecc_translation,
        'ECC euclidean': lambda a, b: run_ecc_euclidean(a, b)[:2],
        'ORB+RANSAC': run_orb_match,
        'CLAHE+phaseCorr': run_clahe_phase_correlate,
        'masked phaseCorr': run_masked_phase_correlate,
    }

    all_results = {}
    ecc_angles = []  # Track rotation from euclidean mode

    for alg_name, alg_fn in algorithms.items():
        print(f"  Running {alg_name}...")
        results = []
        total_time = 0.0

        for pair in gt_pairs:
            fa = stack[pair['frame_a']]
            fb = stack[pair['frame_b']]
            gt_x, gt_y = pair['shift_x'], pair['shift_y']

            t0 = time.perf_counter()
            out = alg_fn(fa, fb)
            elapsed = time.perf_counter() - t0

            pred_x, pred_y = out[0], out[1]
            err_x = pred_x - gt_x
            err_y = pred_y - gt_y
            err_mag = np.sqrt(err_x**2 + err_y**2)

            results.append({
                'frame_a': pair['frame_a'],
                'frame_b': pair['frame_b'],
                'gt_x': gt_x,
                'gt_y': gt_y,
                'pred_x': pred_x,
                'pred_y': pred_y,
                'err_x': err_x,
                'err_y': err_y,
                'err_mag': err_mag,
                'time_s': elapsed,
            })
            total_time += elapsed

            # Collect rotation angle from euclidean mode
            if alg_name == 'ECC euclidean':
                _, _, angle = run_ecc_euclidean(fa, fb)
                ecc_angles.append(angle)

        all_results[alg_name] = results

    # Report rotation info if any
    if ecc_angles:
        valid_angles = [a for a in ecc_angles if not np.isnan(a)]
        if valid_angles:
            print(f"\n  ECC Euclidean detected rotation:")
            print(f"    Mean: {np.mean(valid_angles):.4f} deg")
            print(f"    Max:  {np.max(np.abs(valid_angles)):.4f} deg")
            if np.max(np.abs(valid_angles)) < 0.1:
                print(f"    -> Negligible rotation, translation-only is fine")

    return all_results


def print_summary_table(all_results):
    """Print a formatted comparison table to console.

    Args:
        all_results: dict from benchmark().
    """
    print()
    print("=" * 85)
    header = (f"{'Algorithm':<22} | {'Mean Err (px)':>13} | {'Max Err (px)':>12} | "
              f"{'Sub-pixel?':>10} | {'Time/pair':>9}")
    print(header)
    print("-" * 85)

    for alg_name, results in all_results.items():
        errs = [r['err_mag'] for r in results if not np.isnan(r['err_mag'])]
        times = [r['time_s'] for r in results]

        if errs:
            mean_err = np.mean(errs)
            max_err = np.max(errs)
        else:
            mean_err = float('nan')
            max_err = float('nan')

        mean_time = np.mean(times) if times else 0.0
        subpixel = 'No' if 'moco' in alg_name else 'Yes'
        n_fail = sum(1 for r in results if np.isnan(r['err_mag']))
        fail_str = f" ({n_fail} fail)" if n_fail > 0 else ""

        print(f"{alg_name:<22} | {mean_err:>13.2f} | {max_err:>12.2f} | "
              f"{subpixel:>10} | {mean_time:>8.3f}s{fail_str}")

    print("=" * 85)


def save_results_csv(all_results, output_path):
    """Save detailed per-pair results to CSV.

    Args:
        all_results: dict from benchmark().
        output_path: Path to write CSV.
    """
    fieldnames = ['algorithm', 'frame_a', 'frame_b', 'gt_x', 'gt_y',
                  'pred_x', 'pred_y', 'err_x', 'err_y', 'err_mag', 'time_s']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for alg_name, results in all_results.items():
            for r in results:
                row = dict(r)
                row['algorithm'] = alg_name
                # Format floats
                for key in ['pred_x', 'pred_y', 'err_x', 'err_y', 'err_mag']:
                    if isinstance(row[key], float):
                        row[key] = f'{row[key]:.4f}'
                row['time_s'] = f'{row["time_s"]:.6f}'
                writer.writerow(row)

    print(f"\n  Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark registration algorithms against ground truth"
    )
    parser.add_argument(
        "video",
        help="Video folder name (e.g. vid08)"
    )
    parser.add_argument(
        "--registration-crop", type=float, default=0.5,
        help="Fraction of image center for moco-py (default: 0.5)"
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

    gt_csv = os.path.join(video_folder, "ground_truth", "shifts.csv")
    if not os.path.isfile(gt_csv):
        print(f"Error: ground truth not found: {gt_csv}")
        print(f"Run annotate_shifts.py first:")
        print(f"  python scripts/annotate_shifts.py {args.video}")
        sys.exit(1)

    # Load ground truth
    print(f"[{args.video}] Loading ground truth...")
    gt_pairs = load_ground_truth(gt_csv)
    print(f"  {len(gt_pairs)} annotated pairs (skipped pairs excluded)")

    if not gt_pairs:
        print("Error: no valid (non-skipped) pairs in ground truth CSV")
        sys.exit(1)

    # Load frames
    print(f"  Loading frames...")
    stack, _ = load_tiff_stack(video_folder)

    # Run benchmark
    print(f"\n  Benchmarking {len(gt_pairs)} pairs across 8 algorithms...")
    all_results = benchmark(stack, gt_pairs,
                            registration_crop=args.registration_crop)

    # Print summary
    print_summary_table(all_results)

    # Save detailed CSV
    output_csv = os.path.join(video_folder, "ground_truth", "benchmark_results.csv")
    save_results_csv(all_results, output_csv)


if __name__ == "__main__":
    main()
