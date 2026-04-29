"""
Filename: preprocess_basler.py
-------------------------------
Preprocesses Basler camera .tiff frames: stabilizes with moco-py,
then applies contrast enhancement for the capillary-flow pipeline.

Usage:
    python scripts/preprocess_basler.py              # all vid* folders
    python scripts/preprocess_basler.py vid03         # single folder
    python scripts/preprocess_basler.py --force       # reprocess existing
    python scripts/preprocess_basler.py --basler-dir D:\\other\\path

By: Marcus Forst
"""

import argparse
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
from moco_py.io import save_shift_log, shifts_to_results
from src.capillary_contrast import capillary_contrast
from src.tools.get_images import get_images

DEFAULT_BASLER_DIR = "C:\\Users\\gt8ma\\Basler"


def load_tiff_stack(folder):
    """Load individual .tiff files from a folder into a numpy stack.

    Args:
        folder (str): Path to folder containing .tiff frames.

    Returns:
        tuple: (stack, filenames) where stack is (N, H, W) uint8 array
            and filenames is a list of original filenames.
    """
    filenames = get_images(folder, extension='tiff')
    if not filenames:
        raise FileNotFoundError(f"No .tiff files found in {folder}")

    frames = []
    for fname in filenames:
        img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read {os.path.join(folder, fname)}")
        frames.append(img)

    stack = np.array(frames, dtype=np.uint8)
    print(f"  Loaded {len(filenames)} frames, shape {stack.shape}")
    return stack, filenames


def save_corrected_frames(stack, output_folder, filenames):
    """Save each frame of a stack as an individual .tiff file.

    Args:
        stack (numpy.ndarray): (N, H, W) image stack.
        output_folder (str): Destination folder.
        filenames (list): Original filenames to preserve.
    """
    os.makedirs(output_folder, exist_ok=True)
    for i, fname in enumerate(filenames):
        cv2.imwrite(os.path.join(output_folder, fname), stack[i])
    print(f"  Saved {len(filenames)} frames to {output_folder}")


def process_video_folder(video_folder, force=False, registration_crop=None):
    """Run stabilization and contrast enhancement on a single video folder.

    Args:
        video_folder (str): Path to a vid## folder containing .tiff frames.
        force (bool): If True, reprocess even if moco/ already exists.
        registration_crop (float or None): Fraction of image center to use for
            shift computation. None means use the full frame.
    """
    vid_name = os.path.basename(video_folder)
    moco_folder = os.path.join(video_folder, "moco")
    contrast_folder = os.path.join(video_folder, "moco-contrasted")
    shifts_path = os.path.join(video_folder, "shifts.csv")

    # Skip check
    if os.path.isdir(moco_folder) and not force:
        print(f"  Skipping {vid_name}: moco/ already exists (use --force to reprocess)")
        return

    # Load frames
    stack, filenames = load_tiff_stack(video_folder)

    # Motion correction
    print(f"  Running motion correction ({stack.shape[0]} frames)...")
    mc = MotionCorrector(max_shift=50, crop_edges=False, registration_crop=registration_crop)
    corrected_stack, shifts, rms_errors = mc.correct_stack(
        stack, stack[0], return_rms=True
    )

    # Save stabilized frames
    save_corrected_frames(corrected_stack, moco_folder, filenames)

    # Save shift log
    results = shifts_to_results(shifts, rms_errors)
    save_shift_log(results, shifts_path)
    print(f"  Saved shift log to {shifts_path}")

    # Free memory before contrast step
    del stack
    del corrected_stack

    # Contrast enhancement
    print(f"  Running contrast enhancement...")
    capillary_contrast(moco_folder, contrast_folder)

    print(f"  Done: {vid_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Basler .tiff frames: stabilize + contrast enhance"
    )
    parser.add_argument(
        "videos", nargs="*",
        help="Specific vid## folder names to process (default: all vid* folders)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess even if moco/ already exists"
    )
    parser.add_argument(
        "--basler-dir", default=DEFAULT_BASLER_DIR,
        help=f"Path to Basler directory (default: {DEFAULT_BASLER_DIR})"
    )
    parser.add_argument(
        "--registration-crop", type=float, default=0.5,
        help="Fraction of image center for shift computation (default: 0.5)"
    )
    args = parser.parse_args()

    basler_dir = args.basler_dir
    if not os.path.isdir(basler_dir):
        print(f"Error: Basler directory not found: {basler_dir}")
        sys.exit(1)

    # Determine which folders to process
    if args.videos:
        folders = [os.path.join(basler_dir, v) for v in args.videos]
        # Validate
        for f in folders:
            if not os.path.isdir(f):
                print(f"Error: folder not found: {f}")
                sys.exit(1)
    else:
        folders = sorted([
            os.path.join(basler_dir, d) for d in os.listdir(basler_dir)
            if d.startswith("vid") and os.path.isdir(os.path.join(basler_dir, d))
        ])

    if not folders:
        print(f"No vid* folders found in {basler_dir}")
        sys.exit(1)

    print(f"Processing {len(folders)} video folder(s) in {basler_dir}")
    total_start = time.time()

    for folder in folders:
        vid_name = os.path.basename(folder)
        print(f"\n[{vid_name}]")
        vid_start = time.time()
        try:
            process_video_folder(folder, force=args.force,
                                registration_crop=args.registration_crop)
        except Exception as e:
            print(f"  ERROR processing {vid_name}: {e}")
        print(f"  Time: {time.time() - vid_start:.1f}s")

    print(f"\nTotal time: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
