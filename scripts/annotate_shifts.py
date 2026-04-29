"""
Filename: annotate_shifts.py
------------------------------
Interactive tkinter GUI for manually aligning frame pairs to establish
ground truth shifts for stabilization evaluation.

Workflow:
    1. Load a Basler video stack (e.g., vid08)
    2. Auto-detect candidate frame pairs:
       - Jump pairs: top N pairs with highest frame-to-frame pixel diff
       - Stride pairs: sampled at regular intervals to capture slow drift
       - Stable pairs: lowest diff pairs as sanity checks (shift ~ 0)
    3. Present each pair for manual alignment via blend/diff/flicker views
    4. Save ground truth CSV

Controls:
    Arrow keys     +-1 px shift
    Shift+Arrow    +-5 px shift
    d              Difference view (|A - shifted(B)| -- dark = aligned)
    f              Flicker view (alternate A/B -- no jump = aligned)
    b              Blend view (50/50 alpha overlay, default)
    Enter          Confirm shift, advance to next pair
    s              Skip pair (mark as uncertain)
    z              Reset shift to [0, 0]
    Escape         Save and quit

Usage:
    python scripts/annotate_shifts.py vid08
    python scripts/annotate_shifts.py vid01 --n-pairs 10
    python scripts/annotate_shifts.py vid08 --basler-dir D:\\other\\path

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
CROP_SIZE = 600  # Display crop size (pixels, 1:1 resolution)


def select_frame_pairs(stack, n_jump=8, n_stride=5, n_stable=3):
    """Auto-detect candidate frame pairs for annotation.

    Args:
        stack: (N, H, W) uint8 image stack.
        n_jump: Number of high-difference (jump) pairs.
        n_stride: Number of stride-sampled pairs.
        n_stable: Number of low-difference (stable) pairs.

    Returns:
        list of (frame_a_idx, frame_b_idx, category) tuples, sorted by
        frame_a_idx. Categories: 'jump', 'stride', 'stable'.
    """
    n = stack.shape[0]
    if n < 2:
        return []

    # Compute frame-to-frame mean absolute differences
    diffs = []
    for i in range(n - 1):
        d = np.mean(np.abs(stack[i + 1].astype(np.float32) - stack[i].astype(np.float32)))
        diffs.append((i, i + 1, d))

    # Sort by diff descending for jump pairs
    sorted_by_diff = sorted(diffs, key=lambda x: x[2], reverse=True)
    jump_pairs = set()
    for i, j, d in sorted_by_diff:
        if len(jump_pairs) >= n_jump:
            break
        jump_pairs.add((i, j))

    # Sort by diff ascending for stable pairs
    stable_pairs = set()
    for i, j, d in sorted(diffs, key=lambda x: x[2]):
        if len(stable_pairs) >= n_stable:
            break
        if (i, j) not in jump_pairs:
            stable_pairs.add((i, j))

    # Stride pairs: evenly spaced through the video
    stride = max(1, n // (n_stride + 1))
    stride_pairs = set()
    for k in range(n_stride):
        a = k * stride
        b = min(a + stride, n - 1)
        if a != b and (a, b) not in jump_pairs and (a, b) not in stable_pairs:
            stride_pairs.add((a, b))

    # Combine and tag
    pairs = []
    for a, b in jump_pairs:
        pairs.append((a, b, 'jump'))
    for a, b in stride_pairs:
        pairs.append((a, b, 'stride'))
    for a, b in stable_pairs:
        pairs.append((a, b, 'stable'))

    pairs.sort(key=lambda x: x[0])
    return pairs


def auto_seed_shift(frame_a, frame_b, mc):
    """Compute moco-py's shift estimate to pre-populate the GUI.

    Args:
        frame_a: 2D uint8 array (template).
        frame_b: 2D uint8 array (frame to align).
        mc: MotionCorrector instance.

    Returns:
        list: [shift_x, shift_y] from moco-py.
    """
    try:
        _, shifts, _ = mc.correct_stack(
            frame_b[np.newaxis, ...], frame_a, return_rms=True
        )
        return list(shifts[0])
    except Exception:
        return [0, 0]


def center_crop(image, size):
    """Extract a center crop from an image.

    Args:
        image: 2D numpy array.
        size: Crop size in pixels (square).

    Returns:
        2D numpy array of shape (size, size).
    """
    h, w = image.shape[:2]
    cy, cx = h // 2, w // 2
    half = size // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)
    return image[y0:y1, x0:x1]


def shift_image(image, dx, dy):
    """Shift an image by (dx, dy) pixels using affine transform.

    Args:
        image: 2D numpy array.
        dx: Horizontal shift (positive = right).
        dy: Vertical shift (positive = down).

    Returns:
        Shifted 2D numpy array (same size, border filled with 0).
    """
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class ShiftAnnotator:
    """Tkinter GUI for manually aligning frame pairs."""

    def __init__(self, stack, pairs, output_path, mc):
        """
        Args:
            stack: (N, H, W) uint8 image stack.
            pairs: list of (frame_a, frame_b, category) tuples.
            output_path: Path to save ground truth CSV.
            mc: MotionCorrector for auto-seeding shifts.
        """
        import tkinter as tk
        from PIL import Image, ImageTk

        self.tk = tk
        self.Image = Image
        self.ImageTk = ImageTk

        self.stack = stack
        self.pairs = pairs
        self.output_path = output_path
        self.mc = mc
        self.pair_idx = 0
        self.shift_x = 0
        self.shift_y = 0
        self.view_mode = 'blend'  # blend, diff, flicker
        self.flicker_showing_a = True
        self.flicker_job = None
        self.results = []

        # Load any existing results to allow resuming
        self._load_existing_results()

        self.root = tk.Tk()
        self.root.title("Shift Annotator")
        self.root.configure(bg='#2b2b2b')

        # Main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Image display
        self.panel = tk.Label(main_frame, bg='#1a1a1a', relief='sunken', bd=2)
        self.panel.pack(pady=(0, 10))

        # Status bar
        self.status_var = tk.StringVar()
        status_label = tk.Label(
            main_frame, textvariable=self.status_var,
            font=('Consolas', 10), fg='#e0e0e0', bg='#2b2b2b',
            justify='left', anchor='w'
        )
        status_label.pack(fill='x')

        # Help text
        help_text = (
            "Arrow: +/-1px | Shift+Arrow: +/-5px | "
            "b=Blend d=Diff f=Flicker | Enter=Confirm s=Skip z=Reset | Esc=Quit"
        )
        help_label = tk.Label(
            main_frame, text=help_text,
            font=('Consolas', 8), fg='#888888', bg='#2b2b2b'
        )
        help_label.pack(fill='x', pady=(5, 0))

        # Key bindings
        self.root.bind('<Left>', lambda e: self._adjust_shift(-1, 0))
        self.root.bind('<Right>', lambda e: self._adjust_shift(1, 0))
        self.root.bind('<Up>', lambda e: self._adjust_shift(0, -1))
        self.root.bind('<Down>', lambda e: self._adjust_shift(0, 1))
        self.root.bind('<Shift-Left>', lambda e: self._adjust_shift(-5, 0))
        self.root.bind('<Shift-Right>', lambda e: self._adjust_shift(5, 0))
        self.root.bind('<Shift-Up>', lambda e: self._adjust_shift(0, -5))
        self.root.bind('<Shift-Down>', lambda e: self._adjust_shift(0, 5))
        self.root.bind('<Return>', lambda e: self._confirm())
        self.root.bind('<s>', lambda e: self._skip())
        self.root.bind('<z>', lambda e: self._reset_shift())
        self.root.bind('<b>', lambda e: self._set_mode('blend'))
        self.root.bind('<d>', lambda e: self._set_mode('diff'))
        self.root.bind('<f>', lambda e: self._set_mode('flicker'))
        self.root.bind('<Escape>', lambda e: self._quit())

        # Load first pair
        self._load_pair()
        self.root.mainloop()

    def _load_existing_results(self):
        """Load previously annotated results to allow resuming."""
        if os.path.isfile(self.output_path):
            done_pairs = set()
            with open(self.output_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.results.append(row)
                    done_pairs.add((int(row['frame_a']), int(row['frame_b'])))
            # Skip already-annotated pairs
            while (self.pair_idx < len(self.pairs)
                   and (self.pairs[self.pair_idx][0], self.pairs[self.pair_idx][1]) in done_pairs):
                self.pair_idx += 1
            if self.results:
                print(f"  Resuming: {len(self.results)} pairs already annotated, "
                      f"{len(self.pairs) - self.pair_idx} remaining")

    def _load_pair(self):
        """Load the current frame pair and auto-seed the shift."""
        if self.pair_idx >= len(self.pairs):
            self._save_and_finish()
            return

        a_idx, b_idx, category = self.pairs[self.pair_idx]
        frame_a = self.stack[a_idx]
        frame_b = self.stack[b_idx]

        # Auto-seed with moco-py estimate
        seed = auto_seed_shift(frame_a, frame_b, self.mc)
        self.shift_x = seed[0]
        self.shift_y = seed[1]

        self._stop_flicker()
        self.view_mode = 'blend'
        self._update_display()

    def _adjust_shift(self, dx, dy):
        """Adjust the current shift by (dx, dy)."""
        self.shift_x += dx
        self.shift_y += dy
        self._update_display()

    def _reset_shift(self):
        """Reset shift to [0, 0]."""
        self.shift_x = 0
        self.shift_y = 0
        self._update_display()

    def _set_mode(self, mode):
        """Switch view mode."""
        self._stop_flicker()
        self.view_mode = mode
        if mode == 'flicker':
            self._start_flicker()
        self._update_display()

    def _start_flicker(self):
        """Start alternating between frame A and shifted B."""
        self.flicker_showing_a = True
        self._flicker_tick()

    def _flicker_tick(self):
        """Toggle the flicker display."""
        self.flicker_showing_a = not self.flicker_showing_a
        self._update_display()
        self.flicker_job = self.root.after(250, self._flicker_tick)

    def _stop_flicker(self):
        """Stop flicker mode."""
        if self.flicker_job is not None:
            self.root.after_cancel(self.flicker_job)
            self.flicker_job = None

    def _update_display(self):
        """Render the current view and update the GUI."""
        if self.pair_idx >= len(self.pairs):
            return

        a_idx, b_idx, category = self.pairs[self.pair_idx]
        frame_a = self.stack[a_idx]
        frame_b = self.stack[b_idx]

        # Shift frame B
        shifted_b = shift_image(frame_b, self.shift_x, self.shift_y)

        # Center crop for display
        crop_a = center_crop(frame_a, CROP_SIZE)
        crop_b = center_crop(shifted_b, CROP_SIZE)

        # Compute mean abs diff on the crop
        mad = np.mean(np.abs(crop_a.astype(np.float32) - crop_b.astype(np.float32)))

        # Render based on view mode
        if self.view_mode == 'blend':
            display = ((crop_a.astype(np.float32) + crop_b.astype(np.float32)) / 2).astype(np.uint8)
        elif self.view_mode == 'diff':
            diff = np.abs(crop_a.astype(np.float32) - crop_b.astype(np.float32))
            # Scale for visibility
            display = np.clip(diff * 3, 0, 255).astype(np.uint8)
        elif self.view_mode == 'flicker':
            display = crop_a if self.flicker_showing_a else crop_b
        else:
            display = crop_a

        # Convert to PhotoImage
        img = self.Image.fromarray(display)
        photo = self.ImageTk.PhotoImage(img)
        self.panel.config(image=photo)
        self.panel.image = photo

        # Update status
        self.status_var.set(
            f"Pair {self.pair_idx + 1}/{len(self.pairs)}  |  "
            f"Frames: {a_idx} -> {b_idx} ({category})  |  "
            f"Shift: [{self.shift_x:+d}, {self.shift_y:+d}]  |  "
            f"MAD: {mad:.2f}  |  "
            f"View: {self.view_mode}"
        )

    def _confirm(self):
        """Confirm the current shift and advance to the next pair."""
        a_idx, b_idx, category = self.pairs[self.pair_idx]
        frame_a = self.stack[a_idx]
        frame_b = self.stack[b_idx]

        # Compute MAD at final shift
        shifted_b = shift_image(frame_b, self.shift_x, self.shift_y)
        crop_a = center_crop(frame_a, CROP_SIZE)
        crop_b = center_crop(shifted_b, CROP_SIZE)
        mad = float(np.mean(np.abs(crop_a.astype(np.float32) - crop_b.astype(np.float32))))

        self.results.append({
            'frame_a': a_idx,
            'frame_b': b_idx,
            'shift_x': self.shift_x,
            'shift_y': self.shift_y,
            'mean_abs_diff': f'{mad:.4f}',
            'skipped': False,
            'category': category
        })

        print(f"  Confirmed: frames {a_idx}->{b_idx}  "
              f"shift=[{self.shift_x:+d}, {self.shift_y:+d}]  MAD={mad:.2f}")

        self.pair_idx += 1
        self._load_pair()

    def _skip(self):
        """Skip the current pair."""
        a_idx, b_idx, category = self.pairs[self.pair_idx]
        self.results.append({
            'frame_a': a_idx,
            'frame_b': b_idx,
            'shift_x': 0,
            'shift_y': 0,
            'mean_abs_diff': '',
            'skipped': True,
            'category': category
        })
        print(f"  Skipped: frames {a_idx}->{b_idx}")
        self.pair_idx += 1
        self._load_pair()

    def _save_and_finish(self):
        """Save results to CSV and close."""
        self._stop_flicker()
        self._save_csv()
        print(f"\n  All {len(self.results)} pairs annotated.")
        print(f"  Saved to: {self.output_path}")
        self.status_var.set("Done! All pairs annotated. Close window to exit.")

    def _save_csv(self):
        """Write results to CSV."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        fieldnames = ['frame_a', 'frame_b', 'shift_x', 'shift_y',
                      'mean_abs_diff', 'skipped', 'category']
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

    def _quit(self):
        """Save and quit."""
        self._stop_flicker()
        self._save_csv()
        print(f"\n  Saved {len(self.results)} results to: {self.output_path}")
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive GUI for annotating ground truth frame shifts"
    )
    parser.add_argument(
        "video",
        help="Video folder name (e.g. vid08)"
    )
    parser.add_argument(
        "--n-pairs", type=int, default=16,
        help="Total number of frame pairs to annotate (default: 16)"
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help="Only use first N frames (default: all)"
    )
    parser.add_argument(
        "--registration-crop", type=float, default=0.5,
        help="Fraction of image center for moco-py auto-seed (default: 0.5)"
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

    # Output path
    gt_dir = os.path.join(video_folder, "ground_truth")
    output_path = os.path.join(gt_dir, "shifts.csv")

    # Load frames
    print(f"[{args.video}] Loading frames...")
    stack, filenames = load_tiff_stack(video_folder)

    if args.frames is not None and args.frames < stack.shape[0]:
        stack = stack[:args.frames]
        print(f"  Truncated to first {args.frames} frames")

    # Select frame pairs
    n = args.n_pairs
    n_jump = max(1, n // 2)
    n_stride = max(1, n // 4)
    n_stable = max(1, n - n_jump - n_stride)
    pairs = select_frame_pairs(stack, n_jump=n_jump, n_stride=n_stride,
                               n_stable=n_stable)
    print(f"  Selected {len(pairs)} frame pairs "
          f"({n_jump} jump, {n_stride} stride, {n_stable} stable)")

    # Auto-seed motion corrector
    mc = MotionCorrector(max_shift=50, crop_edges=False,
                         registration_crop=args.registration_crop)

    # Launch GUI
    print(f"  Output: {output_path}")
    print(f"  Starting GUI...")
    ShiftAnnotator(stack, pairs, output_path, mc)


if __name__ == "__main__":
    main()
