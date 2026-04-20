from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def list_image_files(folder: Path) -> List[Path]:
    """Return sorted image paths in a folder using common microscopy/image extensions."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Expected folder path, got file: {folder}")
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def to_grayscale_float(frame: np.ndarray) -> np.ndarray:
    """Convert frame to grayscale float32 without changing spatial dimensions."""
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(np.float32)


def iter_frames(path: Path) -> Iterator[np.ndarray]:
    """
    Yield frames from either:
    - folder of image files, or
    - video readable by OpenCV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_dir():
        files = list_image_files(path)
        if not files:
            raise ValueError(f"No image frames found in folder: {path}")
        for file_path in files:
            frame = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if frame is not None:
                yield frame
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def load_stack_mean(stack_folder: Path) -> np.ndarray:
    """Load all frames from a calibration stack folder and return mean image (float32)."""
    files = list_image_files(stack_folder)
    if not files:
        raise ValueError(f"No calibration images found in: {stack_folder}")

    acc = None
    count = 0
    for file_path in files:
        frame = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            continue
        frame_f = to_grayscale_float(frame)

        if acc is None:
            acc = np.zeros_like(frame_f, dtype=np.float64)
        if frame_f.shape != acc.shape:
            raise ValueError(
                f"Calibration shape mismatch in {stack_folder}. Problem file: {file_path}"
            )

        acc += frame_f
        count += 1

    if acc is None or count == 0:
        raise ValueError(f"No readable calibration frames found in: {stack_folder}")

    return (acc / count).astype(np.float32)


def compute_flatfield_model(
    dark_mean: np.ndarray,
    flat_mean: np.ndarray,
    eps: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Build denominator + scale for dark/flat correction:
        Icorr = (I - D) / (F - D + eps) * mean(F - D)
    """
    if dark_mean.shape != flat_mean.shape:
        raise ValueError("Dark and flat frame shapes do not match.")

    flat_minus_dark = flat_mean - dark_mean
    scale = float(np.mean(flat_minus_dark))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid flat-dark mean. Verify dark/flat capture quality.")

    denom = flat_minus_dark + float(eps)
    return denom.astype(np.float32), scale


def apply_dark_flat_correction(
    frame: np.ndarray,
    dark_mean: np.ndarray,
    flat_denom: np.ndarray,
    flat_scale: float,
) -> np.ndarray:
    """Apply dark/flat correction to one frame and return non-negative float32 image."""
    image = to_grayscale_float(frame)
    if image.shape != dark_mean.shape:
        raise ValueError(
            f"Frame shape {image.shape} does not match calibration shape {dark_mean.shape}."
        )

    corrected = ((image - dark_mean) / flat_denom) * float(flat_scale)
    corrected = np.clip(corrected, 0.0, None)
    return corrected.astype(np.float32)


def load_dark_flat_model(
    dark_folder: Path,
    flat_folder: Path,
    eps: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convenience loader for calibration model.

    Returns:
        dark_mean, flat_denom, flat_scale
    """
    dark_mean = load_stack_mean(dark_folder)
    flat_mean = load_stack_mean(flat_folder)
    flat_denom, flat_scale = compute_flatfield_model(dark_mean, flat_mean, eps=eps)
    return dark_mean, flat_denom, flat_scale
