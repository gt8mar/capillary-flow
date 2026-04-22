"""
Create a median-background-subtracted MP4 from a TIFF image sequence.

The output video is encoded as a signed-difference visualization:
background pixels are gray, brighter-than-background pixels are lighter,
and darker-than-background pixels are darker. This preserves negative
differences that would otherwise be clipped in an 8-bit MP4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a median background and MP4 from background-subtracted TIFF frames."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing numbered TIFF frames.",
    )
    parser.add_argument(
        "--pattern",
        default="*.tif",
        help="Glob pattern for input frames. Default: *.tif",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Output MP4 path. Default: <input_dir>_median_subtracted.mp4",
    )
    parser.add_argument(
        "--background-output",
        type=Path,
        default=None,
        help="Output median background TIFF path. Default: <input_dir>_median_background.tif",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video frame rate. Default: 30",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Symmetric percentile for contrast scaling signed differences. Default: 99",
    )
    return parser.parse_args()


def read_grayscale_frame(path: Path) -> np.ndarray:
    frame = tifffile.imread(str(path))
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] in (3, 4):
        return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported frame shape for {path}: {frame.shape}")


def normalize_to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame

    low, high = np.percentile(frame, (1, 99))
    if high <= low:
        return np.zeros(frame.shape, dtype=np.uint8)
    return np.clip((frame - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def open_mp4_writer(output_path: Path, fps: float, width: int, height: int) -> tuple[cv2.VideoWriter, str]:
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError("Could not open an MP4 writer with avc1, H264, or mp4v.")


def signed_difference_to_display(diff: np.ndarray, scale: float) -> np.ndarray:
    display = 128.0 + (diff.astype(np.float32) * 127.0 / scale)
    return np.clip(display, 0, 255).astype(np.uint8)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_video = args.output_video or input_dir.with_name(f"{input_dir.name}_median_subtracted.mp4")
    background_output = args.background_output or input_dir.with_name(f"{input_dir.name}_median_background.tif")

    frames = sorted(input_dir.glob(args.pattern))
    if not frames:
        raise FileNotFoundError(f"No frames matching {args.pattern!r} found in {input_dir}")

    print(f"Reading {len(frames)} frames from {input_dir}")
    stack = np.stack([normalize_to_uint8(read_grayscale_frame(path)) for path in frames], axis=0)
    background = np.median(stack, axis=0).astype(np.uint8)
    tifffile.imwrite(str(background_output), background)

    diffs = stack.astype(np.int16) - background.astype(np.int16)
    scale = float(np.percentile(np.abs(diffs), args.percentile))
    if scale <= 0:
        scale = 1.0

    height, width = background.shape
    writer, codec = open_mp4_writer(output_video, args.fps, width, height)
    try:
        for diff in diffs:
            display = signed_difference_to_display(diff, scale)
            writer.write(cv2.cvtColor(display, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()

    print(f"Saved background: {background_output}")
    print(f"Saved video: {output_video}")
    print(f"codec={codec} frames={len(frames)} fps={args.fps:g} size={width}x{height} scale={scale:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
