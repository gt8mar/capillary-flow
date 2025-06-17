import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import rawpy
from src.config import get_paths
from typing import Optional

PATHS = get_paths()

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def load_image(image_path):
    """Load an image from *image_path*.

    Supports common image formats (JPG, PNG) as well as Canon/Nikon RAW files
    such as CR2, CR3, NEF, ARW. For RAW images the file is converted to 8-bit
    RGB using *rawpy*.

    Parameters
    ----------
    image_path : str
        Absolute path to the image file.

    Returns
    -------
    tuple
        (image_bgr, image_rgb) where *image_bgr* is the OpenCV-compatible BGR
        image (uint8) and *image_rgb* is the corresponding RGB image.
    """
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()

    if ext in {".cr2", ".cr3", ".nef", ".arw", ".dng"}:  # RAW formats
        try:
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=False,
                    no_auto_bright=True,
                    output_bps=16,
                    gamma=(1, 1),
                    user_flip=0,
                )
            rgb = (rgb / 256).astype(np.uint8)  # 16-bit -> 8-bit
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            print(f"[load_image] Failed to read RAW file {image_path}: {exc}")
            return None, None
    else:  # Standard formats
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"[load_image] Could not read image {image_path}")
            return None, None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return bgr, rgb


def load_mask(mask_path):
    """Load a binary mask from *mask_path*.

    The mask can be any single-channel image. Non-zero pixels are treated as
    True.
    """
    if not os.path.exists(mask_path):
        return None
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img > 0


# -----------------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------------

def compute_red_ratio(image_rgb):
    """Return the per-pixel red ratio R / (R+G+B)."""
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)
    total = r + g + b
    total[total == 0] = 1  # avoid division by zero
    return r / total


def analyse_frog_vs_liver(
    base_filename: str,
    results_dir: str,
    raw_jpg_dir: str = r"H:\\WkSleep_Trans_Up_to_25-5-1_Named_JPG",
    raw_cr2_dir: str = r"H:\\WkSleep_Trans_Up_to_25-5-1_Named_cr2",
    liver_seg_dir: str = r"H:\\whole_frog_livers",
    frog_seg_dir: Optional[str] = None,
    plot: bool = False,
):
    """Compare redness in the frog exterior vs. liver mask.

    Parameters
    ----------
    base_filename : str
        Name of the frog file *without* extension (e.g. "frog001").
    results_dir : str
        Directory where plots/results will be saved.
    raw_jpg_dir, raw_cr2_dir : str
        Directories containing the JPG and CR2 images respectively.
    liver_seg_dir : str
        Directory containing liver masks (*{base}*_mask.png).
    frog_seg_dir : str | None
        Directory containing whole-frog masks. If *None* the path from config
        (PATHS['frog_segmented']) is used.
    plot : bool
        Show matplotlib windows interactively.
    """

    frog_seg_dir = frog_seg_dir or PATHS["frog_segmented"]

    # ------------------------------------------------------------------
    # Locate image file
    # ------------------------------------------------------------------
    candidate_paths = [
        os.path.join(raw_jpg_dir, base_filename + ext)
        for ext in (".jpg", ".JPG", ".jpeg", ".png")
    ] + [
        os.path.join(raw_cr2_dir, base_filename + ext)
        for ext in (".cr2", ".CR2", ".cr3", ".CR3")
    ]
    full_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            full_path = p
            break
    if full_path is None:
        print(f"[analyse_frog_vs_liver] Image for {base_filename} not found.")
        return None

    image_bgr, image_rgb = load_image(full_path)
    if image_bgr is None:
        return None

    # ------------------------------------------------------------------
    # Load masks
    # ------------------------------------------------------------------
    frog_mask_path = os.path.join(frog_seg_dir, base_filename + "_mask.png")
    liver_mask_path = os.path.join(liver_seg_dir, base_filename + "_mask.png")

    frog_mask = load_mask(frog_mask_path)
    if frog_mask is None:
        print(f"[analyse_frog_vs_liver] Frog mask missing for {base_filename} -> {frog_mask_path}")
        return None

    liver_mask = load_mask(liver_mask_path)
    if liver_mask is None:
        print(f"[analyse_frog_vs_liver] Liver mask missing for {base_filename}. Continuing without liver analysis.")

    # ------------------------------------------------------------------
    # Compute regions
    # ------------------------------------------------------------------
    red_ratio = compute_red_ratio(image_rgb)

    outside_mask = frog_mask.copy()
    inside_mask = None
    if liver_mask is not None:
        inside_mask = liver_mask & frog_mask  # ensure liver is within frog
        outside_mask &= ~liver_mask

    # Average values
    results = {}
    results["outside_mean_rr"] = float(np.mean(red_ratio[outside_mask])) if np.any(outside_mask) else np.nan
    if inside_mask is not None and np.any(inside_mask):
        results["liver_mean_rr"] = float(np.mean(red_ratio[inside_mask]))
    else:
        results["liver_mean_rr"] = np.nan

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Original image with masks
    ax[0].imshow(image_rgb)
    overlay = np.zeros((*image_rgb.shape[:2], 4))
    overlay[frog_mask] = [0, 1, 0, 0.3]  # green overlay for frog
    if liver_mask is not None:
        overlay[liver_mask] = [1, 0, 0, 0.3]  # red overlay for liver
    ax[0].imshow(overlay)
    ax[0].set_title("Original image with masks")
    ax[0].axis("off")

    # Heatmap of red ratio
    m = ax[1].imshow(red_ratio, cmap="hot")
    ax[1].set_title("Red ratio R/(R+G+B)")
    ax[1].axis("off")
    fig.colorbar(m, ax=ax[1])

    # Bar plot comparison
    bar_labels = ["Outside"]
    bar_values = [results["outside_mean_rr"]]
    if not np.isnan(results["liver_mean_rr"]):
        bar_labels.append("Liver")
        bar_values.append(results["liver_mean_rr"])
    ax[2].bar(bar_labels, bar_values, color=["green", "red"][: len(bar_labels)])
    ax[2].set_ylabel("Mean red ratio")
    ax[2].set_ylim(0, 1)
    ax[2].set_title("Redness comparison")

    fig.tight_layout()
    save_path = os.path.join(results_dir, f"{base_filename}_outside_vs_liver.png")
    fig.savefig(save_path)
    if plot:
        plt.show()
    plt.close(fig)

    # Return dictionary for further processing if desired
    return results


# -----------------------------------------------------------------------------
# CLI execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse frog outside vs liver redness.")
    parser.add_argument("base_filenames", nargs="+", help="List of base filenames (without extension)")
    parser.add_argument("--plot", action="store_true", help="Show interactive plots")
    args = parser.parse_args()

    results_root = PATHS.get("frog_results", os.path.join(PATHS["frog_dir"], "results"))
    liver_results_dir = os.path.join(results_root, "liver-comparison")

    for base in args.base_filenames:
        analyse_frog_vs_liver(base, liver_results_dir, plot=args.plot) 