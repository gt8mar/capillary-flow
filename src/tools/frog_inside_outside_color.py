import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import rawpy
from src.config import get_paths
from typing import Optional
import pandas as pd

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
                # Try different flip settings to match mask orientation
                rgb = raw.postprocess(
                    use_camera_wb=False,
                    no_auto_bright=True,
                    output_bps=16,
                    gamma=(1, 1),
                    user_flip=None,  # Let rawpy handle orientation automatically
                )
            rgb = (rgb / 256).astype(np.uint8)  # 16-bit -> 8-bit
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            print(f"[load_image] RAW image loaded: {rgb.shape}")
        except Exception as exc:
            print(f"[load_image] Failed to read RAW file {image_path}: {exc}")
            return None, None
    else:  # Standard formats
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"[load_image] Could not read image {image_path}")
            return None, None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        print(f"[load_image] Standard image loaded: {rgb.shape}")

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


def check_and_fix_orientation(image_rgb, frog_mask, liver_mask=None):
    """Check if masks need to be rotated to match image orientation.
    
    This function tries different rotations to find the best match between
    image and mask dimensions, and applies the same rotation to both masks.
    
    Parameters
    ----------
    image_rgb : np.ndarray
        The RGB image
    frog_mask : np.ndarray
        The frog mask (binary)
    liver_mask : np.ndarray, optional
        The liver mask (binary)
        
    Returns
    -------
    tuple
        (corrected_frog_mask, corrected_liver_mask)
    """
    image_height, image_width = image_rgb.shape[:2]
    mask_height, mask_width = frog_mask.shape
    
    print(f"[check_and_fix_orientation] Image shape: {(image_height, image_width)}")
    print(f"[check_and_fix_orientation] Mask shape: {(mask_height, mask_width)}")
    
    # If dimensions are swapped, try rotating the masks
    if (image_height, image_width) == (mask_width, mask_height):
        print("[check_and_fix_orientation] Dimensions are swapped - rotating masks 90 degrees")
        # Rotate masks 90 degrees counterclockwise
        corrected_frog_mask = np.rot90(frog_mask, k=1)
        corrected_liver_mask = np.rot90(liver_mask, k=1) if liver_mask is not None else None
        return corrected_frog_mask, corrected_liver_mask
    
    # If exact match, no rotation needed
    elif (image_height, image_width) == (mask_height, mask_width):
        print("[check_and_fix_orientation] Dimensions match - no rotation needed")
        return frog_mask, liver_mask
    
    # If neither exact match nor simple swap, need to resize
    else:
        print(f"[check_and_fix_orientation] Different aspect ratios - will need resizing")
        return frog_mask, liver_mask


# -----------------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------------

def save_alignment_check_images(
    image_rgb, 
    frog_mask_original, 
    liver_mask_original, 
    frog_mask_corrected, 
    liver_mask_corrected, 
    base_filename, 
    results_dir
):
    """Save before/after alignment images for visual verification.
    
    Parameters
    ----------
    image_rgb : np.ndarray
        The original RGB image
    frog_mask_original : np.ndarray
        Original frog mask before correction
    liver_mask_original : np.ndarray
        Original liver mask before correction (can be None)
    frog_mask_corrected : np.ndarray
        Corrected frog mask after alignment
    liver_mask_corrected : np.ndarray
        Corrected liver mask after alignment (can be None)
    base_filename : str
        Base filename for saving
    results_dir : str
        Directory to save alignment check images
    """
    alignment_dir = os.path.join(results_dir, "alignment-checks")
    os.makedirs(alignment_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: BEFORE alignment
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title(f'Original Image\n{image_rgb.shape}')
    axes[0, 0].axis('off')
    
    # Original frog mask overlay
    axes[0, 1].imshow(image_rgb)
    if frog_mask_original.shape == image_rgb.shape[:2]:
        # If same size, overlay directly
        overlay = np.zeros((*image_rgb.shape[:2], 4))
        overlay[frog_mask_original] = [0, 1, 0, 0.5]  # Green overlay
        axes[0, 1].imshow(overlay)
        axes[0, 1].set_title(f'BEFORE: Frog Mask\n{frog_mask_original.shape} - ALIGNED')
    else:
        # If different size, show separately
        axes[0, 1].imshow(frog_mask_original, cmap='Greens', alpha=0.7)
        axes[0, 1].set_title(f'BEFORE: Frog Mask\n{frog_mask_original.shape} - MISALIGNED')
    axes[0, 1].axis('off')
    
    # Original liver mask overlay
    axes[0, 2].imshow(image_rgb)
    if liver_mask_original is not None:
        if liver_mask_original.shape == image_rgb.shape[:2]:
            overlay = np.zeros((*image_rgb.shape[:2], 4))
            overlay[liver_mask_original] = [1, 0, 0, 0.5]  # Red overlay
            axes[0, 2].imshow(overlay)
            axes[0, 2].set_title(f'BEFORE: Liver Mask\n{liver_mask_original.shape} - ALIGNED')
        else:
            axes[0, 2].imshow(liver_mask_original, cmap='Reds', alpha=0.7)
            axes[0, 2].set_title(f'BEFORE: Liver Mask\n{liver_mask_original.shape} - MISALIGNED')
    else:
        axes[0, 2].text(0.5, 0.5, 'No Liver Mask', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('BEFORE: Liver Mask\nNot Available')
    axes[0, 2].axis('off')
    
    # Bottom row: AFTER alignment
    # Original image (same)
    axes[1, 0].imshow(image_rgb)
    axes[1, 0].set_title(f'Original Image\n{image_rgb.shape}')
    axes[1, 0].axis('off')
    
    # Corrected frog mask overlay
    axes[1, 1].imshow(image_rgb)
    overlay = np.zeros((*image_rgb.shape[:2], 4))
    overlay[frog_mask_corrected] = [0, 1, 0, 0.5]  # Green overlay
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f'AFTER: Frog Mask\n{frog_mask_corrected.shape} - ALIGNED')
    axes[1, 1].axis('off')
    
    # Corrected liver mask overlay
    axes[1, 2].imshow(image_rgb)
    if liver_mask_corrected is not None:
        overlay = np.zeros((*image_rgb.shape[:2], 4))
        overlay[liver_mask_corrected] = [1, 0, 0, 0.5]  # Red overlay
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title(f'AFTER: Liver Mask\n{liver_mask_corrected.shape} - ALIGNED')
    else:
        axes[1, 2].text(0.5, 0.5, 'No Liver Mask', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('AFTER: Liver Mask\nNot Available')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Alignment Check: {base_filename}', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(alignment_dir, f'{base_filename}_alignment_check.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[save_alignment_check_images] Saved alignment check: {save_path}")


def compute_red_ratio(image_rgb):
    """Return the per-pixel red ratio R / (R+G+B)."""
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)
    total = r + g + b
    total[total == 0] = 1  # avoid division by zero
    return r / total


def compute_comprehensive_color_metrics(image_rgb, mask):
    """Compute comprehensive color metrics for a masked region.
    
    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image
    mask : np.ndarray
        Binary mask defining the region
        
    Returns
    -------
    dict
        Dictionary containing various color metrics
    """
    if not np.any(mask):
        return {
            'mean_red': np.nan, 'mean_green': np.nan, 'mean_blue': np.nan,
            'mean_intensity': np.nan, 'red_ratio_rgb': np.nan, 'red_ratio_rg': np.nan,
            'red_minus_green': np.nan, 'red_minus_blue': np.nan, 'green_minus_blue': np.nan,
            'red_dominance': np.nan, 'pixel_count': 0
        }
    
    # Extract RGB channels for the masked region
    r_vals = image_rgb[:, :, 0][mask].astype(np.float32)
    g_vals = image_rgb[:, :, 1][mask].astype(np.float32)
    b_vals = image_rgb[:, :, 2][mask].astype(np.float32)
    
    # Basic statistics
    mean_red = np.mean(r_vals)
    mean_green = np.mean(g_vals)
    mean_blue = np.mean(b_vals)
    mean_intensity = np.mean(r_vals + g_vals + b_vals)
    
    # Ratio calculations
    red_ratio_rgb = mean_red / (mean_red + mean_green + mean_blue)  # R/(R+G+B)
    red_ratio_rg = mean_red / (mean_red + mean_green) if (mean_red + mean_green) > 0 else np.nan  # R/(R+G)
    
    # Difference calculations
    red_minus_green = mean_red - mean_green
    red_minus_blue = mean_red - mean_blue
    green_minus_blue = mean_green - mean_blue
    
    # Red dominance: how much more red than the average of other channels
    red_dominance = mean_red - (mean_green + mean_blue) / 2
    
    return {
        'mean_red': mean_red,
        'mean_green': mean_green,
        'mean_blue': mean_blue,
        'mean_intensity': mean_intensity,
        'red_ratio_rgb': red_ratio_rgb,
        'red_ratio_rg': red_ratio_rg,
        'red_minus_green': red_minus_green,
        'red_minus_blue': red_minus_blue,
        'green_minus_blue': green_minus_blue,
        'red_dominance': red_dominance,
        'pixel_count': np.sum(mask)
    }


def compute_background_metrics(image_rgb, frog_mask):
    """Compute color metrics for the background (non-frog) region.
    
    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image
    frog_mask : np.ndarray
        Binary mask of the frog (background is ~frog_mask)
        
    Returns
    -------
    dict
        Dictionary containing background color metrics
    """
    background_mask = ~frog_mask
    return compute_comprehensive_color_metrics(image_rgb, background_mask)


def analyse_frog_vs_liver(
    base_filename: str,
    results_dir: str,
    raw_jpg_dir: str = r"H:\\WkSleep_Trans_Up_to_25-5-1_Named_JPG",
    raw_cr2_dir: str = r"H:\\WkSleep_Trans_Up_to_25-5-1_Named_cr2",
    liver_seg_dir: str = r"H:\\whole_frog_livers2",
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
    liver_mask_path = os.path.join(liver_seg_dir, base_filename + ".png")

    frog_mask = load_mask(frog_mask_path)
    if frog_mask is None:
        print(f"[analyse_frog_vs_liver] Frog mask missing for {base_filename} -> {frog_mask_path}")
        return None

    liver_mask = load_mask(liver_mask_path)
    if liver_mask is None:
        print(f"[analyse_frog_vs_liver] Liver mask missing for {base_filename}. Continuing without liver analysis.")

    # Store original masks for alignment checking
    frog_mask_original = frog_mask.copy()
    liver_mask_original = liver_mask.copy() if liver_mask is not None else None

    # ------------------------------------------------------------------
    # Check and fix size mismatches
    # ------------------------------------------------------------------
    image_height, image_width = image_rgb.shape[:2]
    
    # First try to fix orientation issues (rotation)
    frog_mask, liver_mask = check_and_fix_orientation(image_rgb, frog_mask, liver_mask)
    
    # Then resize if still needed
    if frog_mask.shape != (image_height, image_width):
        print(f"[analyse_frog_vs_liver] Resizing frog mask from {frog_mask.shape} to {(image_height, image_width)}")
        frog_mask = cv2.resize(frog_mask.astype(np.uint8), (image_width, image_height), interpolation=cv2.INTER_NEAREST) > 0
    
    # Resize liver mask if needed
    if liver_mask is not None and liver_mask.shape != (image_height, image_width):
        print(f"[analyse_frog_vs_liver] Resizing liver mask from {liver_mask.shape} to {(image_height, image_width)}")
        liver_mask = cv2.resize(liver_mask.astype(np.uint8), (image_width, image_height), interpolation=cv2.INTER_NEAREST) > 0

    # Save alignment check images
    save_alignment_check_images(
        image_rgb, 
        frog_mask_original, 
        liver_mask_original, 
        frog_mask, 
        liver_mask, 
        base_filename, 
        results_dir
    )

    # ------------------------------------------------------------------
    # Compute regions
    # ------------------------------------------------------------------
    red_ratio = compute_red_ratio(image_rgb)

    outside_mask = frog_mask.copy()
    inside_mask = None
    if liver_mask is not None:
        inside_mask = liver_mask & frog_mask  # ensure liver is within frog
        outside_mask &= ~liver_mask

    # Compute comprehensive metrics for each region
    outside_metrics = compute_comprehensive_color_metrics(image_rgb, outside_mask)
    liver_metrics = compute_comprehensive_color_metrics(image_rgb, inside_mask) if inside_mask is not None else None
    background_metrics = compute_background_metrics(image_rgb, frog_mask)
    
    # Compile results for CSV
    csv_results = {
        'frog_id': base_filename,
        'image_width': image_rgb.shape[1],
        'image_height': image_rgb.shape[0],
        
        # Outside (frog minus liver) metrics
        'outside_mean_red': outside_metrics['mean_red'],
        'outside_mean_green': outside_metrics['mean_green'], 
        'outside_mean_blue': outside_metrics['mean_blue'],
        'outside_mean_intensity': outside_metrics['mean_intensity'],
        'outside_red_ratio_rgb': outside_metrics['red_ratio_rgb'],
        'outside_red_ratio_rg': outside_metrics['red_ratio_rg'],
        'outside_red_minus_green': outside_metrics['red_minus_green'],
        'outside_red_dominance': outside_metrics['red_dominance'],
        'outside_pixel_count': outside_metrics['pixel_count'],
        
        # Liver metrics
        'liver_mean_red': liver_metrics['mean_red'] if liver_metrics else np.nan,
        'liver_mean_green': liver_metrics['mean_green'] if liver_metrics else np.nan,
        'liver_mean_blue': liver_metrics['mean_blue'] if liver_metrics else np.nan,
        'liver_mean_intensity': liver_metrics['mean_intensity'] if liver_metrics else np.nan,
        'liver_red_ratio_rgb': liver_metrics['red_ratio_rgb'] if liver_metrics else np.nan,
        'liver_red_ratio_rg': liver_metrics['red_ratio_rg'] if liver_metrics else np.nan,
        'liver_red_minus_green': liver_metrics['red_minus_green'] if liver_metrics else np.nan,
        'liver_red_dominance': liver_metrics['red_dominance'] if liver_metrics else np.nan,
        'liver_pixel_count': liver_metrics['pixel_count'] if liver_metrics else 0,
        
        # Background metrics
        'background_mean_red': background_metrics['mean_red'],
        'background_mean_green': background_metrics['mean_green'],
        'background_mean_blue': background_metrics['mean_blue'],
        'background_mean_intensity': background_metrics['mean_intensity'],
        'background_red_ratio_rgb': background_metrics['red_ratio_rgb'],
        'background_red_ratio_rg': background_metrics['red_ratio_rg'],
        'background_red_minus_green': background_metrics['red_minus_green'],
        'background_red_dominance': background_metrics['red_dominance'],
        'background_pixel_count': background_metrics['pixel_count'],
        
        # Comparative metrics
        'liver_vs_outside_red_diff': (liver_metrics['mean_red'] - outside_metrics['mean_red']) if liver_metrics else np.nan,
        'liver_vs_outside_red_ratio_diff': (liver_metrics['red_ratio_rgb'] - outside_metrics['red_ratio_rgb']) if liver_metrics else np.nan,
        'liver_vs_background_red_diff': (liver_metrics['mean_red'] - background_metrics['mean_red']) if liver_metrics else np.nan,
        'outside_vs_background_red_diff': outside_metrics['mean_red'] - background_metrics['mean_red'],
    }

    # Average values (for backward compatibility)
    results = {}
    results["outside_mean_rr"] = outside_metrics['red_ratio_rgb']
    results["liver_mean_rr"] = liver_metrics['red_ratio_rgb'] if liver_metrics else np.nan

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
    save_path = os.path.join(results_dir, f'{base_filename}_outside_vs_liver.png')
    fig.savefig(save_path)
    if plot:
        plt.show()
    plt.close(fig)

    # Return dictionary for further processing if desired
    results['csv_data'] = csv_results
    return results


def find_matching_frogs_with_livers(
    raw_jpg_dir: str = r"H:\\WkSleep_Trans_Up_to_25-5-1_Named_JPG",
    raw_cr2_dir: str = r"H:\\WkSleep_Trans_Up_to_25-5-1_Named_cr2",
    liver_seg_dir: str = r"H:\\whole_frog_livers2",
    frog_seg_dir: Optional[str] = None,
):
    """Find base filenames that have matching raw images, frog masks, AND liver masks.
    
    Parameters
    ----------
    raw_jpg_dir, raw_cr2_dir : str
        Directories containing the JPG and CR2 images respectively.
    liver_seg_dir : str
        Directory containing liver masks (*{base}*_mask.png).
    frog_seg_dir : str | None
        Directory containing whole-frog masks. If *None* the path from config
        (PATHS['frog_segmented']) is used.
        
    Returns
    -------
    list
        List of base filenames that have all required files (image, frog mask, liver mask).
    """
    frog_seg_dir = frog_seg_dir or PATHS["frog_segmented"]
    
    # Get all available raw image base names
    raw_base_names = set()
    
    # Check JPG directory
    if os.path.isdir(raw_jpg_dir):
        for filename in os.listdir(raw_jpg_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(filename)[0]
                raw_base_names.add(base_name)
    
    # Check CR2 directory
    if os.path.isdir(raw_cr2_dir):
        for filename in os.listdir(raw_cr2_dir):
            if filename.lower().endswith(('.cr2', '.cr3')):
                base_name = os.path.splitext(filename)[0]
                raw_base_names.add(base_name)
    
    print(f"Found {len(raw_base_names)} raw images")
    
    # Get all available frog mask base names
    frog_mask_base_names = set()
    if os.path.isdir(frog_seg_dir):
        for filename in os.listdir(frog_seg_dir):
            if filename.endswith("_mask.png"):
                base_name = filename.replace("_mask.png", "")
                frog_mask_base_names.add(base_name)
    
    print(f"Found {len(frog_mask_base_names)} frog masks")
    
    # Get all available liver mask base names
    liver_mask_base_names = set()
    if os.path.isdir(liver_seg_dir):
        for filename in os.listdir(liver_seg_dir):
            if filename.endswith(".png"):
                base_name = filename.replace(".png", "")
                liver_mask_base_names.add(base_name)
    
    print(f"Found {len(liver_mask_base_names)} liver masks")
    
    # Find intersection - frogs that have all three components
    matching_frogs = raw_base_names & frog_mask_base_names & liver_mask_base_names
    
    print(f"Found {len(matching_frogs)} frogs with complete data (image + frog mask + liver mask)")
    
    # Print some examples for verification
    if len(matching_frogs) > 0:
        print("Examples of matching frogs:")
        for i, frog_name in enumerate(sorted(matching_frogs)):
            if i < 5:  # Show first 5 examples
                print(f"  - {frog_name}")
            elif i == 5:
                print(f"  ... and {len(matching_frogs) - 5} more")
                break
    
    return sorted(list(matching_frogs))


# -----------------------------------------------------------------------------
# CLI execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Get all available frog images by scanning the raw image directories
    RAW_JPG_DIR = r"H:\\WkSleep_Trans_Up_to_25-5-1_Named_JPG"
    RAW_CR2_DIR = r"H:\\WkSleep_Trans_Up_to_25-5-1_Named_cr2"
    LIVER_SEG_DIR = r"H:\\whole_frog_livers2"
    FROG_SEG_DIR = PATHS["frog_segmented"]
    
    results_root = PATHS.get("frog_results", os.path.join(PATHS["frog_dir"], "results"))
    liver_results_dir = os.path.join(results_root, "liver-comparison")
    os.makedirs(liver_results_dir, exist_ok=True)
    
    # Collect JPG files first
    jpg_files = []
    if os.path.isdir(RAW_JPG_DIR):
        for file in os.listdir(RAW_JPG_DIR):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                jpg_files.append(file)
    
    # Sort JPG files
    jpg_files.sort()
    
    # Collect CR2 files, but only if no corresponding JPG exists
    cr2_files = []
    if os.path.isdir(RAW_CR2_DIR):
        for file in os.listdir(RAW_CR2_DIR):
            if file.lower().endswith((".cr2", ".cr3")):
                # Check if corresponding JPG exists
                base_name = os.path.splitext(file)[0]
                jpg_exists = any(os.path.splitext(jpg_file)[0] == base_name for jpg_file in jpg_files)
                if not jpg_exists:
                    cr2_files.append(file)
    
    # Sort CR2 files
    cr2_files.sort()
    
    # Combine files in order: JPG first, then CR2
    all_files = jpg_files + cr2_files
    
    # Filter to only include files that have corresponding frog masks AND liver masks
    base_filenames_with_masks = []
    for file in all_files:
        base_filename = os.path.splitext(file)[0]
        frog_mask_path = os.path.join(FROG_SEG_DIR, f"{base_filename}_mask.png")
        liver_mask_path = os.path.join(LIVER_SEG_DIR, f"{base_filename}.png")
        
        if os.path.exists(frog_mask_path) and os.path.exists(liver_mask_path):
            base_filenames_with_masks.append(base_filename)
    
    print(f"Found {len(jpg_files)} JPG files and {len(cr2_files)} CR2 files")
    print(f"Total files to process: {len(all_files)}")
    print(f"Files with both frog and liver masks: {len(base_filenames_with_masks)}")
    
    if len(base_filenames_with_masks) == 0:
        print("No frogs found with complete data (image + frog mask + liver mask)")
        print("Please check that the directories contain matching files.")
        exit(1)
    
    print(f"\nProcessing {len(base_filenames_with_masks)} frogs with liver segmentations...")
    
    # Process each frog and collect CSV data
    successful_analyses = 0
    csv_data_list = []
    
    for i, base_filename in enumerate(base_filenames_with_masks, 1):
        print(f"\n[{i}/{len(base_filenames_with_masks)}] Processing: {base_filename}")
        result = analyse_frog_vs_liver(base_filename, liver_results_dir, plot=False)
        if result is not None:
            successful_analyses += 1
            print(f"  Outside red ratio: {result['outside_mean_rr']:.3f}")
            print(f"  Liver red ratio: {result['liver_mean_rr']:.3f}")
            difference = result['liver_mean_rr'] - result['outside_mean_rr']
            print(f"  Difference (liver - outside): {difference:+.3f}")
            
            # Collect CSV data
            csv_data_list.append(result['csv_data'])
        else:
            print("  Analysis failed")
    
    # Save CSV results
    if csv_data_list:
        csv_df = pd.DataFrame(csv_data_list)
        csv_path = os.path.join(liver_results_dir, "frog_liver_color_analysis.csv")
        csv_df.to_csv(csv_path, index=False)
        print(f"\n✅ CSV results saved to: {csv_path}")
        print(f"   Contains {len(csv_df)} rows and {len(csv_df.columns)} columns")
    
    print(f"\nCompleted {successful_analyses} successful analyses out of {len(base_filenames_with_masks)} total images")
    print(f"Results saved to: {liver_results_dir}") 