"""
Plot Frog Color Metrics

This script creates visualizations of a single frog image using three key color metrics:
- Red ratio (R / (R + G + B))
- Red dominance (R - (G + B)/2)  
- Mean intensity (R + G + B)

The script loads the first available frog image from the dataset and creates heatmaps
showing the spatial distribution of these metrics across the frog.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from src.config import get_paths
import rawpy

# Get paths at module level
PATHS = get_paths()

def load_image(image_path):
    """Load an image file, supporting both regular formats and RAW files.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (numpy.ndarray, numpy.ndarray) - The loaded image in BGR format (for OpenCV) and RGB format
    """
    # Check file extension
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()
    
    if ext in ['.cr2', '.cr3', '.nef', '.arw', '.dng']:  # RAW formats
        try:
            print(f"Loading RAW file: {image_path}")
            # Use rawpy to open and process the RAW file
            with rawpy.imread(image_path) as raw:
                # Get the raw data with minimal processing
                rgb = raw.postprocess(
                    use_camera_wb=False,  # Don't use camera white balance
                    no_auto_bright=True,  # Don't auto-adjust brightness
                    output_bps=16,        # 16-bit output
                    gamma=(1, 1),         # Linear gamma (no gamma correction)
                    user_flip=0           # Don't flip the image
                )
                
                # Convert to 8-bit for consistency with JPEG processing
                rgb = (rgb / 256).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV compatibility
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                return bgr, rgb
        except Exception as e:
            print(f"Error loading RAW file: {e}")
            return None, None
    else:  # Regular image formats (JPEG, PNG, etc.)
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"Could not load image from {image_path}")
            return None, None
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return bgr, rgb

def load_mask(mask_path):
    """Load a binary mask from a file.
    
    Args:
        mask_path (str): Path to the mask file
        
    Returns:
        numpy.ndarray: Binary mask (True/False)
    """
    if not os.path.exists(mask_path):
        return None
    
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask_image > 0


def check_and_fix_orientation(image_rgb, frog_mask):
    """Check if masks need to be rotated to match image orientation.
    
    This function tries different rotations to find the best match between
    image and mask dimensions.
    
    Parameters
    ----------
    image_rgb : np.ndarray
        The RGB image
    frog_mask : np.ndarray
        The frog mask (binary)
        
    Returns
    -------
    np.ndarray
        Corrected frog mask
    """
    image_height, image_width = image_rgb.shape[:2]
    mask_height, mask_width = frog_mask.shape
    
    print(f"[check_and_fix_orientation] Image shape: {(image_height, image_width)}")
    print(f"[check_and_fix_orientation] Mask shape: {(mask_height, mask_width)}")
    
    # If dimensions are swapped, try rotating the mask
    if (image_height, image_width) == (mask_width, mask_height):
        print("[check_and_fix_orientation] Dimensions are swapped - rotating mask 90 degrees")
        # Rotate mask 90 degrees counterclockwise
        corrected_frog_mask = np.rot90(frog_mask, k=1)
        return corrected_frog_mask
    
    # If exact match, no rotation needed
    elif (image_height, image_width) == (mask_height, mask_width):
        print("[check_and_fix_orientation] Dimensions match - no rotation needed")
        return frog_mask
    
    # If neither exact match nor simple swap, need to resize
    else:
        print(f"[check_and_fix_orientation] Different aspect ratios - will need resizing")
        return frog_mask


def save_alignment_check_images(image_rgb, frog_mask_original, frog_mask_corrected, base_filename, results_dir):
    """Save before/after alignment images for visual verification.
    
    Parameters
    ----------
    image_rgb : np.ndarray
        The original RGB image
    frog_mask_original : np.ndarray
        Original frog mask before correction
    frog_mask_corrected : np.ndarray
        Corrected frog mask after alignment
    base_filename : str
        Base filename for saving
    results_dir : str
        Directory to save alignment check images
    """
    alignment_dir = os.path.join(results_dir, "alignment-checks")
    os.makedirs(alignment_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    
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
    
    plt.suptitle(f'Alignment Check: {base_filename}', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(alignment_dir, f'{base_filename}_alignment_check.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[save_alignment_check_images] Saved alignment check: {save_path}")

def compute_color_metrics(image_rgb, mask=None):
    """Compute the three color metrics for the image.
    
    Args:
        image_rgb (numpy.ndarray): RGB image
        mask (numpy.ndarray, optional): Binary mask to restrict computation
        
    Returns:
        tuple: (red_ratio, red_dominance, mean_intensity) arrays
    """
    # Extract RGB channels
    r_channel = image_rgb[:, :, 0].astype(float)
    g_channel = image_rgb[:, :, 1].astype(float)
    b_channel = image_rgb[:, :, 2].astype(float)
    
    # 1. Red ratio: R / (R + G + B)
    rgb_sum = r_channel + g_channel + b_channel
    rgb_sum[rgb_sum == 0] = 1  # Avoid division by zero
    red_ratio = r_channel / rgb_sum
    
    # 2. Red dominance: R - (G + B)/2
    red_dominance = r_channel - (g_channel + b_channel) / 2
    
    # 3. Mean intensity: R + G + B
    mean_intensity = r_channel + g_channel + b_channel
    
    # Apply mask if provided
    if mask is not None:
        red_ratio_masked = np.zeros_like(red_ratio)
        red_dominance_masked = np.zeros_like(red_dominance)
        mean_intensity_masked = np.zeros_like(mean_intensity)
        
        red_ratio_masked[mask] = red_ratio[mask]
        red_dominance_masked[mask] = red_dominance[mask]
        mean_intensity_masked[mask] = mean_intensity[mask]
        
        return red_ratio_masked, red_dominance_masked, mean_intensity_masked
    
    return red_ratio, red_dominance, mean_intensity

def find_first_frog(raw_jpg_dir, raw_cr2_dir, frog_seg_dir):
    """Find the first frog that has both an image and a mask.
    
    Args:
        raw_jpg_dir (str): Directory containing JPG images
        raw_cr2_dir (str): Directory containing CR2 images
        frog_seg_dir (str): Directory containing frog masks
        
    Returns:
        tuple: (image_path, mask_path, base_filename) or (None, None, None) if not found
    """
    # Get all available mask files
    if not os.path.isdir(frog_seg_dir):
        print(f"Frog segmentation directory not found: {frog_seg_dir}")
        return None, None, None
    
    mask_files = [f for f in os.listdir(frog_seg_dir) if f.endswith("_mask.png")]
    if not mask_files:
        print(f"No mask files found in {frog_seg_dir}")
        return None, None, None
    
    # Try to find corresponding image for each mask
    for mask_file in sorted(mask_files):
        base_filename = mask_file.replace("_mask.png", "")
        mask_path = os.path.join(frog_seg_dir, mask_file)
        
        # Look for corresponding image
        candidate_paths = [
            os.path.join(raw_jpg_dir, base_filename + ext)
            for ext in [".jpg", ".JPG", ".jpeg", ".png"]
        ] + [
            os.path.join(raw_cr2_dir, base_filename + ext)
            for ext in [".cr2", ".CR2", ".cr3", ".CR3"]
        ]
        
        for image_path in candidate_paths:
            if os.path.exists(image_path):
                print(f"Found matching pair: {base_filename}")
                return image_path, mask_path, base_filename
    
    print("No matching image-mask pairs found")
    return None, None, None

def plot_frog_metrics(image_rgb, mask, red_ratio, red_dominance, mean_intensity, 
                     base_filename, output_dir):
    """Create a visualization of the frog with the three color metrics.
    
    Args:
        image_rgb (numpy.ndarray): Original RGB image
        mask (numpy.ndarray): Binary mask of the frog
        red_ratio (numpy.ndarray): Red ratio values
        red_dominance (numpy.ndarray): Red dominance values
        mean_intensity (numpy.ndarray): Mean intensity values
        base_filename (str): Base filename for saving
        output_dir (str): Directory to save the plot
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Frog Color Metrics: {base_filename}', fontsize=16, fontweight='bold')
    
    # Original image with mask overlay
    axes[0, 0].imshow(image_rgb)
    # Add mask overlay
    mask_overlay = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4))
    mask_overlay[mask] = [1, 0, 0, 0.3]  # Red with 30% transparency
    axes[0, 0].imshow(mask_overlay)
    axes[0, 0].set_title('Original Image with Mask')
    axes[0, 0].axis('off')
    
    # Red ratio heatmap
    im1 = axes[0, 1].imshow(red_ratio, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('Red Ratio (R / (R+G+B))')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Red dominance heatmap
    # Normalize red dominance for better visualization
    red_dom_min = np.min(red_dominance[mask]) if np.any(mask) else np.min(red_dominance)
    red_dom_max = np.max(red_dominance[mask]) if np.any(mask) else np.max(red_dominance)
    
    im2 = axes[1, 0].imshow(red_dominance, cmap='RdBu_r', vmin=red_dom_min, vmax=red_dom_max)
    axes[1, 0].set_title('Red Dominance (R - (G+B)/2)')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Mean intensity heatmap
    im3 = axes[1, 1].imshow(mean_intensity, cmap='viridis', vmin=0, vmax=765)  # Max is 3*255
    axes[1, 1].set_title('Mean Intensity (R + G + B)')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{base_filename}_color_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved color metrics plot: {save_path}")
    
    plt.show()

def print_metric_statistics(red_ratio, red_dominance, mean_intensity, mask, base_filename):
    """Print summary statistics for the three metrics.
    
    Args:
        red_ratio (numpy.ndarray): Red ratio values
        red_dominance (numpy.ndarray): Red dominance values
        mean_intensity (numpy.ndarray): Mean intensity values
        mask (numpy.ndarray): Binary mask of the frog
        base_filename (str): Base filename for identification
    """
    print(f"\n=== Color Metrics Statistics for {base_filename} ===")
    
    if np.any(mask):
        # Statistics for masked (frog) region
        red_ratio_frog = red_ratio[mask]
        red_dominance_frog = red_dominance[mask]
        mean_intensity_frog = mean_intensity[mask]
        
        print(f"\nFrog Region Statistics:")
        print(f"Red Ratio:      Mean={np.mean(red_ratio_frog):.3f}, Std={np.std(red_ratio_frog):.3f}")
        print(f"Red Dominance:  Mean={np.mean(red_dominance_frog):.3f}, Std={np.std(red_dominance_frog):.3f}")
        print(f"Mean Intensity: Mean={np.mean(mean_intensity_frog):.1f}, Std={np.std(mean_intensity_frog):.1f}")
        
        # Statistics for background region
        background_mask = ~mask
        if np.any(background_mask):
            red_ratio_bg = red_ratio[background_mask]
            red_dominance_bg = red_dominance[background_mask]
            mean_intensity_bg = mean_intensity[background_mask]
            
            print(f"\nBackground Region Statistics:")
            print(f"Red Ratio:      Mean={np.mean(red_ratio_bg):.3f}, Std={np.std(red_ratio_bg):.3f}")
            print(f"Red Dominance:  Mean={np.mean(red_dominance_bg):.3f}, Std={np.std(red_dominance_bg):.3f}")
            print(f"Mean Intensity: Mean={np.mean(mean_intensity_bg):.1f}, Std={np.std(mean_intensity_bg):.1f}")
            
            # Comparison
            print(f"\nFrog vs Background Comparison:")
            print(f"Red Ratio Difference:      {np.mean(red_ratio_frog) - np.mean(red_ratio_bg):.3f}")
            print(f"Red Dominance Difference:  {np.mean(red_dominance_frog) - np.mean(red_dominance_bg):.3f}")
            print(f"Mean Intensity Difference: {np.mean(mean_intensity_frog) - np.mean(mean_intensity_bg):.1f}")
    else:
        print("No frog mask found - showing whole image statistics:")
        print(f"Red Ratio:      Mean={np.mean(red_ratio):.3f}, Std={np.std(red_ratio):.3f}")
        print(f"Red Dominance:  Mean={np.mean(red_dominance):.3f}, Std={np.std(red_dominance):.3f}")
        print(f"Mean Intensity: Mean={np.mean(mean_intensity):.1f}, Std={np.std(mean_intensity):.1f}")

def main():
    """Main function to plot color metrics for the first available frog."""
    # Set up directories
    RAW_JPG_DIR = r"H:\WkSleep_Trans_Up_to_25-5-1_Named_JPG"
    RAW_CR2_DIR = r"H:\WkSleep_Trans_Up_to_25-5-1_Named_cr2"
    FROG_SEG_DIR = PATHS.get('frog_segmented', os.path.join(PATHS['frog_dir'], 'segmented'))
    
    # Set up output directory
    results_root = PATHS.get("frog_results", os.path.join(PATHS["frog_dir"], "results"))
    output_dir = os.path.join(results_root, "color-metrics")
    
    print("Looking for the first available frog...")
    
    # Find the first frog with both image and mask
    image_path, mask_path, base_filename = find_first_frog(RAW_JPG_DIR, RAW_CR2_DIR, FROG_SEG_DIR)
    
    if image_path is None:
        print("❌ No frog found with both image and mask files")
        print(f"Checked directories:")
        print(f"  - Images: {RAW_JPG_DIR}")
        print(f"  - Images: {RAW_CR2_DIR}")
        print(f"  - Masks:  {FROG_SEG_DIR}")
        return
    
    print(f"📸 Loading image: {image_path}")
    print(f"🎭 Loading mask:  {mask_path}")
    
    # Load the image
    image_bgr, image_rgb = load_image(image_path)
    if image_bgr is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    # Load the mask
    mask = load_mask(mask_path)
    if mask is None:
        print(f"❌ Failed to load mask: {mask_path}")
        return
    
    # Store original mask for alignment checking
    mask_original = mask.copy()
    
    # Check and fix orientation issues (rotation)
    mask = check_and_fix_orientation(image_rgb, mask)
    
    # Resize if still needed after rotation
    image_height, image_width = image_rgb.shape[:2]
    if mask.shape != (image_height, image_width):
        print(f"Resizing mask from {mask.shape} to {(image_height, image_width)}")
        mask = cv2.resize(mask.astype(np.uint8), (image_width, image_height), 
                         interpolation=cv2.INTER_NEAREST) > 0
    
    # Save alignment check images
    save_alignment_check_images(image_rgb, mask_original, mask, base_filename, output_dir)
    
    print(f"🔍 Computing color metrics...")
    
    # Compute the three color metrics
    red_ratio, red_dominance, mean_intensity = compute_color_metrics(image_rgb, mask)
    
    # Print statistics
    print_metric_statistics(red_ratio, red_dominance, mean_intensity, mask, base_filename)
    
    # Create the visualization
    print(f"📊 Creating visualization...")
    plot_frog_metrics(image_rgb, mask, red_ratio, red_dominance, mean_intensity, 
                     base_filename, output_dir)
    
    print(f"✅ Analysis complete for {base_filename}")

if __name__ == "__main__":
    main() 