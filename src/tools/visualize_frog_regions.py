"""
Visualize Frog Regions

This script creates visualizations of the 4 concentric regions used in frog color analysis.
Each region is colored differently to show the spatial distribution from center to edge.

The script can process individual frogs or batch process all available frog images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    """Load a binary mask from mask_path.
    
    Args:
        mask_path (str): Path to the mask file
        
    Returns:
        numpy.ndarray or None: Binary mask or None if not found
    """
    if not os.path.exists(mask_path):
        return None
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img > 0


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


def create_regional_masks(frog_mask):
    """Create 4 concentric regional masks from center to edge.
    
    Args:
        frog_mask (np.ndarray): Binary mask of the frog
        
    Returns:
        list: List of 4 binary masks for regions 1-4 (center to edge)
    """
    # Find the center of the frog
    y_coords, x_coords = np.where(frog_mask)
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)
    
    # Calculate distances from each frog pixel to the center
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    
    # Sort the pixels by distance
    sorted_indices = np.argsort(distances)
    num_pixels = len(sorted_indices)
    
    # Divide into 4 regions
    region_size = num_pixels // 4
    
    # Create masks for each region
    regional_masks = []
    for i in range(4):
        start_idx = i * region_size
        end_idx = (i + 1) * region_size if i < 3 else num_pixels
        region_indices = sorted_indices[start_idx:end_idx]
        region_y = y_coords[region_indices]
        region_x = x_coords[region_indices]
        
        # Create binary mask for this region
        region_mask = np.zeros_like(frog_mask)
        region_mask[region_y, region_x] = True
        regional_masks.append(region_mask)
    
    return regional_masks


def visualize_frog_regions(base_filename, output_dir, 
                          raw_jpg_dir=r"H:\\WkSleep_Trans_Up_to_25-5-1_Named_JPG",
                          raw_cr2_dir=r"H:\\WkSleep_Trans_Up_to_25-5-1_Named_cr2",
                          frog_seg_dir=None, plot=False):
    """Create region visualization for a single frog.
    
    Args:
        base_filename (str): Base filename of the frog (without extension)
        output_dir (str): Directory to save visualization files
        raw_jpg_dir (str): Directory containing JPG images
        raw_cr2_dir (str): Directory containing CR2 images  
        frog_seg_dir (str): Directory containing frog masks (uses config if None)
        plot (bool): Whether to display plots interactively
        
    Returns:
        bool: True if successful, False otherwise
    """
    frog_seg_dir = frog_seg_dir or PATHS["frog_segmented"]
    
    # Find the image file
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
        print(f"❌ Image for {base_filename} not found.")
        return False
    
    # Load the image
    image_bgr, image_rgb = load_image(full_path)
    if image_bgr is None:
        print(f"❌ Failed to load image: {full_path}")
        return False
    
    # Load the frog mask
    frog_mask_path = os.path.join(frog_seg_dir, base_filename + "_mask.png")
    frog_mask = load_mask(frog_mask_path)
    if frog_mask is None:
        print(f"❌ Frog mask not found: {frog_mask_path}")
        return False
    
    # Store original mask for alignment checking
    frog_mask_original = frog_mask.copy()
    
    # Check and fix orientation issues (rotation)
    frog_mask = check_and_fix_orientation(image_rgb, frog_mask)
    
    # Resize if still needed after rotation
    image_height, image_width = image_rgb.shape[:2]
    if frog_mask.shape != (image_height, image_width):
        print(f"Resizing mask from {frog_mask.shape} to {(image_height, image_width)}")
        frog_mask = cv2.resize(frog_mask.astype(np.uint8), (image_width, image_height), 
                              interpolation=cv2.INTER_NEAREST) > 0
    
    # Save alignment check images
    save_alignment_check_images(image_rgb, frog_mask_original, frog_mask, base_filename, output_dir)
    
    # Create regional masks
    regional_masks = create_regional_masks(frog_mask)
    
    # Define colors for each region
    region_colors = [
        [1.0, 0.0, 0.0],  # Red - Region 1 (center)
        [0.0, 1.0, 0.0],  # Green - Region 2
        [0.0, 0.0, 1.0],  # Blue - Region 3  
        [1.0, 1.0, 0.0],  # Yellow - Region 4 (edge)
    ]
    
    region_names = [
        "Region 1 (Center)",
        "Region 2", 
        "Region 3",
        "Region 4 (Edge)"
    ]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f'Frog Region Analysis: {base_filename}', fontsize=16, fontweight='bold')
    
    # Top left: Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Top right: Original with colored overlay
    axes[0, 1].imshow(image_rgb)
    colored_overlay = np.zeros((*image_rgb.shape[:2], 4))
    for i, (mask, color) in enumerate(zip(regional_masks, region_colors)):
        colored_overlay[mask] = color + [0.7]  # Add alpha
    axes[0, 1].imshow(colored_overlay)
    axes[0, 1].set_title('Original + Region Overlay')
    axes[0, 1].axis('off')
    
    # Bottom left: Regions only
    region_only_image = np.zeros((*image_rgb.shape[:2], 3))
    for i, (mask, color) in enumerate(zip(regional_masks, region_colors)):
        region_only_image[mask] = color
    axes[1, 0].imshow(region_only_image)
    axes[1, 0].set_title('Regions Only')
    axes[1, 0].axis('off')
    
    # Bottom right: Individual regions in subplots
    gs = axes[1, 1].get_gridspec()
    axes[1, 1].remove()
    subfig = fig.add_subfigure(gs[1, 1])
    subaxes = subfig.subplots(2, 2)
    subfig.suptitle('Individual Regions', fontweight='bold')
    
    for i, (mask, color, name) in enumerate(zip(regional_masks, region_colors, region_names)):
        row, col = i // 2, i % 2
        region_img = np.zeros_like(image_rgb)
        region_img[mask] = [int(c * 255) for c in color]
        subaxes[row, col].imshow(region_img)
        subaxes[row, col].set_title(name, fontsize=10)
        subaxes[row, col].axis('off')
    
    # Add legend
    legend_elements = []
    for name, color in zip(region_names, region_colors):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=name))
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{base_filename}_region_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if plot:
        plt.show()
    else:
        plt.close()
    
    print(f"✅ Saved region visualization: {save_path}")
    
    # Also save individual components
    # Save region-only image
    region_only_path = os.path.join(output_dir, f'{base_filename}_regions_only.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(region_only_image)
    plt.axis('off')
    plt.title(f'Regions Only: {base_filename}', fontsize=14, fontweight='bold')
    plt.savefig(region_only_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True


def batch_visualize_regions(raw_jpg_dir=r"H:\\WkSleep_Trans_Up_to_25-5-1_Named_JPG",
                           raw_cr2_dir=r"H:\\WkSleep_Trans_Up_to_25-5-1_Named_cr2",
                           frog_seg_dir=None, plot=False):
    """Create region visualizations for all available frog images.
    
    Args:
        raw_jpg_dir (str): Directory containing JPG images
        raw_cr2_dir (str): Directory containing CR2 images
        frog_seg_dir (str): Directory containing frog masks (uses config if None)
        plot (bool): Whether to display plots interactively
    """
    frog_seg_dir = frog_seg_dir or PATHS["frog_segmented"]
    
    # Set up output directory
    results_root = PATHS.get("frog_results", os.path.join(PATHS["frog_dir"], "results"))
    output_dir = os.path.join(results_root, "region-visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect JPG files first
    jpg_files = []
    if os.path.isdir(raw_jpg_dir):
        for file in os.listdir(raw_jpg_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                jpg_files.append(file)
    
    # Sort JPG files
    jpg_files.sort()
    
    # Collect CR2 files, but only if no corresponding JPG exists
    cr2_files = []
    if os.path.isdir(raw_cr2_dir):
        for file in os.listdir(raw_cr2_dir):
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
    
    # Filter to only include files that have corresponding masks
    base_filenames_with_masks = []
    for file in all_files:
        base_filename = os.path.splitext(file)[0]
        mask_path = os.path.join(frog_seg_dir, f"{base_filename}_mask.png")
        if os.path.exists(mask_path):
            base_filenames_with_masks.append(base_filename)
    
    print(f"Found {len(jpg_files)} JPG files and {len(cr2_files)} CR2 files")
    print(f"Total files to process: {len(all_files)}")
    print(f"Files with available masks: {len(base_filenames_with_masks)}")
    
    success_count = 0
    for base_filename in base_filenames_with_masks:
        try:
            success = visualize_frog_regions(
                base_filename, output_dir, raw_jpg_dir, raw_cr2_dir, frog_seg_dir, plot
            )
            if success:
                success_count += 1
        except Exception as e:
            print(f"❌ Error processing {base_filename}: {e}")
    
    print(f"\n✅ Successfully processed {success_count}/{len(base_filenames_with_masks)} frogs")
    print(f"📁 Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process specific frog
        base_filename = sys.argv[1]
        results_root = PATHS.get("frog_results", os.path.join(PATHS["frog_dir"], "results"))
        output_dir = os.path.join(results_root, "region-visualizations")
        
        success = visualize_frog_regions(base_filename, output_dir, plot=True)
        if success:
            print(f"✅ Completed visualization for {base_filename}")
        else:
            print(f"❌ Failed to process {base_filename}")
    else:
        # Batch process all frogs
        print("Starting batch processing of all frog region visualizations...")
        batch_visualize_regions(plot=False) 