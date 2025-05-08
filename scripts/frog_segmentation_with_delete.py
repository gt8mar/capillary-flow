"""
Interactive Frog Segmentation using Segment Anything Model (SAM)

This script provides an interactive interface to segment frogs in images using
the Segment Anything Model. The user can click on the frog to provide prompts
for the segmentation algorithm.

Usage:
    1. Run directly in Cursor by pressing play (modify SCRIPT_CONFIG at the bottom of file)
    2. Or run from command line: python -m scripts.frog_segmentation [image_path] [--model_type TYPE]

Arguments:
    image_path: Optional path to a specific image
    --model_type: SAM model type to use (vit_h, vit_l, vit_b) - smaller is faster
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch
import argparse
import json
from segment_anything import sam_model_registry, SamPredictor
from src.config import PATHS

# Global variables
predictor = None
input_points = []
input_labels = []
mask = None
image = None
fig = None
ax = None
mask_overlay = None
status_text = None
image_path = None
original_image_size = None  # To track resizing for point coordinate adjustments
navigation_action = "next"  # Controls navigation between images ("next" or "back")
current_label = 1  # 1 = add (foreground), 0 = subtract (background)
COLOR_MAP = {1: 'r', 0: 'b'}  # Red for add, Blue for subtract

# Path to the progress tracking file
PROGRESS_FILE = os.path.join(PATHS['frog_dir'], "segmentation_progress.json")

# Disable Matplotlib default 's' key (save figure) so we can repurpose it
plt.rcParams['keymap.save'] = []

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Interactive Frog Segmentation")
    parser.add_argument("image_path", nargs="?", default=None, help="Path to specific image file (optional)")
    parser.add_argument("--model_type", choices=["vit_h", "vit_l", "vit_b"], default="vit_b", 
                      help="SAM model type (vit_b is fastest, vit_h is most accurate)")
    parser.add_argument("--resume", action="store_true", help="Resume from previously unfinished files")
    parser.add_argument("--skip_completed", action="store_true", default=True,
                      help="Skip images that already have segmentations saved (default: True)")
    return parser.parse_args()

def setup_sam(model_type="vit_b"):
    """
    Set up the Segment Anything Model.
    
    Args:
        model_type (str): Model type - vit_b (faster) or vit_h (more accurate)
    
    Returns:
        SamPredictor: The SAM predictor object
    """
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define model paths and checksums
    model_checkpoints = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    
    sam_checkpoint = os.path.join(PATHS['downloads'], model_checkpoints[model_type])
    
    # Check if the model exists
    if not os.path.exists(sam_checkpoint):
        print(f"SAM model not found at {sam_checkpoint}")
        print(f"Please download the {model_type} model from https://github.com/facebookresearch/segment-anything#model-checkpoints")
        print(f"and place it in {PATHS['downloads']}")
        sys.exit(1)
    
    print(f"Loading SAM model ({model_type})...")
    
    # Load the model with half precision to reduce memory usage
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    
    # Use half precision for faster inference and lower memory usage
    if device == "cuda":
        sam.half()  # Switch to half precision
        
    sam.to(device=device)
    
    # Create predictor
    predictor = SamPredictor(sam)
    
    # Clear CUDA cache to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("Model loaded successfully.")
    return predictor

def load_image(img_path):
    """
    Load an image for segmentation.
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: The loaded image in RGB format
    """
    global original_image_size
    
    # Check file extension
    _, ext = os.path.splitext(img_path)
    
    # For RAW files (like CR2), try using alternative methods
    if ext.lower() in ['.cr2', '.nef', '.arw', '.dng']:
        try:
            # Try to use rawpy if installed
            import rawpy
            with rawpy.imread(img_path) as raw:
                image = raw.postprocess()
                original_image_size = image.shape[:2]
                return image
        except ImportError:
            print("rawpy module not found. Install with: pip install rawpy")
            print("Falling back to OpenCV (may not work with RAW files)")
    
    # Load the image with OpenCV
    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not load image from {img_path}")
        sys.exit(1)
    
    # Convert to RGB (from BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Save original size for point scaling
    original_image_size = image.shape[:2]
    
    # Resize large images for better performance
    max_dim = 1024
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        ratio = max_dim / max(h, w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image from {w}x{h} to {new_w}x{new_h} for better performance")
    
    return image

def update_status(message):
    """Update the status message on the plot."""
    global status_text, fig
    
    if status_text:
        status_text.set_text(message)
    else:
        status_text = plt.figtext(0.5, 0.01, message, ha="center", fontsize=10)
    
    fig.canvas.draw_idle()

def load_progress():
    """Load progress from the JSON file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print("Error loading progress file. Creating a new one.")
    
    # Create default structure if file doesn't exist or has errors
    return {
        "completed": [],
        "in_progress": {}
    }

def save_progress():
    """Save current progress to the JSON file."""
    global input_points, input_labels, image_path
    
    # Get the current progress
    progress = load_progress()
    
    # Don't record empty progress
    if len(input_points) == 0:
        return
    
    # Update in-progress files
    progress["in_progress"][image_path] = {
        "points": input_points,
        "labels": input_labels
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    
    # Save to file
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"Progress saved for {os.path.basename(image_path)}")

def mark_as_completed():
    """Mark the current image as completed."""
    global image_path
    
    # Get the current progress
    progress = load_progress()
    
    # Add to completed list if not already there
    if image_path not in progress["completed"]:
        progress["completed"].append(image_path)
    
    # Remove from in-progress if it was there
    if image_path in progress["in_progress"]:
        del progress["in_progress"][image_path]
    
    # Save to file
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"Marked {os.path.basename(image_path)} as completed")

def is_completed(img_path):
    """Check if an image has been completed for the *with-delete* workflow.

    An image is considered complete **only** if the new style mask file
    (ending with ``_mask_with_delete.png``) exists.  This allows users who
    have already generated the legacy ``_mask.png`` to rerun the script and
    enrich their annotations with subtract points without having to move or
    delete the old masks.
    """
    progress = load_progress()

    # Locate the *new* mask file that corresponds to the with-delete flow
    base = os.path.splitext(os.path.basename(img_path))[0]
    delete_mask_path = os.path.join(
        PATHS['frog_segmented'], f"{base}_mask_with_delete.png"
    )

    # If the new mask exists, mark as complete (persistently) and skip
    if os.path.exists(delete_mask_path):
        if img_path not in progress["completed"]:
            progress["completed"].append(img_path)
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress, f, indent=2)
        return True

    # Otherwise ensure the image is NOT marked complete (might have been
    # completed in the legacy run). This enables re-processing.
    if img_path in progress["completed"]:
        progress["completed"].remove(img_path)
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)

    return False

def on_click(event):
    """
    Handle mouse click events on the image.
    The annotation label (add/subtract) is controlled by the global
    ``current_label`` which can be toggled with the ``t`` key.
    Positive points (foreground) use label ``1`` (red), negative points
    (background) use label ``0`` (blue) as per Segment Anything docs.
    """
    global input_points, input_labels, mask, mask_overlay

    if event.inaxes != ax:
        return

    update_status("Processing…")

    # Add the clicked point with current label
    input_points.append([event.xdata, event.ydata])
    input_labels.append(current_label)

    # Visualise the point using appropriate colour
    ax.plot(event.xdata, event.ydata, f"{COLOR_MAP[current_label]}o", markersize=8)

    # Generate mask if we have at least one point
    if len(input_points) > 0:
        points_array = np.array(input_points)
        labels_array = np.array(input_labels)

        with torch.no_grad():  # Disable gradient calculation for inference
            masks, scores, logits = predictor.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=True
            )

        mask_idx = np.argmax(scores)
        mask = masks[mask_idx]

        # Remove previous mask overlay if it exists
        if mask_overlay is not None:
            mask_overlay.remove()

        # Create a coloured mask overlay (still red)
        coloured_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
        coloured_mask[mask] = [1, 0, 0, 0.5]  # Red with 50% transparency
        mask_overlay = ax.imshow(coloured_mask)

    fig.canvas.draw()
    mode_str = "Add (red)" if current_label == 1 else "Subtract (blue)"
    update_status(f"Mode: {mode_str}  –  Click to add points, 'Save' when done")

    # Save progress after each point
    save_progress()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def on_reset(event):
    """
    Reset the segmentation process.
    
    Args:
        event: The matplotlib event object
    """
    global input_points, input_labels, mask, mask_overlay
    
    update_status("Resetting...")
    
    # Clear points and mask
    input_points = []
    input_labels = []
    mask = None
    
    # Remove mask overlay
    if mask_overlay is not None:
        mask_overlay.remove()
        mask_overlay = None
    
    # Clear the plot and redraw the image
    ax.clear()
    ax.imshow(image)
    ax.set_title("Click on the frog to segment it")
    ax.axis('off')
    
    # Save empty progress to clear saved points
    save_progress()
    
    fig.canvas.draw()
    update_status("Ready - Click to add points")

def on_save(event):
    """
    Save the segmentation mask and close the window.
    
    Args:
        event: The matplotlib event object
    """
    global fig, image_path, original_image_size
    
    if mask is None:
        update_status("No mask to save. Please segment the image first.")
        return
    
    update_status("Saving…")
    
    # Create output directory if it doesn't exist
    os.makedirs(PATHS['frog_segmented'], exist_ok=True)
    
    # Get output paths
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_filename = os.path.join(PATHS['frog_segmented'], f"{base_name}_mask_with_delete.png")
    
    # Save the mask
    # If the image was resized, we need to resize the mask back to the original size
    saved_mask = mask
    if original_image_size and (original_image_size[0], original_image_size[1]) != mask.shape:
        h, w = original_image_size
        saved_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    
    cv2.imwrite(mask_filename, saved_mask.astype(np.uint8) * 255)
    
    print(f"Saved mask to {mask_filename}")
    
    # Mark as completed in progress tracking
    mark_as_completed()
    
    # After saving we want to advance to the next image
    global navigation_action
    navigation_action = "next"
    # Close the figure
    plt.close(fig)

def on_skip(event):
    """
    Skip the current image without saving.
    
    Args:
        event: The matplotlib event object
    """
    global fig
    global navigation_action
    navigation_action = "next"
    
    print(f"Skipping {os.path.basename(image_path)}")
    
    # Save any progress
    save_progress()
    
    # Close the figure
    plt.close(fig)

def on_back(event=None):
    """Go back to the previous image without marking the current one as completed."""
    global fig, navigation_action
    navigation_action = "back"
    # Preserve any point progress so the user can resume if they return again
    save_progress()
    plt.close(fig)

def on_key_press(event):
    """Map keyboard shortcuts to existing GUI actions."""
    key = event.key.lower()
    if key == 's':
        on_save(event)
    elif key == 'n':
        on_skip(event)
    elif key == 'b':
        on_back(event)
    elif key == 'r':
        on_reset(event)
    elif key == 't':
        global current_label
        current_label = 0 if current_label == 1 else 1
        mode_str = "Add (red)" if current_label == 1 else "Subtract (blue)"
        update_status(f"Toggled mode: {mode_str}")

def interactive_segmentation(img_path):
    """
    Run the interactive segmentation interface for a single image.
    
    Args:
        img_path (str): Path to the image file
    """
    global predictor, image, fig, ax, input_points, input_labels, mask, mask_overlay, image_path
    
    # Set the current image path
    image_path = img_path
    
    # Reset state variables
    input_points = []
    input_labels = []
    mask = None
    mask_overlay = None
    
    # Check if there's saved progress for this image
    progress = load_progress()
    if img_path in progress["in_progress"]:
        saved_progress = progress["in_progress"][img_path]
        input_points = saved_progress["points"]
        input_labels = saved_progress["labels"]
        print(f"Loaded {len(input_points)} saved points for {os.path.basename(img_path)}")
    
    # Load the image
    print(f"Loading image: {os.path.basename(image_path)}")
    image = load_image(image_path)
    
    # Set the image in the predictor
    predictor.set_image(image)
    
    # Create the interactive plot
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.2)
    
    # Display the image
    ax.imshow(image)
    ax.set_title(f"Segmenting: {os.path.basename(image_path)}")
    ax.axis('off')
    
    # Add buttons
    reset_ax = plt.axes([0.59, 0.05, 0.1, 0.075])
    skip_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
    save_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
    
    reset_button = Button(reset_ax, 'Reset')
    skip_button = Button(skip_ax, 'Skip')
    save_button = Button(save_ax, 'Save')
    
    reset_button.on_clicked(on_reset)
    skip_button.on_clicked(on_skip)
    save_button.on_clicked(on_save)
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    # Connect keyboard shortcuts
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # If we have saved points, display them and generate the mask
    if len(input_points) > 0:
        # Display saved points with colour according to label
        for point, lbl in zip(input_points, input_labels):
            ax.plot(point[0], point[1], f"{COLOR_MAP.get(lbl, 'r')}o", markersize=8)
        
        # Generate mask from saved points
        points_array = np.array(input_points)
        labels_array = np.array(input_labels)
        
        # Generate mask
        with torch.no_grad():
            masks, scores, logits = predictor.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=True
            )
        
        # Select the mask with the highest score
        mask_idx = np.argmax(scores)
        mask = masks[mask_idx]
        
        # Create a coloured mask overlay (still red)
        coloured_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
        coloured_mask[mask] = [1, 0, 0, 0.5]  # Red with 50% transparency
        
        # Display the mask overlay
        mask_overlay = ax.imshow(coloured_mask)
    
    # Add status message
    update_status("Ready - Click on the frog to segment it")
    
    # Show the plot
    plt.show()

def get_remaining_images(image_paths, args):
    """
    Filter the list of images to only include those that need processing.
    
    Args:
        image_paths (list): List of all image paths
        args (Namespace): Command line arguments
        
    Returns:
        list: Filtered list of image paths
    """
    if not args.skip_completed:
        return image_paths
    
    remaining = []
    for img_path in image_paths:
        if is_completed(img_path):
            print(f"Skipping {os.path.basename(img_path)} (already completed)")
        else:
            remaining.append(img_path)
    
    return remaining

def get_unfinished_images():
    """
    Get the list of images that have been started but not completed.
    
    Returns:
        list: List of image paths
    """
    progress = load_progress()
    return list(progress["in_progress"].keys())

def process_images(image_paths, args):
    """
    Process multiple images using the same SAM model.
    
    Args:
        image_paths (list): List of image paths to process
        args (Namespace): Command line arguments
    """
    global predictor
    
    # Filter out completed images if requested
    image_paths = get_remaining_images(image_paths, args)
    
    if not image_paths:
        print("No images to process. All images have been completed or skipped.")
        return
    
    # Load the model only once
    predictor = setup_sam(args.model_type)
    
    # Index-controlled loop so we can move backwards if requested
    i = 0
    while i < len(image_paths):
        img_path = image_paths[i]
        print(f"\n[{i+1}/{len(image_paths)}] Processing {os.path.basename(img_path)}")
        # Reset navigation flag before showing the GUI
        global navigation_action
        navigation_action = "next"
        interactive_segmentation(img_path)

        # Adjust index based on user action
        if navigation_action == "back":
            i = max(i - 1, 0)
        else:
            i += 1

        # Clear CUDA cache between images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_with_config(config):
    """
    Run the segmentation with the given configuration.
    
    Args:
        config (dict): Configuration dictionary
    """
    # Create args object to maintain compatibility with existing code
    class Args:
        pass
    
    args = Args()
    args.model_type = config.get("model_type", "vit_b")
    args.resume = config.get("resume", False)
    args.skip_completed = config.get("skip_completed", True)
    args.image_path = config.get("specific_image_path", None)
    
    # Process specific image if provided
    if args.image_path:
        if os.path.isfile(args.image_path):
            process_images([args.image_path], args)
        else:
            print(f"Error: Image file not found: {args.image_path}")
            return
    else:
        # Get the image folder from config or use default
        image_folder = config.get("image_folder", None)
        if image_folder is None:
            # image_folder = os.path.join(PATHS['downloads'], "whole-frog")
            image_folder = os.path.join('H:\\WkSleep_Trans_Up_to_25-5-1_Named')
        image_folder = os.path.normpath(image_folder)
        
        # Check if the folder exists
        if not os.path.exists(image_folder):
            print(f"Error: Image folder not found: {image_folder}")
            return
        
        # Get list of image files
        image_files = [f for f in os.listdir(image_folder) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.cr2', '.nef', '.arw', '.dng'))]
        
        if not image_files:
            print(f"No image files found in {image_folder}")
            return
        
        # Convert to full paths
        image_paths = [os.path.join(image_folder, f) for f in image_files]
        
        print(f"Found {len(image_paths)} images to process")
        
        # Process unfinished images first if resuming
        if args.resume:
            unfinished_images = get_unfinished_images()
            if unfinished_images:
                print(f"Resuming {len(unfinished_images)} unfinished images")
                process_images(unfinished_images, args)
            else:
                print("No unfinished images found.")
        
        # Process all images with the same model instance
        process_images(image_paths, args)
    
    print("Processing complete.")

if __name__ == "__main__":
    # Configuration options for running from Cursor (edit these as needed)
    SCRIPT_CONFIG = {
        # Model options: "vit_b" (faster), "vit_l" (medium), "vit_h" (more accurate)
        "model_type": "vit_b",
        
        # Set to True to resume working on images you started but didn't finish
        "resume": True,
        
        # Set to True to skip images that already have saved segmentations
        "skip_completed": True,
        
        # Optional: Path to a specific image to process (leave as None to process all images in the folder)
        "specific_image_path": None,
        
        # Optional: Path to folder with images (leave as None to use default folder)
        "image_folder": None,
    }
    
    # If command line arguments are provided, use them instead of SCRIPT_CONFIG
    if len(sys.argv) > 1:
        args = parse_arguments()
        
        # If resuming, process unfinished images first
        if args.resume:
            unfinished_images = get_unfinished_images()
            if unfinished_images:
                print(f"Resuming {len(unfinished_images)} unfinished images")
                process_images(unfinished_images, args)
            else:
                print("No unfinished images found.")
        
        # If a specific image path is provided, use it
        if args.image_path:
            if os.path.isfile(args.image_path):
                process_images([args.image_path], args)
            else:
                print(f"Error: Image file not found: {args.image_path}")
        else:
            # Otherwise process all images in the default folder
            image_folder = os.path.join(PATHS['downloads'], "whole-frog")
            image_folder = os.path.normpath(image_folder)
            
            # Check if the folder exists
            if not os.path.exists(image_folder):
                print(f"Error: Image folder not found: {image_folder}")
                sys.exit(1)
            
            # Get list of image files
            image_files = [f for f in os.listdir(image_folder) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.cr2', '.nef', '.arw', '.dng'))]
            
            if not image_files:
                print(f"No image files found in {image_folder}")
                sys.exit(1)
            
            # Convert to full paths
            image_paths = [os.path.join(image_folder, f) for f in image_files]
            
            print(f"Found {len(image_paths)} images to process")
            
            # Process all images with the same model instance
            process_images(image_paths, args)
    else:
        # Run with configuration from SCRIPT_CONFIG
        run_with_config(SCRIPT_CONFIG)
