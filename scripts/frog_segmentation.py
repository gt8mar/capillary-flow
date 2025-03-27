"""
Interactive Frog Segmentation using Segment Anything Model (SAM)

This script provides an interactive interface to segment frogs in images using
the Segment Anything Model. The user can click on the frog to provide prompts
for the segmentation algorithm.

Usage:
    python -m src.tools.frog_segmentation [image_path] [--model_type TYPE]

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

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Interactive Frog Segmentation")
    parser.add_argument("image_path", nargs="?", default=None, help="Path to specific image file (optional)")
    parser.add_argument("--model_type", choices=["vit_h", "vit_l", "vit_b"], default="vit_b", 
                      help="SAM model type (vit_b is fastest, vit_h is most accurate)")
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
    # Check file extension
    _, ext = os.path.splitext(img_path)
    
    # For RAW files (like CR2), try using alternative methods
    if ext.lower() in ['.cr2', '.nef', '.arw', '.dng']:
        try:
            # Try to use rawpy if installed
            import rawpy
            with rawpy.imread(img_path) as raw:
                image = raw.postprocess()
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

def on_click(event):
    """
    Handle mouse click events on the image.
    
    Args:
        event: The matplotlib event object
    """
    global input_points, input_labels, mask, mask_overlay
    
    if event.inaxes != ax:
        return
    
    update_status("Processing...")
    
    # Add the clicked point
    input_points.append([event.xdata, event.ydata])
    input_labels.append(1)  # 1 for foreground
    
    # Update the plot with the new point
    ax.plot(event.xdata, event.ydata, 'ro', markersize=8)
    
    # Generate mask if we have at least one point
    if len(input_points) > 0:
        # Convert points to numpy array
        points_array = np.array(input_points)
        labels_array = np.array(input_labels)
        
        # Generate mask
        with torch.no_grad():  # Disable gradient calculation for inference
            masks, scores, logits = predictor.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=True
            )
        
        # Select the mask with the highest score
        mask_idx = np.argmax(scores)
        mask = masks[mask_idx]
        
        # Remove previous mask overlay if it exists
        if mask_overlay is not None:
            mask_overlay.remove()
        
        # Create a colored mask overlay
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
        colored_mask[mask] = [1, 0, 0, 0.5]  # Red with 50% transparency
        
        # Display the mask overlay
        mask_overlay = ax.imshow(colored_mask)
    
    fig.canvas.draw()
    update_status("Ready - Click to add points, 'Save' when done")
    
    # Clear CUDA cache to free up memory
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
    
    fig.canvas.draw()
    update_status("Ready - Click to add points")

def on_save(event):
    """
    Save the segmentation mask and close the window.
    
    Args:
        event: The matplotlib event object
    """
    global fig, image_path
    
    if mask is None:
        update_status("No mask to save. Please segment the image first.")
        return
    
    update_status("Saving...")
    
    # Create output directory if it doesn't exist
    os.makedirs(PATHS['frog_segmented'], exist_ok=True)
    
    # Get output paths
    mask_filename = os.path.join(PATHS['frog_segmented'], f"{os.path.splitext(os.path.basename(image_path))[0]}_mask.png")
    segmented_filename = os.path.join(PATHS['frog_segmented'], f"{os.path.splitext(os.path.basename(image_path))[0]}_segmented.png")
    
    # Save the mask
    cv2.imwrite(mask_filename, mask.astype(np.uint8) * 255)
    
    # Save the segmented image (original with mask overlay)
    segmented_image = image.copy()
    segmented_image[mask] = segmented_image[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    cv2.imwrite(segmented_filename, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    
    print(f"Saved mask to {mask_filename}")
    print(f"Saved segmented image to {segmented_filename}")
    
    # Close the figure
    plt.close(fig)

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
    reset_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
    save_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
    
    reset_button = Button(reset_ax, 'Reset')
    save_button = Button(save_ax, 'Save')
    
    reset_button.on_clicked(on_reset)
    save_button.on_clicked(on_save)
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Add status message
    update_status("Ready - Click on the frog to segment it")
    
    # Show the plot
    plt.show()

def process_images(image_paths, args):
    """
    Process multiple images using the same SAM model.
    
    Args:
        image_paths (list): List of image paths to process
        args (Namespace): Command line arguments
    """
    global predictor
    
    # Load the model only once
    predictor = setup_sam(args.model_type)
    
    # Process each image
    for i, img_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing {os.path.basename(img_path)}")
        interactive_segmentation(img_path)
        
        # Clear CUDA cache between images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
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
        
    print("Processing complete.")
