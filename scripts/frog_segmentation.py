"""
Interactive Frog Segmentation using Segment Anything Model (SAM)

This script provides an interactive interface to segment frogs in images using
the Segment Anything Model. The user can click on the frog to provide prompts
for the segmentation algorithm.

Usage:
    python -m src.tools.frog_segmentation [image_path]

If no image path is provided, the script will use a default image path.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch
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

def setup_sam():
    """
    Set up the Segment Anything Model.
    
    Returns:
        SamPredictor: The SAM predictor object
    """
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define model paths
    sam_checkpoint = os.path.join(PATHS['downloads'], "sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    
    # Check if the model exists
    if not os.path.exists(sam_checkpoint):
        print(f"SAM model not found at {sam_checkpoint}")
        print("Please download the model from https://github.com/facebookresearch/segment-anything#model-checkpoints")
        print(f"and place it in {PATHS['downloads']}")
        sys.exit(1)
    
    # Load the model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # Create predictor
    predictor = SamPredictor(sam)
    
    return predictor

def load_image(image_path):
    """
    Load an image for segmentation.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: The loaded image in RGB format
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image from {image_path}")
        sys.exit(1)
    
    # Convert to RGB (from BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def on_click(event):
    """
    Handle mouse click events on the image.
    
    Args:
        event: The matplotlib event object
    """
    global input_points, input_labels, mask, mask_overlay
    
    if event.inaxes != ax:
        return
    
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

def on_reset(event):
    """
    Reset the segmentation process.
    
    Args:
        event: The matplotlib event object
    """
    global input_points, input_labels, mask, mask_overlay
    
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

def on_save(event):
    """
    Save the segmentation mask.
    
    Args:
        event: The matplotlib event object
    """
    if mask is None:
        print("No mask to save. Please segment the image first.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(PATHS['frog_segmented'], exist_ok=True)
    
    # Save the mask
    mask_filename = os.path.join(PATHS['frog_segmented'], f"{os.path.splitext(os.path.basename(image_path))[0]}_mask.png")
    cv2.imwrite(mask_filename, mask.astype(np.uint8) * 255)
    
    # Save the segmented image (original with mask overlay)
    segmented_image = image.copy()
    segmented_image[mask] = segmented_image[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    segmented_filename = os.path.join(PATHS['frog_segmented'], f"{os.path.splitext(os.path.basename(image_path))[0]}_segmented.png")
    cv2.imwrite(segmented_filename, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    
    print(f"Saved mask to {mask_filename}")
    print(f"Saved segmented image to {segmented_filename}")

def interactive_segmentation(image_path):
    """
    Run the interactive segmentation interface.
    
    Args:
        image_path (str): Path to the image file
    """
    global predictor, image, fig, ax, input_points, input_labels, mask, mask_overlay
    
    # Setup SAM
    predictor = setup_sam()
    
    # Load the image
    image = load_image(image_path)
    
    # Set the image in the predictor
    predictor.set_image(image)
    
    # Create the interactive plot
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.2)
    
    # Display the image
    ax.imshow(image)
    ax.set_title("Click on the frog to segment it")
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
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a default image path
        image_path = os.path.join(PATHS['downloads'], "whole-frog", "frog_sample.JPG")
    
    # Run the interactive segmentation
    interactive_segmentation(image_path) 