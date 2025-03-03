#!/usr/bin/env python
"""
Simple test script to check if matplotlib can display an image.

Usage:
    python scripts/test_display.py [image_path]
"""

import os
import sys
import cv2
import matplotlib.pyplot as plt
from src.config import PATHS

def test_display(image_path):
    """Test if an image can be displayed using matplotlib."""
    print(f"Attempting to load and display image: {image_path}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Failed to load image: {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("Image loaded successfully")
    print(f"Image shape: {image.shape}")
    
    # Create a figure
    print("Creating matplotlib figure...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display the image
    print("Displaying image...")
    ax.imshow(image_rgb)
    ax.set_title("Test Image Display")
    ax.axis('off')
    
    # Show the plot
    print("Calling plt.show()...")
    plt.show()
    
    print("plt.show() completed")

if __name__ == "__main__":
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find an image in the default location
        default_dir = os.path.join(PATHS['downloads'], "whole-frog")
        if os.path.exists(default_dir):
            jpg_files = [f for f in os.listdir(default_dir) if f.lower().endswith('.jpg')]
            if jpg_files:
                image_path = os.path.join(default_dir, jpg_files[0])
                print(f"Using first found image: {image_path}")
            else:
                print("No JPG files found in default directory")
                sys.exit(1)
        else:
            print(f"Default directory not found: {default_dir}")
            sys.exit(1)
    
    # Test display
    test_display(image_path) 