"""
Script to run the interactive frog segmentation tool.

Usage:
    python -m src.scripts.segment_frog [image_path]
"""

import sys
from scripts.frog_segmentation import interactive_segmentation

if __name__ == "__main__":
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        interactive_segmentation(image_path)
    else:
        print("Please provide an image path:")
        print("python -m src.scripts.segment_frog path/to/frog_image.jpg") 