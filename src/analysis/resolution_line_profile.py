"""
Filename: src/analysis/resolution_line_profile.py

This script is used to analyze the resolution line profile of
a USAF resolution target.

By Marcus Forst
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Local imports
from src.tools.parse_filename import parse_filename
from src.config import PATHS  # Import PATHS from config module

# Use paths from config instead of platform-specific checks
cap_flow_path = PATHS['cap_flow']
downloads_path = PATHS['downloads']


def main():
    user_folder = os.path.dirname(cap_flow_path)
    image_path = os.path.join(user_folder, "Desktop", "data", "resolution_target.png")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()

