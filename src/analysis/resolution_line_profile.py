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
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
from datetime import datetime

# Local imports
from src.tools.parse_filename import parse_filename
from src.config import PATHS, load_source_sans  # Import PATHS from config module

# Use paths from config instead of platform-specific checks
cap_flow_path = PATHS['cap_flow']
downloads_path = PATHS['downloads']
results_path = os.path.join(cap_flow_path, "results")
source_sans = load_source_sans()





def setup_plot_style():
    """Set up standard plot styling according to coding standards."""
    sns.set_style("whitegrid")
    
    plt.rcParams.update({
        'pdf.fonttype': 42,  # For editable text in PDFs
        'ps.fonttype': 42,   # For editable text in PostScript
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 5,
        'lines.linewidth': 0.5
    })
    
    return source_sans


def analyze_resolution_profile(image_path, profile_row=566, profile_col_start=600, profile_col_end=615, 
                               save=True, show=True):
    """Analyze resolution line profile from a USAF resolution target image.
    
    Extracts a line profile from the specified row and column range of a grayscale image
    and plots the intensity profile. The profile shows the contrast between bars on
    a resolution target, which can be used to quantify the resolving power of the
    optical system.
    
    Args:
        image_path: Path to the input image
        profile_row: Row index to extract the line profile from
        profile_col_start: Starting column index for the profile
        profile_col_end: Ending column index for the profile
        save: Whether to save the plot
        show: Whether to display the plot
        
    Returns:
        Tuple containing the extracted profile and figure object
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None, None
    
    # Configure plot style
    setup_plot_style()
    
    # Extract the line profile
    profile = image[profile_row, profile_col_start:profile_col_end]
    
    # Create the figure with standard dimensions
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    
    # Create x-axis values based on original column indices
    x_values = np.arange(profile_col_start, profile_col_end)
    
    # Plot the profile using original column indices
    ax.plot(x_values, profile, linewidth=0.75, color='#1f77b4')
    
    # Set x-axis to use integer ticks only
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Set plot labels and title with the appropriate font
    if source_sans:
        ax.set_xlabel('Column Index (pixels)', fontproperties=source_sans)
        ax.set_ylabel('Intensity', fontproperties=source_sans)
        ax.set_title('Resolution Line Profile', fontsize=8, fontproperties=source_sans)
    else:
        ax.set_xlabel('Column Index (pixels)')
        ax.set_ylabel('Intensity')
        ax.set_title('Resolution Line Profile', fontsize=8)
    
    # Tight layout to optimize use of space
    plt.tight_layout()
    
    # Save the figure without timestamp
    if save:
        save_dir = os.path.join(results_path, "resolution")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "resolution_profile.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    # Show the plot if requested
    if show:
        plt.show()
    
    return profile, fig


def main():
    """Main function to run the resolution line profile analysis."""
    image_path = os.path.join(downloads_path, "Image__2024-05-22__09-28-06.tiff")
    
    # Run the analysis with default parameters
    analyze_resolution_profile(
        image_path,
        profile_row=566,
        profile_col_start=600,
        profile_col_end=615,
        save=True,
        show=True
    )


if __name__ == "__main__":
    main()

