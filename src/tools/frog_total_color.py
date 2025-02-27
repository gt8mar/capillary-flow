"""
Segment Frog

This script performs segmentation on an image of a frog using the Segment Anything Model (SAM). 
It allows for segmentation based on user input (point, box, or two points) and analyzes the segmented 
frog to determine the redness of different areas. The script also includes quality control checks 
for lighting and contrast.

Steps:
1. Load the Segment Anything Model (SAM).
2. Load an image.
3. Add SAM to the code.
4. Run SAM.
5. Input either a point, a box, or two points to segment the frog.
6. Outputs:
    - Masks: A collection of points representing the segmented areas.
    - Scores: Confidence scores for the segmentation.
    - Logits: Raw prediction values (check what this is).
7. Analyze the segmented frog:
    - Determine how red the frog is.
    - Separate RGB channels and name them.
    - Compare green and red channels.
    - Identify areas that are more red than others.
    - Segment more parts of the frog and check those.
    - Analyze if the inside is more red than the outside (using radial circles).
    - Skeletonize the frog image.
8. Quality control:
    - Assess the evenness of the light (background and on the frog).
    - Generate row light profile and average across rows.
    - Generate column light profile and average across columns.
    - Normalize contrast.
    - Compare background colors and contrast.

Things to remember:
- Separate RGB channels and name them.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import platform
from src.config import get_paths

# Get paths at module level for use in functions
PATHS = get_paths()

def main(frog_path, plot=False):
    # load_sam()   
    # add_sam_to_code()
    # masks, scores, logits = run_sam(image)
    # frog_path = "C://Users//gt8ma//Downloads//FrogPhoto-removebg-preview.png"
    download_path = "C://Users//gt8ma//Downloads//whole-frog//"
    image = cv2.imread(download_path + frog_path)
    
    # Extract filename without extension for output files
    base_filename = os.path.splitext(os.path.basename(frog_path))[0]
    print(f'Analyzing {base_filename}')

    
    analyze_frog(image, base_filename, plot)
    # quality_control(image)

def load_sam():
    # Load the Segment Anything Model (SAM)
    pass

def add_sam_to_code():
    # Add SAM to the code
    pass

def run_sam(image):
    # Run SAM
    # Input either a point, a box, or two points to segment the frog
    # Outputs: Masks, Scores, Logits
    return 0, 0, 0

def analyze_radial_regions(frog_mask, red_ratio, image_rgb, base_filename="frog", plot=False, output_dir=None):
    """Analyze the redness of concentric regions of the frog from center outward.
    
    Args:
        frog_mask (np.ndarray): Binary mask of the frog
        red_ratio (np.ndarray): Array containing red ratio values (R/RGB)
        image_rgb (np.ndarray): RGB image of the frog
        base_filename (str): Base filename for saving output files
        plot (bool): Whether to display plots
        output_dir (str): Directory to save output files
        
    Returns:
        tuple: Region red ratios (region_1_red_ratio, region_2_red_ratio, 
                region_3_red_ratio, region_4_red_ratio)
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
    
    # Get the red ratio values for each region
    region_values = []
    region_rg_values = []  # Red/(Red+Green) values
    
    # Extract red and green channels
    r_channel = image_rgb[:, :, 0].astype(float)
    g_channel = image_rgb[:, :, 1].astype(float)
    
    # Calculate Red/(Red+Green) ratio
    rg_sum = r_channel + g_channel
    rg_sum[rg_sum == 0] = 1  # Avoid division by zero
    red_rg_ratio = r_channel / rg_sum
    
    for i in range(4):
        start_idx = i * region_size
        end_idx = (i + 1) * region_size if i < 3 else num_pixels
        region_indices = sorted_indices[start_idx:end_idx]
        region_y = y_coords[region_indices]
        region_x = x_coords[region_indices]
        
        # Calculate R/RGB ratio for this region
        region_red_ratios = red_ratio[region_y, region_x]
        region_values.append(np.mean(region_red_ratios))
        
        # Calculate R/(R+G) ratio for this region
        region_rg_ratios = red_rg_ratio[region_y, region_x]
        region_rg_values.append(np.mean(region_rg_ratios))
    
    # Make bar plot of the red ratios
    plt.figure(figsize=(10, 6))
    
    # Set up bar positions
    x = np.arange(4)
    width = 0.35
    
    # Create grouped bar chart
    plt.bar(x - width/2, region_values, width, label='Red/(Red+Green+Blue)')
    plt.bar(x + width/2, region_rg_values, width, label='Red/(Red+Green)')
    
    plt.xlabel('Region (Center to Edge)')
    plt.ylabel('Ratio Value')
    plt.title('Red Ratios by Region (Center to Edge)')
    plt.xticks(x, ['Region 1\n(Center)', 'Region 2', 'Region 3', 'Region 4\n(Edge)'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Use output_dir if provided, otherwise use current directory
    if output_dir is None:
        output_path = f'{base_filename}_regions.png'
    else:
        output_path = os.path.join(output_dir, f'{base_filename}_regions.png')
    
    plt.savefig(output_path)
    if plot:
        plt.show()
    else:
        plt.close()
    
    return tuple(region_values), tuple(region_rg_values)

def analyze_rgb_by_region(frog_mask, image_rgb, base_filename="frog", plot=False, output_dir=None):
    """Analyze the pure RGB values of concentric regions of the frog from center outward.
    
    Args:
        frog_mask (np.ndarray): Binary mask of the frog
        image_rgb (np.ndarray): RGB image of the frog
        base_filename (str): Base filename for saving output files
        plot (bool): Whether to display plots
        output_dir (str): Directory to save output files
        
    Returns:
        tuple: Region RGB values (r_values, g_values, b_values)
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
    
    # Extract RGB channels
    r_channel = image_rgb[:, :, 0].astype(float)
    g_channel = image_rgb[:, :, 1].astype(float)
    b_channel = image_rgb[:, :, 2].astype(float)
    
    # Get the RGB values for each region
    r_values = []
    g_values = []
    b_values = []
    
    for i in range(4):
        start_idx = i * region_size
        end_idx = (i + 1) * region_size if i < 3 else num_pixels
        region_indices = sorted_indices[start_idx:end_idx]
        region_y = y_coords[region_indices]
        region_x = x_coords[region_indices]
        
        # Calculate average RGB values for this region
        r_values.append(np.mean(r_channel[region_y, region_x]))
        g_values.append(np.mean(g_channel[region_y, region_x]))
        b_values.append(np.mean(b_channel[region_y, region_x]))
    
    # Make bar plot of the RGB values
    plt.figure(figsize=(10, 6))
    
    # Set up bar positions
    x = np.arange(4)
    width = 0.25
    
    # Create grouped bar chart
    plt.bar(x - width, r_values, width, color='red', label='Red')
    plt.bar(x, g_values, width, color='green', label='Green')
    plt.bar(x + width, b_values, width, color='blue', label='Blue')
    
    plt.xlabel('Region (Center to Edge)')
    plt.ylabel('Average Channel Value (0-255)')
    plt.title('RGB Channel Values by Region (Center to Edge)')
    plt.xticks(x, ['Region 1\n(Center)', 'Region 2', 'Region 3', 'Region 4\n(Edge)'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Use output_dir if provided, otherwise use current directory
    if output_dir is None:
        output_path = f'{base_filename}_rgb_regions.png'
    else:
        output_path = os.path.join(output_dir, f'{base_filename}_rgb_regions.png')
    
    plt.savefig(output_path)
    if plot:
        plt.show()
    else:
        plt.close()
    
    return tuple(r_values), tuple(g_values), tuple(b_values)

def analyze_frog(image, base_filename="frog", plot=False):
    """ Analyze the segmented frog

    Args:
        image (numpy.ndarray): The image of the frog.
        base_filename (str): Base filename for saving output files
        plot (bool): Whether to display plots

    Returns:
        dict: A dictionary containing the following keys:
            - 'red_ratio': The red ratio of the frog.
            - 'red_minus_green': The red minus green of the frog.
            - 'red_ratio_by_region': The red ratio of the frog by region.
            - 'red_rg_ratio_by_region': The red/(red+green) ratio by region.
            - 'rgb_by_region': The RGB values by region.
    """
    # Convert BGR to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Separate RGB channels
    r_channel = image_rgb[:, :, 0]
    g_channel = image_rgb[:, :, 1]
    b_channel = image_rgb[:, :, 2]
    
    # Create a mask for the frog (non-transparent areas)
    # Assuming the background is transparent or white
    if image.shape[2] == 4:  # Image has alpha channel
        frog_mask = image[:, :, 3] > 0
    else:
        # Create a simple mask based on white background
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, frog_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        frog_mask = frog_mask > 0
    
    # Calculate redness metrics
    # 1. Red minus green (higher values indicate more red than green)
    red_minus_green = r_channel.astype(int) - g_channel.astype(int)
    
    # 2. Red ratio (red / (red + green + blue))
    rgb_sum = r_channel.astype(float) + g_channel.astype(float) + b_channel.astype(float)
    # Avoid division by zero
    rgb_sum[rgb_sum == 0] = 1
    red_ratio = r_channel.astype(float) / rgb_sum
    
    # Apply the frog mask to the metrics
    red_minus_green_masked = np.zeros_like(red_minus_green)
    red_minus_green_masked[frog_mask] = red_minus_green[frog_mask]
    
    red_ratio_masked = np.zeros_like(red_ratio)
    red_ratio_masked[frog_mask] = red_ratio[frog_mask]
    
    # Create a heatmap of redness
    # Normalize red_minus_green for better visualization
    red_minus_green_norm = red_minus_green_masked.astype(float)
    if np.max(red_minus_green_norm) > np.min(red_minus_green_norm):
        red_minus_green_norm = (red_minus_green_norm - np.min(red_minus_green_norm)) / (np.max(red_minus_green_norm) - np.min(red_minus_green_norm))
    
    # Create a colormap for the heatmap
    heatmap = np.zeros((red_minus_green_norm.shape[0], red_minus_green_norm.shape[1], 3), dtype=np.uint8)
    heatmap[:, :, 0] = (255 * red_minus_green_norm).astype(np.uint8)  # Red channel
    
    # Use the paths from config
    output_dir = PATHS['frog_analysis']
    
    # Analyze inside vs outside redness
    region_rgb_ratios, region_rg_ratios = analyze_radial_regions(frog_mask, red_ratio, image_rgb, base_filename, plot, output_dir)
    
    # Analyze pure RGB values by region
    r_values, g_values, b_values = analyze_rgb_by_region(frog_mask, image_rgb, base_filename, plot, output_dir)
    
    # Plot the results
    plt.figure(figsize=(15, 12))
    
    # Original image
    plt.subplot(3, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Individual channels
    plt.subplot(3, 2, 2)
    plt.imshow(r_channel, cmap='Reds')
    plt.title('Red Channel')
    plt.axis('off')
    
    # Red minus green heatmap
    plt.subplot(3, 2, 3)
    plt.imshow(red_minus_green_masked, cmap='hot')
    plt.colorbar(label='Red - Green')
    plt.title('Red Minus Green')
    plt.axis('off')
    
    # Red ratio heatmap
    plt.subplot(3, 2, 4)
    plt.imshow(red_ratio_masked, cmap='hot')
    plt.colorbar(label='Red Ratio')
    plt.title('Red Ratio (R / RGB)')
    plt.axis('off')
    
    # Bar plot of the red ratios    
    plt.subplot(3, 2, 5)
    x = np.arange(4)
    width = 0.35
    plt.bar(x - width/2, region_rgb_ratios, width, label='R/(R+G+B)')
    plt.bar(x + width/2, region_rg_ratios, width, label='R/(R+G)')
    plt.title('Red Ratios by Region')
    plt.xlabel('Region (Center to Edge)')
    plt.ylabel('Ratio Value')
    plt.xticks(x, ['R1', 'R2', 'R3', 'R4'])
    plt.legend()
    
    # Bar plot of the RGB values
    plt.subplot(3, 2, 6)
    width = 0.25
    plt.bar(x - width, r_values, width, color='red', label='Red')
    plt.bar(x, g_values, width, color='green', label='Green')
    plt.bar(x + width, b_values, width, color='blue', label='Blue')
    plt.title('RGB Channel Values by Region')
    plt.xlabel('Region (Center to Edge)')
    plt.ylabel('Channel Value')
    plt.xticks(x, ['R1', 'R2', 'R3', 'R4'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_filename}_analysis.png'))
    if plot:
        plt.show()
    else:
        plt.close()
    
    # Print analysis results
    print(f"Average red ratio in the frog: {np.mean(red_ratio[frog_mask]):.3f}")
    print(f"Average R/(R+G+B) ratio by region (center to edge):")
    for i, ratio in enumerate(region_rgb_ratios):
        print(f"  Region {i+1}: {ratio:.3f}")
    print(f"Average R/(R+G) ratio by region (center to edge):")
    for i, ratio in enumerate(region_rg_ratios):
        print(f"  Region {i+1}: {ratio:.3f}")
    print(f"Average RGB values by region (center to edge):")
    for i in range(4):
        print(f"  Region {i+1}: R={r_values[i]:.1f}, G={g_values[i]:.1f}, B={b_values[i]:.1f}")

    # Return the results
    return {
        'red_ratio': red_ratio_masked,
        'red_minus_green': red_minus_green_masked,
        'red_ratio_by_region': region_rgb_ratios,
        'red_rg_ratio_by_region': region_rg_ratios,
        'rgb_by_region': (r_values, g_values, b_values)
    }

def quality_control(image):
    # Quality control
    # Assess the evenness of the light (background and on the frog)
    # Generate row light profile and average across rows
    # Generate column light profile and average across columns
    # Normalize contrast
    # Compare background colors and contrast
    pass

if __name__ == "__main__":
    for file in os.listdir("C://Users//gt8ma//Downloads//whole-frog"):
        if file.endswith(".JPG"):
            print(file)
            main(file, plot = False)