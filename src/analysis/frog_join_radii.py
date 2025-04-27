"""
Filename: frog_join_radii.py
-------------------------------------------------
This script loads RBC counting data from frog_counting.py and merges it with
capillary radius information from centerline files.

It calculates mean_diameter and bulk_diameter values for each capillary
and merges them into the counting dataframe, then saves the result with
a _radii suffix.

By: [Your Name]
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import PATHS  # Import PATHS from config module

# Constants
CENTERLINE_DIR = os.path.join('H:', '240729', 'Frog2', 'Right', 'centerlines')
COORDS_DIR = os.path.join('H:', '240729', 'Frog2', 'Right', 'centerlines', 'coords')
PIX_UM = 0.8
MICRONS_PER_PIXEL = 1.25


def load_counting_data(counts_file_path):
    """
    Load RBC counting data from the specified file.
    
    Args:
        counts_file_path: Path to the RBC counting CSV file
        
    Returns:
        DataFrame containing RBC counting data
    """
    try:
        counts_df = pd.read_csv(counts_file_path)
        print(f"Successfully loaded counting data with {len(counts_df)} entries")
        return counts_df
    except Exception as e:
        print(f"Error loading counting data: {e}")
        return None

def process_centerline_file(file_path):
    """
    Process a single centerline file to extract coordinates and radii.
    Calculate mean_diameter and bulk_diameter.
    
    Args:
        file_path: Path to the centerline CSV file
        
    Returns:
        Dictionary with video name, capillary number, mean_diameter, and bulk_diameter
    """
    try:
        # Extract video_name and capillary_number from filename
        filename = os.path.basename(file_path)
        # Example filename format: video_name_centerline_coords_capnum.csv
        parts = filename.split('_centerline_coords_')
        video_name = parts[0]
        capillary_number = parts[1].replace('.csv', '')
        
        # Load centerline data
        centerline_data = np.loadtxt(file_path, delimiter=',')
        
        # Extract coordinates and radii
        # Centerline format: row, col, radius
        radii = centerline_data[:, 2]
        
        # Calculate diameters (radius * 2) in microns
        diameters = radii * 2 * MICRONS_PER_PIXEL
        
        # Calculate mean diameter
        mean_diameter = np.mean(diameters)
        
        # Calculate bulk diameter using area-based method
        # Area = π * (diameter/2)^2
        total_area = np.sum(np.pi * (diameters/2)**2)
        # Equivalent diameter from total area
        bulk_diameter = 2 * np.sqrt(total_area / (np.pi * len(diameters)))
        filename = os.path.basename(file_path)
        filename = filename.replace('_centerline_coords_', '_kymograph_')
        filename = filename.replace('.csv', '.tiff')
        filename = filename.replace('24-07-29_', '')
        
        return {
            'Filename': filename,
            'Capillary': capillary_number,
            'Mean_Diameter': mean_diameter,
            'Bulk_Diameter': bulk_diameter,
            'Centerline_Length': len(centerline_data),
            'Mean_Radius': np.mean(radii) * MICRONS_PER_PIXEL,
            'Std_Radius': np.std(radii) * MICRONS_PER_PIXEL,
            'Std_Diameter': np.std(diameters)
        }
    except Exception as e:
        print(f"Error processing centerline file {file_path}: {e}")
        return None

def load_all_centerlines(coords_dir):
    """
    Load all centerline files from the specified directory.
    
    Args:
        coords_dir: Directory containing centerline coordinate files
        
    Returns:
        DataFrame containing centerline information
    """
    print(f"Loading centerline files from: {coords_dir}")
    
    # Get all centerline files
    centerline_files = glob.glob(os.path.join(coords_dir, "*_centerline_coords_*.csv"))
    print(f"Found {len(centerline_files)} centerline files")
    
    if len(centerline_files) == 0:
        print("No centerline files found. Check the path.")
        return None
    
    # Process each centerline file
    centerline_data = []
    for file_path in centerline_files:
        result = process_centerline_file(file_path)
        if result:
            centerline_data.append(result)
    
    # Create DataFrame
    centerline_df = pd.DataFrame(centerline_data)
    print(f"Successfully processed {len(centerline_df)} centerline files")
    
    return centerline_df

def merge_counting_and_centerline_data(counts_df, centerline_df):
    """
    Merge the RBC counting data with centerline diameter information.
    
    Args:
        counts_df: DataFrame containing RBC counting data
        centerline_df: DataFrame containing centerline information
        
    Returns:
        Merged DataFrame
    """
    print("\nMerging counting data with centerline information...")
    
   
    
    # Convert Capillary column to string in both dataframes for matching
    counts_df['Capillary'] = counts_df['Capillary'].astype(str)
    centerline_df['Capillary'] = centerline_df['Capillary'].astype(str)
    
    # Merge dataframes on Video and Capillary columns
    merged_df = pd.merge(counts_df, centerline_df, on=['Filename'], how='left')
    
    # Check for missing matches
    missing_diameter = merged_df[merged_df['Mean_Diameter'].isna()]
    if not missing_diameter.empty:
        print(f"Warning: {len(missing_diameter)} entries from counting data have no matching centerline information")
    
    return merged_df

def main():
    """
    Main function to load RBC counting data, process centerline files,
    merge the data, and save the result.
    """
    # Set up paths
    counts_file_path = os.path.join('H:', '240729', 'Frog2', 'Right', 'counts', 'final_predictions.csv')
    output_file_path = os.path.join('H:', '240729', 'Frog2', 'Right', 'counts', 'final_predictions_radii.csv')
    
    # Load RBC counting data
    counts_df = load_counting_data(counts_file_path)
    if counts_df is None:
        return 1
    
    # Load centerline data
    centerline_df = load_all_centerlines(COORDS_DIR)
    if centerline_df is None:
        return 1
    
    # Merge the data
    merged_df = merge_counting_and_centerline_data(counts_df, centerline_df)
    
    # Save the merged data
    merged_df.to_csv(output_file_path, index=False)
    print(f"Merged data saved to: {output_file_path}")
    
    # Generate some basic statistics
    print("\nBasic Statistics:")
    print(f"Total entries in merged data: {len(merged_df)}")
    print(f"Entries with diameter information: {merged_df['Mean_Diameter'].notna().sum()}")
    print(f"Mean diameter across all capillaries: {merged_df['Mean_Diameter'].mean():.2f} μm")
    
    return 0

if __name__ == "__main__":
    main() 