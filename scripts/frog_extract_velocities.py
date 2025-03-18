"""
Filename: scripts/frog_extract_velocities.py

Script to extract velocity CSV files from the source file structure and copy them to an output directory
with standardized naming convention.
"""

import os
import shutil
from pathlib import Path
import logging
import re


def main():
    """
    Main function to copy velocity CSV files from the source filetree to the output directory
    and rename them to the format:
    {date}_{frog_id}_{right_or_left}_{velocity_filename}

    Source Filetree:
    - f"H:\\frog\\data\\{date}
        - f"H:\\frog\\data\\{date}\\{frog_id}
            - f"H:\\frog\\data\\{date}\\{frog_id}\\{right_or_left}
                - f"H:\\frog\\data\\{date}\\{frog_id}\\{right_or_left}\\velocities\\{velocity_filename}
            - ...
        - ...    

    Folder naming examples:
    date: YYMMDD
    frog_id: Frog1, Frog2, etc.
    right_or_left: Right, Left
    velocity_filename: velocity_data.csv, etc.

    Output directory:
    - f"D:\\frog\\results\\velocities

    New filename:
    - f"D:\\frog\\results\\velocities\\{date}_{frog_id}_{right_or_left}_{velocity_filename}"
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define source and output directories
    source_root = Path("H:\\frog\\data")
    output_dir = Path("D:\\frog\\results\\velocities")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Counter for tracking processed files
    processed_count = 0
    
    # Traverse the source directory structure
    for date_dir in source_root.iterdir():
        if not date_dir.is_dir():
            continue
        
        # Check if the date directory is in the format YYMMDD
        if not re.match(r'^\d{6}$', date_dir.name):
            logging.warning(f"Skipping non-date directory: {date_dir.name}")
            continue
        
        date = date_dir.name
        logging.info(f"Processing date directory: {date}")
        
        for frog_dir in date_dir.iterdir():
            if not frog_dir.is_dir():
                continue
            
            frog_id = frog_dir.name
            logging.info(f"Processing frog directory: {frog_id}")
            
            for side_dir in frog_dir.iterdir():
                if not side_dir.is_dir():
                    continue
                
                side = side_dir.name  # Right or Left
                if side not in ["Right", "Left"]:
                    continue
                
                # Check for velocities directory
                velocities_dir = os.path.join(side_dir, "velocities")
                if not os.path.exists(velocities_dir) or not os.path.isdir(velocities_dir):
                    logging.warning(f"No velocities directory found in {side_dir}")
                    continue
                
                # Process all velocity CSV files
                for velocity_file in os.listdir(velocities_dir):
                    if not velocity_file.endswith(".csv"):
                        continue
                    
                    velocity_filename = velocity_file
                    
                    # Create the new standardized filename
                    new_filename = f"{date}_{frog_id}_{side}_{velocity_filename}"
                    output_path = os.path.join(output_dir, new_filename)
                    
                    # Create the full source path
                    source_path = os.path.join(velocities_dir, velocity_file)
                    
                    # Copy the file
                    try:
                        shutil.copy2(source_path, output_path)
                        processed_count += 1
                        logging.info(f"Copied: {source_path} -> {output_path}")
                    except Exception as e:
                        logging.error(f"Error copying {source_path}: {e}")
    
    logging.info(f"Processing complete. Total files processed: {processed_count}")


if __name__ == "__main__":
    main() 