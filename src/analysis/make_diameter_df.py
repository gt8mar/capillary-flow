"""
Filename: centerline_lengths.py
------------------------------------------------------
This function takes a directory of centerline files and returns a DataFrame with the 
filenames and the number of rows in each file.

By: Marcus Forst
"""

"""Module for creating and analyzing diameter data from capillary measurements.

This module processes centerline and area data from capillary measurements,
calculates various diameter metrics, and provides summary statistics.
It supports progress tracking and performance timing.

Example:
    >>> from src.analysis.make_diameter_df import main
    >>> diameter_df = main(write=True)
"""

# Standard library imports
import os
import time
import platform
from typing import Dict, List, Tuple, Optional

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# Local imports
from src.tools.parse_filename import parse_filename
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from src.analysis.update_velocities_2 import calculate_age_on_date
from src.analysis.plot_big import create_color_map

# Define a function to plot the histograms
def plot_histograms(df, hist_color_map, point_color_map, hist_var, point_var, participant_order):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    num_bins = 5  # Specify the number of bins for the histograms
    
    # Compute median values for the histogram variable for each participant
    median_values_hist = df.groupby('Participant')[hist_var].median()
    median_values_point = df.groupby('Participant')[point_var].median()

    for participant in participant_order:
        participant_data = df[df['Participant'] == participant]
        participant_index = participant_order[participant]
        
        # Calculate bins and frequencies
        diameters = participant_data['Diameter']
        bins = np.linspace(diameters.min(), diameters.max(), num_bins + 1)
        bin_indices = np.digitize(diameters, bins) - 1  # Bin index for each velocity

        # Normalize the bar heights to the total number of measurements for the participant
        total_measurements = len(diameters)
        bin_heights = np.array([np.sum(bin_indices == bin_index) for bin_index in range(num_bins)]) / total_measurements

        # Get the median value for the histogram variable for the participant
        hist_attribute_median = median_values_hist[participant]

        # Plot bars for each bin
        for bin_index, bar_height in enumerate(bin_heights):
            if bar_height == 0:
                continue
            color = hist_color_map[hist_attribute_median]
            ax.bar(participant_index + (bin_index - num_bins / 2) * 0.1, bar_height,
                   color=color, width=0.1, align='center')

    # Customize the plot
    ax.set_xlabel('Participant')
    ax.set_ylabel('Frequency of {hist_var}')
    ax.set_title(f'Histogram of diameters by Participant\nColored by {hist_var}')
    
    # Create secondary y-axis for the points
    ax2 = ax.twinx()
    for participant in participant_order:
        participant_data = df[df['Participant'] == participant]
        participant_index = participant_order[participant]
        
        # Get the attribute value for the points
        point_attribute_median = median_values_point[participant]
        # Check if median in the color map
        if point_attribute_median not in point_color_map:
            # If not, get the closest value
            closest_value = min(point_color_map.keys(), key=lambda x:abs(x-point_attribute_median))
            point_color = point_color_map[closest_value]
        else:
            point_color = point_color_map[point_attribute_median]
        
        # Plot the point
        ax2.plot(participant_index, point_attribute_median, 'X', color='red', markersize=10)         # could make this point color
    
    ax2.set_ylabel(f'{point_var} Value')

    # Set x-ticks to be the participant names
    ax.set_xticks(list(participant_order.values()))
    ax.set_xticklabels(list(participant_order.keys()))

    # Create a legend for the attribute
    hist_legend_elements = [Patch(facecolor=color, edgecolor='gray', label=label)
                       for label, color in hist_color_map.items()]
    ax.legend(handles=hist_legend_elements, title=hist_var, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Create a legend for the points
    point_legend_elements = [Patch(facecolor='red', edgecolor='red', label=point_var)]
    ax2.legend(handles=point_legend_elements, title=point_var, bbox_to_anchor=(1.15, 0.9), loc='upper left')
    plt.show()
    return 0

def capillary_area_summary(directory):
    """Summarizes capillary area data from segmentation files.
    
    Processes all segmentation files in the given directory, extracting
    area measurements for each capillary.
    
    Args:
        directory: Path to the directory containing segmentation files
        
    Returns:
        pd.DataFrame: DataFrame containing area data with columns for
        Participant, Date, Location, Video, Capillary, and Area
    """
    metadata_folder = 'C:\\Users\\gt8mar\\capillary-flow\\metadata'
    # List to store file details
    file_details = []

    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    total_files = len(csv_files)
    
    print(f"Found {total_files} area files to process")
    
    # Process each file with progress tracking
    for i, filename in enumerate(csv_files):
        # Display progress
        progress = (i + 1) / total_files * 100
        print(f"\rProcessing area files: {i+1}/{total_files} [{progress:.1f}%]", end="")
        
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Read the CSV file and count nonzero pixels
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        # set all pixels with value > 0 to 1
        image[image > 0] = 1
        area = sum(sum(image))
        
        # Parse the filename to get participant, date, location, video
        participant, date, location, video, __ = parse_filename(filename)
        # Get the metadata file name
        metadata_name = f'{participant}_{date}.xlsx'
        # Read in the metadata
        metadata = pd.read_excel(os.path.join(metadata_folder, metadata_name), sheet_name = 'Sheet1') 

        capillary = filename.split('.')[0].split('_')[-1]

        # Get the age from the first row of the metadata
        age = calculate_age_on_date(date, str(metadata['Birthday'].values[0]))
        sys_bp = metadata['BP'].values[0].split('/')[0]

        # Append the details to the list
        file_details.append({'Participant': participant, 'Date': date, 
                             'Location': location, 'Video': video, 'Capillary': capillary,
                             'Area': area, 'Age': age, 'SYS_BP': sys_bp})

    print("\nArea processing complete!")

    # Convert the list to a DataFrame
    area_df = pd.DataFrame(file_details)

    return area_df

def centerline_length_radius_summary(directory):
    """Summarizes centerline length and radius data from centerline files.
    
    Processes all centerline files in the given directory, extracting centerline
    length and radius statistics for each capillary.
    
    Args:
        directory: Path to the directory containing centerline files
        
    Returns:
        pd.DataFrame: DataFrame containing centerline length and radius data
        with columns for Participant, Date, Location, Video, Capillary,
        Centerline_Length, Mean_Radius, and Std_Radius
    """
    # List to store file details
    file_details = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            
            # Read the CSV file and count the rows
            try:
                df = pd.read_csv(file_path)
                # calculate centerline length using the first and second column (row and column values respectively)
                distances = []
                for row in df:
                    distance = np.sqrt((row['row'] - df['row'].values[0])**2 + (row['column'] - df['column'].values[0])**2)
                    distances.append(distance)
                centerline_length = sum(distances)
                # The radius is the third column
                radii = df.iloc[:, 2]
                # Calculate the mean radius
                mean_radius = np.mean(radii)
                # Calculate the standard deviation of the radius
                std_radius = np.std(radii)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            # print(f"{filename}: {row_count} rows")
            participant, date, location, video, __ = parse_filename(filename)
            capillary = filename.split('.')[0].split('_')[-1]
            # Append the details to the list
            file_details.append({'Participant': participant, 'Date': date, 
                                 'Location': location, 'Video': video, 'Capillary': capillary,
                                 'Centerline_Length': centerline_length, 'Mean_Radius': mean_radius,
                                 'Std_Radius': std_radius})

    # Convert the list to a DataFrame
    centerline_df = pd.DataFrame(file_details)
    
    return centerline_df

def main(write = True):
    """Main function to create and save the diameter summary DataFrame.
    
    Processes centerline and area data, merges them, and calculates diameter
    measurements. Optionally writes the results to a CSV file.
    
    Args:
        write: Boolean flag to determine if the DataFrame should be saved to disk
        
    Returns:
        pd.DataFrame: The final summary DataFrame with diameter measurements
    """
    if platform.system() == 'Windows':
        # Define the directory
        centerlines_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results\\centerlines'
        area_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results\\segmented\\individual_caps_original'
    
    # Start timing the execution
    start_time = time.time()
    print("Starting diameter data processing...")

    # Process centerline data with progress tracking
    print("Processing centerline data...")
    centerline_df = centerline_length_radius_summary(centerlines_folder)
    centerline_time = time.time()
    print(f"Centerline processing completed in {centerline_time - start_time:.2f} seconds")
    
    # Process area data with progress tracking
    print("Processing area data...")
    area_df = capillary_area_summary(area_folder)
    area_time = time.time()
    print(f"Area processing completed in {area_time - centerline_time:.2f} seconds")

    # Merge the DataFrames
    print("Merging data and calculating diameters...")
    summary_df = pd.merge(centerline_df, area_df, how='left', on=['Participant', 'Date', 'Location', 'Video', 'Capillary'])

    # Make a column for bulk diameter (area/centerline length), robust to if the centerline length is 0
    summary_df['Bulk_Diameter'] = np.where(
        summary_df['Centerline_Length'] > 0,
        summary_df['Area'] / summary_df['Centerline_Length'],
        0
    )

    # make column for diameters coming from centerline_length_radius_summary
    summary_df['Mean_Diameter'] = summary_df['Mean_Radius'] * 2
    summary_df['Std_Diameter'] = summary_df['Std_Radius'] * 2

    # Print the DataFrame
    print(summary_df)
    
    # Save the DataFrame if write is True
    if write:
        summary_df.to_csv('C:\\Users\\gt8mar\\capillary-flow\\results\\cap_diameters_areas.csv', index=False)
     # Calculate and print total execution time
    end_time = time.time()
    print(f"Total processing completed in {end_time - start_time:.2f} seconds")
    return summary_df

if __name__ == '__main__':
    main(write = True)

    