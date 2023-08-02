"""
Filename: centerline_hist.py
------------------------------------------------------
This program plots a histogram of the centerline lengths
for a given participant.

By: Marcus Forst
"""

import os, sys, gc, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
from src.tools.load_csv_list import load_csv_list

def parse_filename(filename):
    """
    This function parses a centerline filename into its components.

    Args:
        filename (str): the filename to be parsed
    Returns:
        set_number (str): the set number
        part_number (str): the participant number
        date (str): the date of the video
        video_id (str): the video id
        zeroes_number (str): the number of zeroes in the centerline
    """
    # Split the filename into parts based on underscores
    parts = filename.split('_')

    # Extract the required components
    set_number = parts[0]
    part_number = parts[1]
    date = parts[2]
    video_id = parts[3]
    coords_number = parts[-2]
    zeroes_number = parts[-1].split('.')[0]  # Remove the ".csv" extension and get the last part

    return set_number, part_number, date, video_id, zeroes_number
 
# Function to extract the video ID from the filename
def extract_video_id(filename):
    parts = filename.split('_')
    print(parts[3])
    return parts[4]

# Function to plot histograms for the count of rows in each video DataFrame
def plot_video_histograms(csv_folder):
    """
    This function plots histograms for the count of rows in each video DataFrame.

    This doesn't work because the csv files are not named in a way that allows us to compare capillaries to other capillaries
    """
    # Initialize a dictionary to store dataframes for each video and their row counts
    video_data = {}

    # Loop through all files in the folder
    for filename in os.listdir(csv_folder):
        if filename.endswith('.csv'):
            # Read the CSV file into a pandas dataframe
            file_path = os.path.join(csv_folder, filename)

            # TODO: Extract the capillary ID from the filename or the metadata

            video_id = extract_video_id(filename)
            df = pd.read_csv(file_path)

            # Get the count of rows in the dataframe
            row_count = df.shape[0]

            # Add the row count to the dictionary with video_id as the key
            if video_id not in video_data:
                video_data[video_id] = [row_count]
            else:
                video_data[video_id].append(row_count)

    # Plot histograms for the count of rows in each video DataFrame
    plt.figure(figsize=(10, 6))
    for video_id, row_counts in video_data.items():
        plt.hist(row_counts, bins=8, alpha=0.5, label=video_id)

    plt.xlabel('Number of Rows')
    plt.ylabel('Frequency')
    plt.title('Histogram Comparison for the Number of Rows in Videos')
    plt.legend()
    plt.show()

    return 0

def centerline_hist():
    if platform.system() == 'Windows':
        centerlines_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results\\centerlines'
    else:
        centerlines_folder = '/hpc/projects/capillary-flow/results/centerlines'
    centerlines = load_csv_list(centerlines_folder)
    lengths = []
    for centerline in centerlines:
        lengths.append(centerline.shape[0])
    plt.hist(lengths, bins = 20)
    plt.show()
    # plot_video_histograms(centerlines_folder)
    return 0

if __name__ == "__main__":
    centerline_hist()