"""
Filename: centerline_lengths.py
------------------------------------------------------
This function takes a directory of centerline files and returns a DataFrame with the 
filenames and the number of rows in each file.

By: Marcus Forst
"""

import os, platform, time
import pandas as pd
from src.tools.parse_filename import parse_filename
import cv2
from matplotlib import pyplot as plt
import numpy as np
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

    metadata_folder = 'C:\\Users\\gt8mar\\capillary-flow\\metadata'
    # List to store file details
    file_details = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # print(filename)
            file_path = os.path.join(directory, filename)
            
            # Read the png file and count nonzero pixels
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # set all pixels with value > 0 to 1
            image[image > 0] = 1
            area = sum(sum(image))
            
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

    # Convert the list to a DataFrame
    area_df = pd.DataFrame(file_details)

    return area_df

def centerline_length_summary(directory):
    # List to store file details
    file_details = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            
            # Read the CSV file and count the rows
            try:
                df = pd.read_csv(file_path)
                centerline_length = len(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            # print(f"{filename}: {row_count} rows")
            participant, date, location, video, __ = parse_filename(filename)
            capillary = filename.split('.')[0].split('_')[-1]
            # Append the details to the list
            file_details.append({'Participant': participant, 'Date': date, 
                                 'Location': location, 'Video': video, 'Capillary': capillary,
                                 'Centerline Length': centerline_length})

    # Convert the list to a DataFrame
    centerline_df = pd.DataFrame(file_details)

    return centerline_df

def main(df, variable):
    #------------------------Histograms------------------------
    if variable == 'Age':
        point_variable = 'SYS_BP'
    else:
        point_variable = 'Age'

    # Create color map for 'Age' and 'SYS_BP'
    variable_color_map = create_color_map(df, variable)
    point_color_map = create_color_map(df, point_variable)


    # Calculate the median 'variable' for each participant and sort them
    median_variable_per_participant = df.groupby('Participant')[variable].median().sort_values()
    sorted_participant_indices = {participant: index for index, participant in enumerate(median_variable_per_participant.index)}
    
    # Plot histograms colored by 'Age' or 'SYS_BP', as specified by the variable argument
    plot_histograms(df, variable_color_map, point_color_map, variable, point_variable, 
                              participant_order=sorted_participant_indices)

def make_df(write = True):
    if platform.system() == 'Windows':
        # Define the directory
        centerlines_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results\\centerlines'
        area_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results\\segmented\\individual_caps_original'
    ticks = time.time()
    # Call the function
    centerline_df = centerline_length_summary(centerlines_folder)
    area_df = capillary_area_summary(area_folder)

    # Merge the DataFrames
    summary_df = pd.merge(centerline_df, area_df, how='left', on=['Participant', 'Date', 'Location', 'Video', 'Capillary'])

    # Make a column for Diameter (area/centerline length)
    summary_df['Diameter'] = summary_df['Area'] / summary_df['Centerline Length']

    
    # Print the DataFrame
    print(summary_df)
    # write the DataFrame to a csv
    if write:
        summary_df.to_csv('C:\\Users\\gt8mar\\capillary-flow\\results\\cap_diameters.csv', index=False)
    
    print(f"finished in {ticks-time.time()} seconds")
    return summary_df

if __name__ == '__main__':
    # Make or load the DataFrame
    # summary_df = make_df(write = False)
    size_df = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\results\\cap_diameters.csv')
    velocity_df = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\big_df.csv')

    # Merge the DataFrames
    summary_df = pd.merge(size_df, velocity_df, how='left', on=['Participant', 'Date', 'Location', 'Video', 'Capillary'])
    
    main(summary_df, 'Age')
    main(summary_df, 'SYS_BP')

    