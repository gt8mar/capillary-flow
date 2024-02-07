import os, platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import skew, kurtosis  


# Function to create a color map for a given column
def create_color_map(df, column, cmap='viridis'):
    unique_values = sorted(df[column].unique())
    colors = sns.color_palette(cmap, len(unique_values))    
    return dict(zip(unique_values, colors))

def find_fwhm(bin_edges, bin_heights):
    max_height = max(bin_heights)
    half_max = max_height / 2

    # Find the left and right side of the peak at half maximum
    left_idx = np.where(bin_heights >= half_max)[0][0]
    right_idx = np.where(bin_heights >= half_max)[0][-1]

    # Full Width at Half Maximum
    fwhm = bin_edges[right_idx] - bin_edges[left_idx]
    return fwhm

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean)**2 / (2 * standard_deviation ** 2))

def calculate_fwhm(standard_deviation):
    # FWHM for a Gaussian distribution is calculated as 2 * sqrt(2 * ln(2)) * standard deviation
    return 2 * np.sqrt(2 * np.log(2)) * standard_deviation

def calculate_metrics(velocities):
    metrics = {}
    metrics['std_dev'] = np.std(velocities)
    metrics['skewness'] = skew(velocities)
    metrics['kurtosis'] = kurtosis(velocities, fisher=False)  # Fisher=False for Pearson's definition of kurtosis
    metrics['peakiness'] = max(velocities) / metrics['std_dev']  # Example definition for peakiness
    metrics['coeff_variation'] = metrics['std_dev'] / np.mean(velocities)
    return metrics

def plot_kurtosis(df, participant_order):
    # Create a plot for each metric
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    for participant in participant_order:
        participant_data = df[df['Participant'] == participant]
        participant_index = participant_order[participant]
        
        # Calculate metrics
        velocities = participant_data['Corrected Velocity']
        metrics = calculate_metrics(velocities)
        std_dev = metrics['std_dev']
        skewness = metrics['skewness']
        kurtosis_value = metrics['kurtosis']
        ax.plot(participant_index, kurtosis_value, '.', color='green', markersize=10)

    # Customize the plot
    ax.set_xlabel('Participant')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Kurtosis by Participant')

    # Set x-ticks to be the participant names
    ax.set_xticks(list(participant_order.values()))
    ax.set_xticklabels(list(participant_order.keys()))

    # Create a legend for the metrics
    metric_legend_elements = [Patch(facecolor='green', edgecolor='green', label='Kurtosis')]
    ax.legend(handles=metric_legend_elements, title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()
    



def plot_metrics(df, participant_order):
    # Create a plot for each metric
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    for participant in participant_order:
        participant_data = df[df['Participant'] == participant]
        participant_index = participant_order[participant]
        
        # Calculate metrics
        velocities = participant_data['Velocity']
        metrics = calculate_metrics(velocities)
        std_dev = metrics['std_dev']
        skewness = metrics['skewness']
        kurtosis_value = metrics['kurtosis']
        peakiness = metrics['peakiness']
        coeff_variation = metrics['coeff_variation']

        # Create a plot for each metric
        ax.plot(participant_index, std_dev, 'X', color='red', markersize=10)
        ax.plot(participant_index, skewness, 'X', color='blue', markersize=10)
        ax.plot(participant_index, kurtosis_value, 'X', color='green', markersize=10)
        ax.plot(participant_index, peakiness, 'X', color='purple', markersize=10)
        ax.plot(participant_index, coeff_variation, 'X', color='orange', markersize=10)

    # Customize the plot
    ax.set_xlabel('Participant')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Metrics by Participant')

    # Set x-ticks to be the participant names
    ax.set_xticks(list(participant_order.values()))
    ax.set_xticklabels(list(participant_order.keys()))

    # Create a legend for the metrics
    metric_legend_elements = [Patch(facecolor='red', edgecolor='red', label='Standard Deviation'),
                          Patch(facecolor='blue', edgecolor='blue', label='Skewness'),
                          Patch(facecolor='green', edgecolor='green', label='Kurtosis'),
                          Patch(facecolor='purple', edgecolor='purple', label='Peakiness'),
                          Patch(facecolor='orange', edgecolor='orange', label='Coefficient of Variation')]
    ax.legend(handles=metric_legend_elements, title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_loc_histograms(df, hist_color_map, point_color_map, hist_var, point_var):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    num_bins = 5  # Specify the number of bins for the histograms
    
    # Create a unique identifier for each participant-location combination
    df['Participant_Location'] = df['Participant'] + "-" + df['Location']
    unique_ids = df['Participant_Location'].unique()
    
    # Compute median values for the histogram variable for each participant-location
    median_values_hist = df.groupby('Participant_Location')[hist_var].median()
    median_values_point = df.groupby('Participant_Location')[point_var].median()

    # Create a mapping for x-axis positions
    x_positions = {id: index for index, id in enumerate(unique_ids)}

    for id in unique_ids:
        participant_data = df[df['Participant_Location'] == id]
        x_position = x_positions[id]

        # Calculate bins and frequencies
        velocities = participant_data['Velocity']
        bins = np.linspace(velocities.min(), velocities.max(), num_bins + 1)
        bin_indices = np.digitize(velocities, bins) - 1  # Bin index for each velocity

        # Normalize the bar heights
        total_measurements = len(velocities)
        bin_heights = np.array([np.sum(bin_indices == bin_index) for bin_index in range(num_bins)]) / total_measurements

        # Get the median value for the histogram variable
        hist_attribute_median = median_values_hist[id]

        # Plot bars for each bin
        for bin_index, bar_height in enumerate(bin_heights):
            if bar_height == 0:
                continue
            color = hist_color_map[hist_attribute_median]
            ax.bar(x_position + (bin_index - num_bins / 2) * 0.1, bar_height, color=color, width=0.1, align='center')

    # Customize the plot
    ax.set_xlabel('Participant-Location')
    ax.set_ylabel(f'Frequency of {hist_var}')
    ax.set_title(f'Histogram of Velocities by Participant and Location\nColored by {hist_var}')
    
    # Secondary y-axis for the points
    ax2 = ax.twinx()
    for id in unique_ids:
        x_position = x_positions[id]
        point_attribute_median = median_values_point[id]
        point_color = point_color_map[point_attribute_median]
        ax2.plot(x_position, point_attribute_median, 'X', color='red', markersize=10)

    ax2.set_ylabel(f'{point_var} Value')

    # Set x-ticks to be the participant-location names
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()), rotation=45)

    # Legends
    hist_legend_elements = [Patch(facecolor=color, edgecolor='gray', label=label) for label, color in hist_color_map.items()]
    ax.legend(handles=hist_legend_elements, title=hist_var, bbox_to_anchor=(1.05, 1), loc='upper left')

    point_legend_elements = [Patch(facecolor='red', edgecolor='red', label=point_var)]
    ax2.legend(handles=point_legend_elements, title=point_var, bbox_to_anchor=(1.15, 0.9), loc='upper left')

    plt.show()
    return 0


# Define a function to plot the histograms with dual-axis (for Gaussian FWHM and the point variable)
def plot_histograms(df, hist_color_map, point_color_map, hist_var, point_var, participant_order, fwhm = False):
    # Create the figure with two subplots
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

    num_bins = 5  # Specify the number of bins for the histograms

    # Compute median values for the histogram variable for each participant
    median_values = df.groupby('Participant')[hist_var].median()
    
    fwhm_values = {}  # To store FWHM values for each participant

    for participant in participant_order:
        participant_data = df[df['Participant'] == participant]
        participant_index = participant_order[participant]
        
        # Calculate bins and frequencies for the histogram variable
        velocities = participant_data['Velocity']
        bins = np.linspace(velocities.min(), velocities.max(), num_bins + 1)
        bin_heights, _ = np.histogram(velocities, bins=bins)
        bin_edges = 0.5 * (bins[:-1] + bins[1:])

        # Calculate metrics
        metrics = calculate_metrics(velocities)
        std_dev = metrics['std_dev']
        skewness = metrics['skewness']
        kurtosis_value = metrics['kurtosis']
        peakiness = metrics['peakiness']
        coeff_variation = metrics['coeff_variation']
        max_height = max(bin_heights)/len(velocities)

        # Display metrics on the plot (e.g., as text or in a separate legend)
        ax.text(participant_index, max_height, f'SD: {std_dev:.2f}\nSkew: {skewness:.2f}\nKurt: {kurtosis_value:.2f}\nPeakiness: {peakiness:.2f}\nCoeff. Var.: {coeff_variation:.2f}', ha='center')


        # Normalize and mirror the histogram
        total_measurements = len(velocities)
        normalized_bin_heights = bin_heights / total_measurements
        mirrored_velocities = np.concatenate([velocities, -velocities + 2 * velocities.mean()])

        # Fit a Gaussian to the mirrored data
        mirrored_bins = np.linspace(mirrored_velocities.min(), mirrored_velocities.max(), 2 * num_bins + 1)
        mirrored_bin_heights, _ = np.histogram(mirrored_velocities, bins=mirrored_bins, density=True)
        mirrored_bin_centers = 0.5 * (mirrored_bins[:-1] + mirrored_bins[1:])
        popt, _ = curve_fit(gaussian, mirrored_bin_centers, mirrored_bin_heights, p0=[velocities.mean(), 1, velocities.std()])

        # Calculate FWHM for the Gaussian fit
        fwhm_values[participant] = calculate_fwhm(popt[2])  # popt[2] is the standard deviation of the Gaussian fit

        # Plot bars for each bin
        for bin_index, bar_height in enumerate(normalized_bin_heights):
            if bar_height == 0:
                continue
            hist_attribute_median = median_values[participant]
            color = hist_color_map[hist_attribute_median]
            ax.bar(participant_index + (bin_index - num_bins / 2) * 0.1, bar_height, color=color, width=0.1, align='center')

    # Customize the primary y-axis
    ax.set_xlabel('Participant')
    ax.set_ylabel(f'Frequency of {hist_var}')
    ax.set_title(f'Dual-axis Histogram by Participant')

    # Create secondary y-axis for the points and FWHM
    ax2 = ax.twinx()
    for participant in participant_order:
        participant_index = participant_order[participant]

        if fwhm:
            # Plot the FWHM as a line
            fwhm_value = fwhm_values[participant]
            ax2.plot([participant_index - 0.25, participant_index + 0.25], [fwhm_value, fwhm_value], '-', color='green', lw=2)

        # Get the attribute value for the points
        point_attribute_value = df[df['Participant'] == participant][point_var].iloc[0]
        point_color = point_color_map[point_attribute_value]
        
        # Plot the point
        ax2.plot(participant_index, point_attribute_value, 'X', color='red', markersize=10)
    
    ax2.set_ylabel(f'{point_var} Value & FWHM')

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

def main(path, variable = 'Age'):
    df = pd.read_csv(path) # Load your data
    if variable == 'Age':
        point_variable = 'SYS_BP'
    else:
        point_variable = 'Age'

    # Create color map for 'Age' and 'SYS_BP'
    variable_color_map = create_color_map(df, variable)
    point_color_map = create_color_map(df, point_variable)


    # Calculate the median 'SYS_BP' for each participant and sort them
    median_variable_per_participant = df.groupby('Participant')[variable].median().sort_values()
    sorted_participant_indices = {participant: index for index, participant in enumerate(median_variable_per_participant.index)}


    
    # Plot histograms colored by 'Age'
    # plot_histograms(df, variable_color_map, point_color_map, variable, point_variable, 
    #                           participant_order=sorted_participant_indices)
    # plot_loc_histograms(df, variable_color_map, point_color_map, variable, point_variable)
    # plot_metrics(df, sorted_participant_indices)
    plot_kurtosis(df, sorted_participant_indices)







    # # Plot histograms colored by 'SYS_BP'
    # plot_histograms(df, bp_color_map, 'SYS_BP', axs)
    # plt.show()


# # Scatter plot for Age vs Velocity
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='Pressure', y='Velocity',hue = 'SYS_BP', data=df)
# # Color points by age

# plt.title('Pressure vs Velocity')
# plt.xlabel('Pressure')
# plt.ylabel('Velocity')
# plt.show()

# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# # Load data
# # df = pd.read_csv('your_data.csv')
# df = pd.read_csv('C:\\Users\\gt8ma\\capillary-flow\\results\\velocities\\velocities\\big_df.csv') # Load your data


# # Standardizing the features
# features = ['Velocity', 'Pressure', 'SYS_BP']
# x = df.loc[:, features].values
# x = StandardScaler().fit_transform(x)

# # PCA
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(x)

# # Create a DataFrame with the PCA results
# principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
# principalDf['Age'] = df['Age']  # Add the age column

# # Plotting
# plt.figure(figsize=(10,8))
# sns.scatterplot(x='principal component 1', y='principal component 2', hue='Age', data=principalDf, palette='viridis')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('2 Component PCA Colored by Age')
# plt.colorbar(scatter,label='Age')
# plt.show()
    
    return 0
    
if __name__ == '__main__':
    if platform.system() == 'Windows':
        if 'gt8mar' in os.getcwd():
            path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\big_df.csv'
        else:
            path = 'C:\\Users\\gt8ma\\capillary-flow\\results\\velocities\\velocities\\big_df.csv'
    else:
        path = '/hpc/projects/capillary-flow/results/velocities/big_df.csv'
    main(path, 'Age')
    main(path, 'SYS_BP')