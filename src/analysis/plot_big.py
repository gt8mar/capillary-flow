import os, platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns


# Function to create a color map for a given column
def create_color_map(df, column, cmap='rocket'):
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

# Define a function to plot the histograms
def plot_histograms(df, hist_color_map, point_color_map, hist_var, point_var, ax, participant_order):
    num_bins = 5  # Specify the number of bins for the histograms
    
    # Compute median values for the histogram variable for each participant
    median_values_hist = df.groupby('Participant')[hist_var].median()
    median_values_point = df.groupby('Participant')[point_var].median()

    for participant in participant_order:
        participant_data = df[df['Participant'] == participant]
        participant_index = participant_order[participant]
        
        # Calculate bins and frequencies
        velocities = participant_data['Velocity']
        bins = np.linspace(velocities.min(), velocities.max(), num_bins + 1)
        bin_indices = np.digitize(velocities, bins) - 1  # Bin index for each velocity

        # Normalize the bar heights to the total number of measurements for the participant
        total_measurements = len(velocities)
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
    ax.set_title(f'Histogram of Velocities by Participant\nColored by {hist_var}')
    
    # Create secondary y-axis for the points
    ax2 = ax.twinx()
    for participant in participant_order:
        participant_data = df[df['Participant'] == participant]
        participant_index = participant_order[participant]
        
        # Get the attribute value for the points
        point_attribute_median = median_values_point[participant]
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


    # Create the figure with two subplots
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

    # Plot histograms colored by 'Age'
    plot_histograms(df, variable_color_map, point_color_map, variable, point_variable, ax, 
                              participant_order=sorted_participant_indices)
    plt.show()

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
    main(path, 'SYS_BP')