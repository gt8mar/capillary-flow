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

# Define a function to plot the histograms
def plot_histograms(df, color_map, title, ax):
    # Assign each participant a unique index
    participant_indices = {participant: index for index, participant in enumerate(df['Participant'].unique())}
    num_bins = 5  # Specify the number of bins for the histograms
    
    for participant in df['Participant'].unique():
        participant_data = df[df['Participant'] == participant]
        participant_index = participant_indices[participant]
        
        # Calculate bins and frequencies
        velocities = participant_data['Velocity']
        bins = np.linspace(velocities.min(), velocities.max(), num_bins + 1)
        bin_indices = np.digitize(velocities, bins) - 1  # Bin index for each velocity

        # Normalize the bar heights to the total number of measurements for the participant
        total_measurements = len(velocities)
        bin_heights = np.array([np.sum(bin_indices == bin_index) for bin_index in range(num_bins)]) / total_measurements

        # Plot bars for each bin
        for bin_index, bar_height in enumerate(bin_heights):
            if bar_height == 0:
                continue
            attribute_value = participant_data.iloc[0][title]
            color = color_map[attribute_value]
            ax.bar(participant_index + (bin_index - num_bins / 2) * 0.1, bar_height,
                   color=color, width=0.1, align='center')

    # Customize the plot
    ax.set_xlabel('Participant')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Histogram of Velocities by Participant\nColored by {title}')
    ax.set_xticks(list(participant_indices.values()))
    ax.set_xticklabels(list(participant_indices.keys()))

    # Create a legend for the attribute
    legend_elements = [Patch(facecolor=color, edgecolor='gray', label=label)
                       for label, color in color_map.items()]
    ax.legend(handles=legend_elements, title=title, bbox_to_anchor=(1.05, 1), loc='upper left')

def main(path):
    df = pd.read_csv(path) # Load your data

    # Create color maps for 'Age' and 'SYS_BP'
    age_color_map = create_color_map(df, 'Age')
    bp_color_map = create_color_map(df, 'SYS_BP')

    # Create the figure with two subplots
    fig, axs = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

    # # Plot histograms colored by 'Age'
    # plot_histograms(df, age_color_map, 'Age', axs)
    # plt.show()

    # Plot histograms colored by 'SYS_BP'
    plot_histograms(df, bp_color_map, 'SYS_BP', axs)
    plt.show()


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
    main(path)