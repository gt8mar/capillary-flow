import os, platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
import seaborn as sns
from scipy.integrate import simps, trapezoid
from scipy.stats import skew, kurtosis, wilcoxon, mannwhitneyu
import statsmodels.api as sm
from src.tools.parse_filename import parse_filename

def calculate_metrics(velocities):
    # Remove NaN values from the velocities
    velocities = velocities.dropna()
    metrics = {}
    metrics['std_dev'] = np.std(velocities)
    metrics['skewness'] = skew(velocities)
    metrics['kurtosis'] = kurtosis(velocities, fisher=False)  # Fisher=False for Pearson's definition of kurtosis
    metrics['peakiness'] = max(velocities) / metrics['std_dev']  # Example definition for peakiness
    metrics['coeff_variation'] = metrics['std_dev'] / np.mean(velocities)
    return metrics

# Function to create a color map for a given column
def create_color_map(df, column, cmap='viridis'):
    unique_values = sorted(df[column].unique())
    colors = sns.color_palette(cmap, len(unique_values))  
    return dict(zip(unique_values, colors))

def compare_participants(df1, df2):
    """ 
    Compare the participants in two DataFrames and return a list of participants that are in one DataFrame 
    but not the other, or have different counts in the two DataFrames.

    Args:
        df1 (DataFrame): the first DataFrame
        df2 (DataFrame): the second DataFrame
    
    Returns:
        
    """
    # Group by 'Participant' and 'Capillary' and count the number of rows for each group
    df1_grouped = df1.groupby(['Participant', 'Video', 'Capillary']).size()
    df2_grouped = df2.groupby(['Participant', 'Video', 'Capillary']).size()

    # Group by 'Participant' and count the total number of rows for each participant
    df1_participant_counts = df1['Participant'].value_counts()
    df2_participant_counts = df2['Participant'].value_counts()

    different_rows = []
    different_participants = []


    for row in df1_grouped.index:
        if row not in df2_grouped or df1_grouped[row] != df2_grouped[row]:
            different_rows.append(row)

    for row in df2_grouped.index:
        if row not in df1_grouped:
            different_rows.append(row)
    
    # Check for different total counts for participants
    for participant in df1_participant_counts.index:
        if participant not in df2_participant_counts or df1_participant_counts[participant] != df2_participant_counts[participant]:
            different_participants.append(participant)

    for participant in df2_participant_counts.index:
        if participant not in df1_participant_counts:
            different_participants.append(participant)

    return different_rows, different_participants

def plot_pca(df):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    features = ['Velocity', 'Pressure', 'SYS_BP']
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    # Create a DataFrame with the PCA results
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    principalDf['Age'] = df['Age']  # Add the age column

    # Plotting
    plt.figure(figsize=(10,8))
    scatter = sns.scatterplot(x='principal component 1', y='principal component 2', hue='Age', data=principalDf, palette='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2 Component PCA Colored by Age')
    plt.colorbar(scatter,label='Age')
    plt.show()
    return 0
def plot_velocity_vs_pressure(df, hue = 'Age'):
    # plot velocity vs diameter scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Diameter', y='Corrected Velocity',hue = hue, data=df)
    plt.title('Diameter vs Velocity')
    plt.xlabel('Diameter')
    plt.ylabel('Velocity')
    plt.show()
    return 0
def plot_pressure_vs_diameter(df, hue = 'Age'):
    # Scatter plot for Age vs Velocity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Pressure', y='Corrected Velocity', hue = hue, data=df)
    plt.title('Pressure vs Velocity')
    plt.xlabel('Pressure')
    plt.ylabel('Velocity')
    plt.show()
    return 0

def plot_loc_histograms(df, variable, metrics = False):
    if variable == 'Age':
        point_variable = 'SYS_BP'
    else:
        point_variable = 'Age'

    # Create color map for 'Age' and 'SYS_BP'
    variable_color_map = create_color_map(df, variable)
    point_color_map = create_color_map(df, point_variable)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    num_bins = 5  # Specify the number of bins for the histograms
    
    # Create a unique identifier for each participant-location combination
    df['Participant_Location'] = df['Participant'] + "-" + df['Location']
    unique_ids = df['Participant_Location'].unique()
    
    # Compute median values for the histogram variable for each participant-location
    median_values_hist = df.groupby('Participant_Location')[variable].median()
    median_values_point = df.groupby('Participant_Location')[point_variable].median()

    # Create a mapping for x-axis positions
    x_positions = {id: index for index, id in enumerate(unique_ids)}

    for id in unique_ids:
        participant_data = df[df['Participant_Location'] == id]
        x_position = x_positions[id]

        # Calculate bins and frequencies
        velocities = participant_data['Corrected Velocity']
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
            color = variable_color_map[hist_attribute_median]
            ax.bar(x_position + (bin_index - num_bins / 2) * 0.1, bar_height, color=color, width=0.1, align='center')

    # Customize the plot
    ax.set_xlabel('Participant-Location')
    ax.set_ylabel(f'Frequency of velocity')
    ax.set_title(f'Histogram of Velocities by Participant and Location\nColored by {variable}')
    
    # Secondary y-axis for the points
    ax2 = ax.twinx()
    for id in unique_ids:
        x_position = x_positions[id]
        point_attribute_median = median_values_point[id]
        point_color = point_color_map[point_attribute_median]
        ax2.plot(x_position, point_attribute_median, 'X', color='red', markersize=10)

    ax2.set_ylabel(f'{point_variable} Value')

    # Set x-ticks to be the participant-location names
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()), rotation=45)

    # Legends
    hist_legend_elements = [Patch(facecolor=color, edgecolor='gray', label=label) for label, color in variable_color_map.items()]
    ax.legend(handles=hist_legend_elements, title=variable, bbox_to_anchor=(1.05, 1), loc='upper left')

    point_legend_elements = [Patch(facecolor='red', edgecolor='red', label=point_variable)]
    ax2.legend(handles=point_legend_elements, title=point_variable, bbox_to_anchor=(1.15, 0.9), loc='upper left')

    plt.show()
    return 0
def plot_histograms(df, variable = 'Age', diam_slice = None, normalize_bins = 'Total', gradient = True):
    """
    Plot histograms of the velocities for each participant, colored by the specified variable.

    Args:
        df (DataFrame): the DataFrame to be plotted
        variable (str): the variable to color the histograms by
        diam_slice (str): the slice of the DataFrame to be plotted. Default is None. Options are 'smaller' and 'larger'
        normalize_bins (str): the method for normalizing the bin heights. Options are 'Total' and 'Participant'
    Returns:
        0 if successful
    """
    
    #------------------------Histograms------------------------
    if variable == 'Age':
        point_variable = 'SYS_BP'
    else:
        point_variable = 'Age'

    if gradient:
        # Determine the range of the variable for the gradient
        variable_range = (df[variable].min(), df[variable].max())
        norm = mcolors.Normalize(vmin=variable_range[0], vmax=variable_range[1])
        scalar_map = ScalarMappable(norm=norm, cmap='viridis')  
    else:
        # Create color map for 'Age' and 'SYS_BP'
        variable_color_map = create_color_map(df, variable)
        point_color_map = create_color_map(df, point_variable)

    # Calculate the median 'variable' for each participant and sort them
    median_variable_per_participant = df.groupby('Participant')[variable].median().sort_values()
    participant_order = {participant: index for index, participant in enumerate(median_variable_per_participant.index)}

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    num_bins = 5  # Specify the number of bins for the histograms
    
    # Compute median values for the histogram variable for each participant
    median_values_hist = df.groupby('Participant')[variable].median()
    median_values_point = df.groupby('Participant')[point_variable].median()

    for participant in participant_order:
        participant_data = df[df['Participant'] == participant]
        participant_index = participant_order[participant]
        participant_velocities = participant_data['Corrected Velocity']
        max_velocity = participant_velocities.max()
        if max_velocity > 2000:
            bins = [-0.0001, 5, calc_vel_bins(15), calc_vel_bins(30), calc_vel_bins(45), calc_vel_bins(60), calc_vel_bins(75), 2000, max_velocity+1]
        else:
            bins = [-0.0001, 5, calc_vel_bins(15), calc_vel_bins(30), calc_vel_bins(45), calc_vel_bins(60), calc_vel_bins(75), 2000]
        
        # bins = np.linspace(velocities.min(), velocities.max(), num_bins + 1)
        # bins = [0, 5, 55, 161, df['Corrected Velocity'].max()]
        num_bins = len(bins) - 1
        bin_indices = np.digitize(participant_velocities, bins) - 1  # Bin index for each velocity

        # Normalize the bar heights to the total number of measurements for the participant
        total_measurements = len(participant_velocities)
        bin_heights = np.array([np.sum(bin_indices == bin_index) for bin_index in range(num_bins)])#/ total_measurements

        # Get the median value for the histogram variable for the participant
        hist_attribute_median = median_values_hist[participant]

        # Plot bars for each bin
        for bin_index, bar_height in enumerate(bin_heights):
            if bar_height == 0:
                continue
            if gradient:
                color = scalar_map.to_rgba(hist_attribute_median)
            else:
                color = variable_color_map[hist_attribute_median]
            ax.bar(participant_index + (bin_index - num_bins / 2) * 0.1, bar_height,
                   color=color, width=0.1, align='center')

    # Customize the plot
    ax.set_xlabel('Participant')
    ax.set_ylabel(f'Frequency of velocity')
    if diam_slice == 'smaller':
        ax.set_title(f'Histogram of Velocities by Participant\nColored by {variable} \n(Smaller Diameters)')
    elif diam_slice == 'larger':
        ax.set_title(f'Histogram of Velocities by Participant\nColored by {variable} \n(Larger Diameters)')
    else:
        ax.set_title(f'Histogram of Velocities by Participant\nColored by {variable}')
    
    # Create secondary y-axis for the points
    ax2 = ax.twinx()
    for participant in participant_order:
        participant_data = df[df['Participant'] == participant]
        participant_index = participant_order[participant]
        
        # Get the attribute value for the points
        point_attribute_median = median_values_point[participant]

        # if point_attribute_median has a decimal, round to the nearest whole number
        if point_attribute_median % 1 >= 0.5:
            point_attribute_median = np.ceil(point_attribute_median)
        else:
            point_attribute_median = np.floor(point_attribute_median)
        
        # Plot the point
        ax2.plot(participant_index, point_attribute_median, 'o', color='red', markersize=5)         # could make this point color
    
    ax2.set_ylabel(f'{point_variable} Value')

    # Set x-ticks to be the participant names
    ax.set_xticks(list(participant_order.values()))
    ax.set_xticklabels(list(participant_order.keys()))

    if gradient:
        hist_legend_elements = [Patch(facecolor=scalar_map.to_rgba(value), edgecolor='gray', label=value)
                                 for value in median_variable_per_participant.values]
    else:
        # Create a legend for the attribute
        hist_legend_elements = [Patch(facecolor=color, edgecolor='gray', label=label)
                       for label, color in variable_color_map.items()]
    ax.legend(handles=hist_legend_elements, title=variable, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Create a legend for the points
    point_legend_elements = [Patch(facecolor='red', edgecolor='red', label=point_variable)]
    ax2.legend(handles=point_legend_elements, title=point_variable, bbox_to_anchor=(1.15, 0.9), loc='upper left')
    plt.show()
    return 0
def plot_velocities(participant_df, write=False):
    participant_df = participant_df.copy()
    # Assuming there's a 'Location' column or similar to distinguish the same capillary used in different contexts
    # If not, you might need to create a composite key or a unique identifier based on your specific needs
    participant_df['Capillary_Location'] = participant_df['Capillary'].astype(str) + '_' + participant_df['Location'].astype(str)
    
    # Group the data by the new 'Capillary_Location'
    grouped_df = participant_df.groupby('Capillary_Location')
    # Get the unique capillary-location identifiers
    capillary_locations = participant_df['Capillary_Location'].unique()
    participant = participant_df['Participant'].unique()[0]

    # Calculate the number of plots and rows needed
    num_plots = len(capillary_locations)
    num_rows = (num_plots + 3) // 4

    # Adjust the height here; you can increase the multiplier as needed
    # For example, use 3 or 4 instead of 2 to make each plot taller
    fig_height = 3 * num_rows  # Increase the multiplier to give more space

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, fig_height), sharey=True, sharex=True)
    axes = axes.flatten()

    for i, capillary_location in enumerate(capillary_locations):
        capillary_data = grouped_df.get_group(capillary_location).copy()
        ax = axes[i]

        # Create 'Up/Down' column based on pressure changes
        capillary_data.loc[:, 'Up/Down'] = 'Up'
        max_pressure = capillary_data['Pressure'].max()
        max_index = capillary_data['Pressure'].idxmax()
        capillary_data.loc[max_index:, 'Up/Down'] = 'Down'

        # Separate data for plotting
        data_up = capillary_data[capillary_data['Up/Down'] == 'Up']
        data_down = capillary_data[capillary_data['Up/Down'] == 'Down']

        # Plot data
        ax.plot(data_up['Pressure'], data_up['Corrected Velocity'], marker='o', linestyle='-', label='Increase in Pressure')
        ax.plot(data_down['Pressure'], data_down['Corrected Velocity'], color='purple', marker='o', linestyle='-', label='Decrease in Pressure')
        
        ax.set_xlabel('Pressure (psi)')
        ax.set_ylabel('Velocity (um/s)')
        ax.set_title(f'{participant} {capillary_location}')
        ax.grid(True)
        ax.legend()

    # Remove unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    if write:
        plt.savefig(f'C:\\Users\\gt8mar\\capillary-flow\\results\\{participant}_fav_cap_v.png')
        plt.close()
    else:
        plt.show()

    return 0
def plot_caps_by_size(summary_df):
    """
    Plot the histograms of velocities for the 9 smallest diameter participants and the 9 largest diameter participants.

    Args:
        summary_df (DataFrame): the DataFrame to be plotted
    Returns:
        0 if successful
    """
    # find the median diameter for each participant
    median_diameter_per_participant = summary_df.groupby('Participant')['Diameter'].median().sort_values()
    sorted_participant_indices = {participant: index for index, participant in enumerate(median_diameter_per_participant.index)}

    # choose the 9 smallest diameter participants: (0-8)
    smaller_diameters = list(sorted_participant_indices.keys())[0:9]
    larger_diameters = list(sorted_participant_indices.keys())[9:19]

    # slice summary_df into smaller and larger diameter dfs
    smaller_diameter_df = summary_df[summary_df['Participant'].isin(smaller_diameters)]
    larger_diameter_df = summary_df[summary_df['Participant'].isin(larger_diameters)]

    # plot the histogram of velocities for each participant in sliced dfs
    main(smaller_diameter_df, 'Age', diam_slice = 'smaller')
    main(larger_diameter_df, 'Age', diam_slice = 'larger')
    main(smaller_diameter_df, 'SYS_BP', diam_slice = 'smaller')
    main(larger_diameter_df, 'SYS_BP', diam_slice = 'larger')
    return 0
def plot_median_diameter(summary_df):
    """
    Plot the median diameter for each participant.

    Args:
        summary_df (DataFrame): the DataFrame to be plotted, containing 'Participant', 'Diameter', and 'Age' columns
    Returns:
        0 if successful
    """
    # find the median diameter for each participant
    median_diameter_per_participant = summary_df.groupby('Participant')['Diameter'].median().sort_values()
    sorted_participant_indices = {participant: index for index, participant in enumerate(median_diameter_per_participant.index)}

    # plot the median diameter for each participant
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    # make blue bars hollow and the edges thick
    ax2 = ax.twinx()
    ax.bar(sorted_participant_indices.values(), median_diameter_per_participant.values, width=0.5)
    # add a red dot for the age of each participant on second y-axis
    ax2.plot(sorted_participant_indices.values(), summary_df.groupby('Participant')['Age'], '.', color='black', markersize=10)
    ax.set_xlabel('Participant')
    ax.set_ylabel('Median Diameter')
    ax2.set_ylabel('Age')
    ax.set_title('Median Diameter for Each Participant')
    ax.set_xticks(list(sorted_participant_indices.values()), list(sorted_participant_indices.keys()))
    plt.show()
    return 0
def compile_metadata():
    metadata_folder = 'C:\\Users\\gt8mar\\capillary-flow\\metadata'
    # Read the metadata files if they are csvs
    metadata_files = [f for f in os.listdir(metadata_folder) if f.endswith('.xlsx')]
    metadata_dfs = [pd.read_excel(os.path.join(metadata_folder, f)) for f in metadata_files]
    metadata_df = pd.concat(metadata_dfs)

    # make slice of metadata_df with only bp measurements
    non_bp_metadata = metadata_df[~metadata_df['Video'].str.contains('bp')]
   
    # add 'loc' and a leading zero to the location column
    non_bp_metadata['Location'] = 'loc' + non_bp_metadata['Location'].astype(str).str.zfill(2)

    # Convert 'Video' identifiers to integers for comparison
    non_bp_metadata['VideoID'] = non_bp_metadata['Video'].str.extract('(\d+)').astype(int)

    # remove all part09 videos greater than vid59:
    non_bp_metadata = non_bp_metadata[~((non_bp_metadata['Participant'] == 'part09') & (non_bp_metadata['VideoID'] > 59))]

    # keep only participant, date, location, and video columns
    non_bp_metadata = non_bp_metadata[['Participant', 'Date', 'Location', 'Video']]
    return non_bp_metadata

def check_inserted_rows(summary_df):
     # if the row has 'inserted' in the notes column, that means that the area is the same as the original area from the 'Capillary' column
    condition_inserted = summary_df['Notes'].str.contains('inserted')
    # print the rows that have 'inserted' in the notes column
    print(summary_df[condition_inserted][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Capillary_new', 'Area']])

    # print all part26 loc01 rows
    print(summary_df[(summary_df['Participant'] == 'part26') & (summary_df['Location'] == 'loc01') & (summary_df['Video'] == 'vid05')][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Capillary_new', 'Area', 'Corrected Velocity', 'Diameter']])
    return 0
def handle_dotted_evac(summary_df):
    # Fill NaN values in the 'Notes' column with an empty string
    summary_df['Notes'] = summary_df['Notes'].fillna('')
    # Now it is time to handle the added rows in velocity_df

    # if the row has 'dotted' in the notes column, that means that the velocity and area are zero.
    # if the row has 'evac' in the notes column, that means that the velocity is zero and the area is zero.

    # Condition to find rows with 'NaN' in 'Area' and 'dotted' in 'Notes'
    condition = summary_df['Area'].isna() & summary_df['Notes'].str.contains('dotted')
    condition_evac = summary_df['Area'].isna() & summary_df['Notes'].str.contains('evac')
    summary_df.loc[condition, 'Area'] = 0
    summary_df.loc[condition, 'Diameter'] = 0
    summary_df.loc[condition, 'Corrected Velocity'] = 0
    summary_df.loc[condition, 'Centerline'] = 0
    summary_df.loc[condition_evac, 'Area'] = 0
    summary_df.loc[condition_evac, 'Diameter'] = 0
    summary_df.loc[condition_evac, 'Corrected Velocity'] = 0
    summary_df.loc[condition_evac, 'Centerline'] = 0
    return summary_df
def merge_vel_size(verbose=False):
    size_df = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\results\\cap_diameters.csv')
    velocity_df = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\big_df - Copy.csv')
    velocity_df_old = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\big_df.csv')
    metadata_df = compile_metadata()
    print(metadata_df.head)

    # modify size_df to remove all bp measurements
    print(f'size_df shape: {size_df.shape}')
    
    # remove 'bp' from the video column
    size_df['Video'] = size_df['Video'].str.replace('bp', '')

    # use outer merge to find the rows in size_df that are not bp measurements
    size_df = size_df.merge(metadata_df, on=['Participant', 'Date', 'Location', 'Video'], how='inner', indicator=False)
    print(f'new size_df shape: {size_df.shape}')
    print(size_df.head())

    different_rows, different_participants = compare_participants(size_df, velocity_df)

    # save for testing
    # size_df.to_csv('C:\\Users\\gt8mar\\capillary-flow\\size_test.csv', index=False)
    # velocity_df.to_csv('C:\\Users\\gt8mar\\capillary-flow\\velocity_test.csv', index=False)

    # remove part22 and part23 from different rows
    different_rows = [row for row in different_rows if row[0] != 'part22' and row[0] != 'part23']
    print(different_rows)
    print(different_participants)

    # pd.set_option('display.max_rows', None)

    # print(velocity_df[velocity_df['Participant'] == 'part15'][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Corrected Velocity']])
    # velocity_part15_shape = velocity_df[velocity_df['Participant'] == 'part15'].shape
    # print(f'Velocity df shape: {velocity_part15_shape}')
    # print(size_df[size_df['Participant'] == 'part15'][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Diameter']])
    # size_part15_shape = size_df[size_df['Participant'] == 'part15'].shape
    # print(f'Size df shape: {size_part15_shape}')

    # remove SYS_BP column from size_df
    size_df = size_df.drop(columns=['SYS_BP'])

    # Merge the DataFrames
    summary_df = pd.merge(size_df, velocity_df, how='outer',on=['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Age'], indicator=True)
    
    summary_df = handle_dotted_evac(summary_df)

    if verbose: 
        check_inserted_rows(summary_df)
    
    # print any rows where area is NaN
    print(summary_df[summary_df['Area'].isna()][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Area', 'Corrected Velocity', 'Diameter']])
    
    # save summary_df to csv
    summary_df.to_csv('C:\\Users\\gt8mar\\capillary-flow\\summary_df_test.csv', index=False)
    return summary_df
def calc_vel_bins(theta):
    v_tan = np.tan(np.radians(theta))*2.44*(227.8/2)
    return v_tan

def plot_and_calculate_area(df, method='trapezoidal', plot = False, normalize = False, verbose=False):
    """
    Plot 'Pressure' vs. 'Corrected Velocity' from a DataFrame and calculate the area under the curve.
    
    Args:
        df (DataFrame): pressure and velocity data from a specific capillary location and up or down run. 
        method (str): The method to be used for the area calculation. Options are 'trapezoidal' and 'simpson'.
    
    Returns:
        area (float): The area under the curve calculated using the specified method.
    """
    df = df.copy()
    # Check if required columns are in the DataFrame
    if 'Pressure' not in df.columns or 'Corrected Velocity' not in df.columns:
        raise ValueError("DataFrame must contain 'Pressure' and 'Corrected Velocity' columns")
    
    # Normalization and hysteresis plotting
    if normalize:
        max_velocity = df['Corrected Velocity'].max()
        if max_velocity > 2000:
            bins = [-0.0001, 5, calc_vel_bins(15), calc_vel_bins(30), calc_vel_bins(45), calc_vel_bins(60), calc_vel_bins(75), 2000, max_velocity+1]
        else:
            bins = [-0.0001, 5, calc_vel_bins(15), calc_vel_bins(30), calc_vel_bins(45), calc_vel_bins(60), calc_vel_bins(75), 2000]
        bin_labels = range(len(bins)-1)  # Assigning a specific value to each bin
        df.loc[:, 'Velocity Binned'] = pd.cut(df['Corrected Velocity'].copy(), bins=bins, labels=bin_labels)
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(df['Pressure'], df['Velocity Binned'])
            plt.title('Hysteresis Plot')
            plt.xlabel('Pressure')
            plt.ylabel('Binned Corrected Velocity')
            plt.legend()
            plt.grid(True)
            plt.show()
        if verbose:
            print(f'the binned velocities are: {df["Corrected Velocity"]} to {df["Velocity Binned"]}')
        # Area calculation    
        if method == 'trapezoidal':
            area = trapezoid(df['Velocity Binned'], df['Pressure'])
        elif method == 'simpson' and len(df) % 2 == 0:
            # Simpson's rule requires an even number of samples, adding a check
            print("Warning: Simpson's rule requires an even number of intervals. Adjusting by removing the last data point.")
            area = simps(df['Velocity Binned'][:-1], df['Pressure'][:-1])
        elif method == 'simpson':
            area = simps(df['Velocity Binned'], df['Pressure'])
        else:
            raise ValueError("Method must be either 'trapezoidal' or 'simpson'")
        if verbose:
            print(f"Calculated normalized area under the curve using {method} rule: {area}")
    else:
        if plot:
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(df['Pressure'], df['Corrected Velocity'], label='Corrected Velocity vs. Pressure', marker='o')
            plt.fill_between(df['Pressure'], df['Corrected Velocity'], alpha=0.2)
            plt.title('Corrected Velocity vs. Pressure')
            plt.xlabel('Pressure')
            plt.ylabel('Corrected Velocity')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        # Area calculation
        if method == 'trapezoidal':
            area = trapezoid(df['Corrected Velocity'], df['Pressure'])
        elif method == 'simpson' and len(df) % 2 == 0:
            # Simpson's rule requires an even number of samples, adding a check
            print("Warning: Simpson's rule requires an even number of intervals. Adjusting by removing the last data point.")
            area = simps(df['Corrected Velocity'][:-1], df['Pressure'][:-1])
        elif method == 'simpson':
            area = simps(df['Corrected Velocity'], df['Pressure'])
        else:
            raise ValueError("Method must be either 'trapezoidal' or 'simpson'")
        if verbose:
            print(f"Calculated area under the curve using {method} rule: {area}")
    return area

def plot_densities(df):
    # Subset data into old vs young
    old_df = df[df['Age'] > 50]
    young_df = df[df['Age'] <= 50]
    # Plot density
    sns.kdeplot(df['Corrected Velocity'], label='Entire Dataset', fill=True)
    sns.kdeplot(old_df['Corrected Velocity'], label='old', fill=True, alpha=0.5)
    sns.kdeplot(young_df['Corrected Velocity'], label='young', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of Entire Dataset vs. Subset')
    plt.show()

    # Subset data into low BP vs high BP
    normBP_df = df[df['SYS_BP'] <= 120]
    highBP_df = df[df['SYS_BP'] > 120]
    print(f'the participants with high BP are: {highBP_df["Participant"].unique()}')

    # Plot density
    sns.kdeplot(df['Corrected Velocity'], label='Entire Dataset', fill=True)
    sns.kdeplot(highBP_df['Corrected Velocity'], label='high BP', fill=True, alpha=0.5)
    sns.kdeplot(normBP_df['Corrected Velocity'], label='normal', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of Entire Dataset vs. Subset')
    plt.show()

    # Plot density of old high BP vs young high BP vs old low BP vs young low BP
    old_highBP_df = old_df[old_df['SYS_BP'] > 120]
    young_highBP_df = young_df[young_df['SYS_BP'] > 120]
    old_normBP_df = old_df[old_df['SYS_BP'] <= 120]
    young_normBP_df = young_df[young_df['SYS_BP'] <= 120]

    # Plot density
    sns.kdeplot(df['Corrected Velocity'], label='Entire Dataset', fill=True)
    sns.kdeplot(old_highBP_df['Corrected Velocity'], label='old high BP', fill=True, alpha=0.5)
    sns.kdeplot(young_highBP_df['Corrected Velocity'], label='young high BP', fill=True, alpha=0.5)
    sns.kdeplot(old_normBP_df['Corrected Velocity'], label='old normal BP', fill=True, alpha=0.5)
    sns.kdeplot(young_normBP_df['Corrected Velocity'], label='young normal BP', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of Entire Dataset vs. Subset')
    plt.show()

    # compare high BP old vs young
    # Plot density
    sns.kdeplot(df['Corrected Velocity'], label='Entire Dataset', fill=True)
    sns.kdeplot(old_highBP_df['Corrected Velocity'], label='old high BP', fill=True, alpha=0.5)
    sns.kdeplot(young_highBP_df['Corrected Velocity'], label='young high BP', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of high BP participants')
    plt.show()

    # compare low BP old vs young
    # Plot density
    sns.kdeplot(df['Corrected Velocity'], label='Entire Dataset', fill=True)
    sns.kdeplot(old_normBP_df['Corrected Velocity'], label='old normal BP', fill=True, alpha=0.5)
    sns.kdeplot(young_normBP_df['Corrected Velocity'], label='young normal BP', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of normal BP participants')
    plt.show()
    return 0

def plot_densities_pressure(summary_df):
        # Subset data into old vs young
    old_df = summary_df[summary_df['Age'] > 50]
    young_df = summary_df[summary_df['Age'] <= 50]
    # Plot density
    sns.kdeplot(summary_df['Pressure'], label='Entire Dataset', fill=True)
    sns.kdeplot(old_df['Pressure'], label='old', fill=True, alpha=0.5)
    sns.kdeplot(young_df['Pressure'], label='young', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of Entire Dataset vs. Subset')
    plt.show()

    # Subset data into low BP vs high BP
    normBP_df = summary_df[summary_df['SYS_BP'] <= 120]
    highBP_df = summary_df[summary_df['SYS_BP'] > 120]
    print(f'the participants with high BP are: {highBP_df["Participant"].unique()}')

    # Plot density
    sns.kdeplot(summary_df['Pressure'], label='Entire Dataset', fill=True)
    sns.kdeplot(normBP_df['Pressure'], label='normal', fill=True, alpha=0.5)
    sns.kdeplot(highBP_df['Pressure'], label='high BP', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of Entire Dataset vs. Subset')
    plt.show()

    # Plot density of old high BP vs young high BP vs old low BP vs young low BP
    old_highBP_df = old_df[old_df['SYS_BP'] > 120]
    young_highBP_df = young_df[young_df['SYS_BP'] > 120]
    old_normBP_df = old_df[old_df['SYS_BP'] <= 120]
    young_normBP_df = young_df[young_df['SYS_BP'] <= 120]

    # Plot density
    sns.kdeplot(summary_df['Pressure'], label='Entire Dataset', fill=True)
    sns.kdeplot(old_highBP_df['Pressure'], label='old high BP', fill=True, alpha=0.5)
    sns.kdeplot(young_highBP_df['Pressure'], label='young high BP', fill=True, alpha=0.5)
    sns.kdeplot(old_normBP_df['Pressure'], label='old normal BP', fill=True, alpha=0.5)
    sns.kdeplot(young_normBP_df['Pressure'], label='young normal BP', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of Entire Dataset vs. Subset')
    plt.show()

    # compare high BP old vs young
    # Plot density
    sns.kdeplot(summary_df['Pressure'], label='Entire Dataset', fill=True)
    sns.kdeplot(old_highBP_df['Pressure'], label='old high BP', fill=True, alpha=0.5)
    sns.kdeplot(young_highBP_df['Pressure'], label='young high BP', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of high BP participants')
    plt.show()

    # compare low BP old vs young
    # Plot density
    sns.kdeplot(summary_df['Pressure'], label='Entire Dataset', fill=True)
    sns.kdeplot(old_normBP_df['Pressure'], label='old normal BP', fill=True, alpha=0.5)
    sns.kdeplot(young_normBP_df['Pressure'], label='young normal BP', fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of normal BP participants')
    plt.show()
    return 0


def plot_densities_individual(summary_df, participant_df, participant):
    # Plot density
    sns.kdeplot(summary_df['Corrected Velocity'], label='Entire Dataset', fill=True)
    sns.kdeplot(participant_df['Corrected Velocity'], label=participant, fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of Entire Dataset vs. Subset')
    plt.show()
    return 0

def plot_densities_pressure_individual(summary_df, participant_df, participant):
    # Plot density
    sns.kdeplot(summary_df['Pressure'], label='Entire Dataset', fill=True)
    sns.kdeplot(participant_df['Pressure'], label=participant, fill=True, alpha=0.5)
    plt.legend()
    plt.title('Density Plot of Entire Dataset vs. Subset')
    plt.show()
    return 0

def plot_hist_pressure(summary_df, normalize = False, density = False):
    # Subset data into old vs young
    old_df = summary_df[summary_df['Age'] > 50]
    young_df = summary_df[summary_df['Age'] <= 50]
    # normalize by the number of participants
    plt.hist(summary_df['Pressure'], bins=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,10], label='Entire Dataset', alpha = 0.5, density=density)
    plt.hist(old_df['Pressure'], bins=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,10], label='Old Participants', alpha = 0.5, density=density)
    plt.hist(young_df['Pressure'], bins=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,10], label='Young Participants', alpha = 0.5, density=density)
    plt.legend()
    plt.title('Pressure Histogram')
    plt.show()

    # Subset data into low BP vs high BP
    normBP_df = summary_df[summary_df['SYS_BP'] <= 120]
    highBP_df = summary_df[summary_df['SYS_BP'] > 120]
    plt.hist(summary_df['Pressure'], bins=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,10], label='Entire Dataset', alpha = 0.5, density=density)
    plt.hist(highBP_df['Pressure'], bins=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,10], label='High BP', alpha = 0.5, density=density)
    plt.hist(normBP_df['Pressure'], bins=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,10], label='Normal BP', alpha = 0.5, density=density)
    plt.legend()
    plt.title('Pressure Histogram')
    plt.show()
    return 0

def plot_hist_specific_pressure(df, pressure, density = False, hist = True):
    specific_velocities = df[df['Pressure'] == pressure]
    old_df = specific_velocities[specific_velocities['Age'] > 50]
    young_df = specific_velocities[specific_velocities['Age'] <= 50]
    if hist:
        plt.hist(specific_velocities['Corrected Velocity'], bins=20, label=f'Pressure: {pressure}', alpha = 0.5, density=density)
        plt.hist(old_df['Corrected Velocity'], bins=20, label=f'Old Participants', alpha = 0.5, density=density)
        plt.hist(young_df['Corrected Velocity'], bins=20, label=f'Young Participants', alpha = 0.5, density=density)
    else:
        sns.kdeplot(specific_velocities['Corrected Velocity'], label=f'Pressure: {pressure}', fill=True)
        sns.kdeplot(old_df['Corrected Velocity'], label=f'Old Participants', fill=True)
        sns.kdeplot(young_df['Corrected Velocity'], label=f'Young Participants', fill=True)
    plt.legend()
    plt.title(f'Velocity Histogram at Pressure: {pressure}')
    plt.show()

    # Subset data into low BP vs high BP
    normBP_df = specific_velocities[specific_velocities['SYS_BP'] <= 120]
    highBP_df = specific_velocities[specific_velocities['SYS_BP'] > 120]
    plt.hist(specific_velocities['Corrected Velocity'], bins=20, label=f'Pressure: {pressure}', alpha = 0.5, density=density)
    plt.hist(normBP_df['Corrected Velocity'], bins=20, label=f'Normal BP', alpha = 0.5, density=density)
    plt.hist(highBP_df['Corrected Velocity'], bins=20, label=f'High BP', alpha = 0.5, density=density)
    plt.legend()
    plt.title(f'Velocity Histogram at Pressure: {pressure}')
    plt.show()
    return 0

def plot_hist_comp_pressure(df, normalize = False, density = False, hist = True, fill = False):
    low_p = df[df['Pressure'] == 0.2]
    med_p = df[df['Pressure'] == 0.8]
    high_p = df[df['Pressure'] == 1.2]

    # Plot density
    if hist:
        plt.hist(low_p['Corrected Velocity'], bins=20, label='0.2 psi', alpha = 0.5, density=density)
        plt.hist(med_p['Corrected Velocity'], bins=20, label='0.8 psi', alpha = 0.5, density=density)
        plt.hist(high_p['Corrected Velocity'], bins=20, label='1.2 psi', alpha = 0.5, density=density)
    else:
        sns.kdeplot(low_p['Corrected Velocity'], label='0.2 psi', fill=True)
        sns.kdeplot(med_p['Corrected Velocity'], label='0.8 psi', fill=True)
        sns.kdeplot(high_p['Corrected Velocity'], label='1.2 psi', fill=True)
    plt.legend()
    plt.title('Density Plot of different pressures')
    plt.show()

    # make subset of old and young participants
    old_low_p = low_p[low_p['Age'] > 50]
    young_low_p = low_p[low_p['Age'] <= 50]
    old_med_p = med_p[med_p['Age'] > 50]
    young_med_p = med_p[med_p['Age'] <= 50]
    old_high_p = high_p[high_p['Age'] > 50]
    young_high_p = high_p[high_p['Age'] <= 50]

    # # Plot density old first
    # if hist:
    #     plt.hist(low_p['Corrected Velocity'], bins=20, label='0.2 psi', alpha = 0.5, density=density, color = 'C0', histtype='step')
    #     plt.hist(old_low_p['Corrected Velocity'], bins=20, label='old 0.2 psi', alpha = 0.5, density=density, color = 'C1', histtype='step')
    #     plt.hist(low_p['Corrected Velocity'], bins=20, label='0.8 psi', alpha = 0.5, density=density, color = 'C2')
    #     plt.hist(old_med_p['Corrected Velocity'], bins=20, label='old 0.8 psi', alpha = 0.5, density=density, color = 'C3')
    #     plt.hist(low_p['Corrected Velocity'], bins=20, label='1.2 psi', alpha = 0.5, density=density, color = 'C4', histtype='step')
    #     plt.hist(old_high_p['Corrected Velocity'], bins=20, label='old 1.2 psi', alpha = 0.5, density=density, color = 'C5', histtype='step')
    #     # plt.hist(young_low_p['Corrected Velocity'], bins=20, label='young 0.2 psi', alpha = 0.5, density=density)
    # else:
    #     sns.kdeplot(low_p['Corrected Velocity'], label='0.2 psi', fill=True, color='C0')
    #     sns.kdeplot(old_low_p['Corrected Velocity'], label='old 0.2 psi', fill=True, color='C1')
    #     sns.kdeplot(med_p['Corrected Velocity'], label='0.8 psi', fill=True, color='C2')
    #     sns.kdeplot(old_med_p['Corrected Velocity'], label='old 0.8 psi', fill=True, color='C3')
    #     sns.kdeplot(high_p['Corrected Velocity'], label='1.2 psi', fill=True, color='C4')
    #     sns.kdeplot(old_high_p['Corrected Velocity'], label='old 1.2 psi', fill=True, color='C5')
    #     # sns.kdeplot(young_low_p['Corrected Velocity'], label='young 0.2 psi', fill=True)
    # plt.legend()
    # plt.title('Density Plot of different pressures young and old')
    # plt.show()

    # # Plot density old vs young
    # if hist:
    #     plt.hist(low_p['Corrected Velocity'], bins=20, label='0.2 psi', alpha = 0.5, density=density)
    #     plt.hist(old_low_p['Corrected Velocity'], bins=20, label='old 0.2 psi', alpha = 0.5, density=density)
    #     plt.hist(young_low_p['Corrected Velocity'], bins=20, label='young 0.2 psi', alpha = 0.5, density=density)
    #     plt.hist(med_p['Corrected Velocity'], bins=20, label='0.8 psi', alpha = 0.5, density=density)
    #     plt.hist(old_med_p['Corrected Velocity'], bins=20, label='old 0.8 psi', alpha = 0.5, density=density)
    #     plt.hist(young_med_p['Corrected Velocity'], bins=20, label='young 0.8 psi', alpha = 0.5, density=density)
    #     plt.hist(high_p['Corrected Velocity'], bins=20, label='1.2 psi', alpha = 0.5, density=density)
    #     plt.hist(old_high_p['Corrected Velocity'], bins=20, label='old 1.2 psi', alpha = 0.5, density=density)
    #     plt.hist(young_high_p['Corrected Velocity'], bins=20, label='young 1.2 psi', alpha = 0.5, density=density)
    # else:
    #     sns.kdeplot(low_p['Corrected Velocity'], label='0.2 psi', fill=True)
    #     sns.kdeplot(old_low_p['Corrected Velocity'], label='old 0.2 psi', fill=True)
    #     sns.kdeplot(young_low_p['Corrected Velocity'], label='young 0.2 psi', fill=True)
    #     sns.kdeplot(med_p['Corrected Velocity'], label='0.8 psi', fill=True)
    #     sns.kdeplot(old_med_p['Corrected Velocity'], label='old 0.8 psi', fill=True)
    #     sns.kdeplot(young_med_p['Corrected Velocity'], label='young 0.8 psi', fill=True)
    #     sns.kdeplot(high_p['Corrected Velocity'], label='1.2 psi', fill=True)
    #     sns.kdeplot(old_high_p['Corrected Velocity'], label='old 1.2 psi', fill=True)
    #     sns.kdeplot(young_high_p['Corrected Velocity'], label='young 1.2 psi', fill=True)
    # plt.legend()
    # plt.title('Density Plot of different pressures young and old')
    # plt.show()

    # Plot 0.2 psi old vs young
    if hist:
        plt.hist(low_p['Corrected Velocity'], bins=20, label='0.2 psi', alpha = 0.5, density=density)
        plt.hist(old_low_p['Corrected Velocity'], bins=20, label='old 0.2 psi', alpha = 0.5, density=density)
        plt.hist(young_low_p['Corrected Velocity'], bins=20, label='young 0.2 psi', alpha = 0.5, density=density)
    else:
        sns.kdeplot(low_p['Corrected Velocity'], label='0.2 psi', fill=True)
        sns.kdeplot(old_low_p['Corrected Velocity'], label='old 0.2 psi', fill=True)
        sns.kdeplot(young_low_p['Corrected Velocity'], label='young 0.2 psi', fill=True)
    plt.legend()
    plt.title('Density Plot of 0.2 psi young and old')
    plt.show()

    # Plot 0.8 psi old vs young
    if hist:
        plt.hist(med_p['Corrected Velocity'], bins=20, label='0.8 psi', alpha = 0.5, density=density)
        plt.hist(old_med_p['Corrected Velocity'], bins=20, label='old 0.8 psi', alpha = 0.5, density=density)
        plt.hist(young_med_p['Corrected Velocity'], bins=20, label='young 0.8 psi', alpha = 0.5, density=density)
    else:
        sns.kdeplot(med_p['Corrected Velocity'], label='0.8 psi', fill=True)
        sns.kdeplot(old_med_p['Corrected Velocity'], label='old 0.8 psi', fill=True)
        sns.kdeplot(young_med_p['Corrected Velocity'], label='young 0.8 psi', fill=True)
    plt.legend()
    plt.title('Density Plot of 0.8 psi young and old')
    plt.show()

    # Plot 1.2 psi old vs young
    if hist:
        plt.hist(high_p['Corrected Velocity'], bins=20, label='1.2 psi', alpha = 0.5, density=density)
        plt.hist(old_high_p['Corrected Velocity'], bins=20, label='old 1.2 psi', alpha = 0.5, density=density)
        plt.hist(young_high_p['Corrected Velocity'], bins=20, label='young 1.2 psi', alpha = 0.5, density=density)
    else:
        sns.kdeplot(high_p['Corrected Velocity'], label='1.2 psi', fill=True)
        sns.kdeplot(old_high_p['Corrected Velocity'], label='old 1.2 psi', fill=True)
        sns.kdeplot(young_high_p['Corrected Velocity'], label='young 1.2 psi', fill=True)
    plt.legend()
    plt.title('Density Plot of 1.2 psi young and old')
    plt.show()



    return 0

def plot_cdf_comp_pressure(df, title = 'CDF Plot for Different Pressures', write = False):
    """
    Plots the CDF for corrected velocities across different pressures and age groups within each pressure category.

    Parameters:
    - df: DataFrame containing 'Pressure', 'Corrected Velocity', and 'Age' columns.
    """
    pressures = df['Pressure'].unique()
    colors = ['C0', 'C1', 'C2']  # Colors for the plots
    labels = ['0.2 psi', '0.8 psi', '1.2 psi']  # Labels for different pressures

    # Plot CDF for different pressures
    plt.figure(figsize=(12, 8))
    for pressure, color, label in zip(pressures, colors, labels):
        subset = df[df['Pressure'] == pressure]
        values = np.sort(subset['Corrected Velocity'])
        cdf = np.arange(1, len(values) + 1) / len(values)
        plt.plot(values, cdf, label=f'{label}', color=color)

    plt.title(title)
    plt.xlabel('Velocity (um/s)')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    if write:
        filename = title.replace(' ', '_')
        filename += '.png'
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()

    # Plot CDF for age groups within each pressure category
    plt.figure(figsize=(12, 8))
    for pressure, color, label in zip(pressures, colors, labels):
        for age_group, linestyle in [('old', 'dashed'), ('young', 'solid')]:
            subset = df[(df['Pressure'] == pressure) & (df['Age'] > 50 if age_group == 'old' else df['Age'] <= 50)]
            values = np.sort(subset['Corrected Velocity'])
            cdf = np.arange(1, len(values) + 1) / len(values)
            plt.plot(values, cdf, label=f'{age_group} {label}', color=color, linestyle=linestyle)

    plt.title(title + ' by Age Group')
    plt.xlabel('Velocity (um/s)')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    if write:
        filename = title.replace(' ', '_') + '_age_group'
        filename += '.png'
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()
    return 0


def quantile_analysis(data, subset, quantiles=[0.25, 0.5, 0.75]):
    data_quantiles = np.quantile(data, quantiles)
    subset_quantiles = np.quantile(subset, quantiles)
    
    print("Quantile Analysis:")
    for q, dq, sq in zip(quantiles, data_quantiles, subset_quantiles):
        print(f"{int(q*100)}th percentile - Entire Dataset: {dq}, Subset: {sq}")
    return 0

def plot_cdf(data, subsets, labels=['Entire Dataset', 'Subset'], title = 'CDF Comparison', write = False):
    """
    Plots the CDF of the entire dataset and the inputtedc subsets.

    Args:
        data (array-like): The entire dataset
        subsets (list of array-like): The subsets to be compared
        labels (list of str): The labels for the entire dataset and the subsets
        title (str): The title of the plot
        write (bool): Whether to write the plot to a file
    
    Returns:
        0 if successful
    """
    # Calculate CDF for the entire dataset
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    
    # value is the column of the dataframe that we are interested in
    value = 'Velocity (um/s)' if 'Pressure' not in title else 'Pressure (psi)'
    
    if len(subsets) == 1:
        # Calculate CDF for the subset
        subset_sorted = np.sort(subsets[0])
        p_subset = 1. * np.arange(len(subsets[0])) / (len(subsets[0]) - 1)
        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(data_sorted, p, label=labels[0])
        plt.plot(subset_sorted, p_subset, label=labels[1], linestyle='--')
        plt.ylabel('CDF')
        plt.xlabel(value)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if write:
            filename = title.replace(' ', '_')
            filename += '.png'
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()
    elif len(subsets) == 0:
        return 1
    else:
        plt.figure(figsize=(8, 5))
        plt.plot(data_sorted, p, label=labels[0])
        labels = labels[1:]
        for subset, label in zip(subsets, labels):
            # Calculate CDF for the subset
            subset_sorted = np.sort(subset)
            p_subset = 1. * np.arange(len(subset)) / (len(subset) - 1)
            # Plotting
            plt.plot(subset_sorted, p_subset, label=label, linestyle='--')
        plt.ylabel('CDF')
        plt.xlabel(value)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if write:
            filename = title.replace(' ', '_')
            filename += '.png'
            filename = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\results', filename)
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()        
    return 0

def plot_boxplot(data, subset, labels=['Entire Dataset', 'Subset']):
    # Combine data and subset into a single dataset for plotting
    combined_data = np.concatenate([data, subset])
    # Create a list of labels corresponding to the data and subset
    combined_labels = np.concatenate([[labels[0]]*len(data), [labels[1]]*len(subset)])
    
    # Create a DataFrame for easier plotting with seaborn
    import pandas as pd
    df = pd.DataFrame({'Value': combined_data, 'Group': combined_labels})
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Group', y='Value', data=df)
    plt.title('Box Plot of Entire Dataset vs. Subset')
    plt.show()
    return 0

def plot_violinplot(data, subset, labels=['Entire Dataset', 'Subset']):
    # Combine data and subset into a single dataset for plotting
    combined_data = np.concatenate([data, subset])
    # Create a list of labels corresponding to the data and subset
    combined_labels = np.concatenate([[labels[0]]*len(data), [labels[1]]*len(subset)])
    
    # Create a DataFrame for easier plotting with seaborn
    import pandas as pd
    df = pd.DataFrame({'Value': combined_data, 'Group': combined_labels})
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Group', y='Value', data=df, inner='quartile')
    plt.title('Violin Plot of Entire Dataset vs. Subset')
    plt.show()
    return 0

def collapse_df(df):
    # Calculate median velocity for specific pressures and overall median velocity
    pressure_medians = df.groupby(['Participant', 'Pressure'])['Corrected Velocity'].median().unstack()
    pressure_medians.columns = [f'Pressure {col}' for col in pressure_medians.columns]

    # Overall median velocity and median SYS_BP
    overall_median_velocity = df.groupby('Participant')['Corrected Velocity'].median().rename('Median Velocity')
    median_sys_bp = df.groupby('Participant')['SYS_BP'].median().rename('Median SYS_BP')

    # Combine all the data
    final_df = pd.concat([pressure_medians, overall_median_velocity, median_sys_bp], axis=1).reset_index()

    # Assume the Age for each participant is constant and just take the first one.
    ages = df.groupby('Participant')['Age'].first()

    # Combine age into the final DataFrame
    final_df = pd.merge(final_df, ages, left_on='Participant', right_index=True)
    
    # If there is an 'Area Score' column, add it to the final DataFrame
    if 'Area Score' in df.columns:
        area_scores = df.groupby('Participant')['Area Score'].mean().rename('Area Score')
        final_df = pd.merge(final_df, area_scores, left_on='Participant', right_index=True)
    
    # If there is a 'Log Area Score' column, add it to the final DataFrame
    if 'Log Area Score' in df.columns:
        log_area_scores = df.groupby('Participant')['Log Area Score'].mean().rename('Log Area Score')
        final_df = pd.merge(final_df, log_area_scores, left_on='Participant', right_index=True)
    
    return final_df

def plot_stats(df):
    # Setting the style
    sns.set_theme(style="whitegrid")

    # Plotting distributions of Median Velocity, Age, and Median SYS_BP
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(df['Median Velocity'], kde=True, ax=ax[0], color='skyblue')
    ax[0].set_title('Distribution of Median Velocity')

    sns.histplot(df['Age'], kde=True, ax=ax[1], color='lightgreen')
    ax[1].set_title('Distribution of Age')

    sns.histplot(df['Median SYS_BP'], kde=True, ax=ax[2], color='salmon')
    ax[2].set_title('Distribution of Median SYS_BP')

    plt.tight_layout()
    plt.show()

    # Pair plot to visualize relationships between Median Velocity and other features
    sns.pairplot(df[['Median Velocity', 'Pressure 0.2', 'Pressure 0.8', 'Pressure 1.2', 'Age', 'Median SYS_BP']])
    plt.show()

    # Correlation matrix
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    return 0

def make_models(df, y = 'Median Velocity', plot = False):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import RandomizedSearchCV

    # Preparing the data
    X = df.drop(['Participant', 'Median Velocity', 'Median SYS_BP'], axis=1)  # Using pressures and age as features
    if y == 'Median Velocity':
        y = df['Median Velocity']
    else:
        y = df['Log Area Score']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initializing models
    linear_reg = LinearRegression()
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # # Optimized Random Forest caused issues with residuals
    # random_forest_optimized = RandomForestRegressor(n_estimators=200, min_samples_split=2, min_samples_leaf=1, random_state=42)
    # random_forest = random_forest_optimized

    # Training Linear Regression
    linear_reg.fit(X_train, y_train)
    y_pred_lr = linear_reg.predict(X_test)

    # Training Random Forest Regressor
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)

    # Evaluating models
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

    print(mae_lr, rmse_lr, mae_rf, rmse_rf)

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Actual vs. Predicted for Linear Regression
        axs[0, 0].scatter(y_test, y_pred_lr, color='blue', alpha=0.5)
        axs[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        axs[0, 0].set_title('Linear Regression: Actual vs. Predicted')
        axs[0, 0].set_xlabel('Actual')
        axs[0, 0].set_ylabel('Predicted')

        # Residuals for Linear Regression
        residuals_lr = y_test - y_pred_lr
        axs[1, 0].scatter(y_pred_lr, residuals_lr, color='red', alpha=0.5)
        axs[1, 0].hlines(y=0, xmin=y_pred_lr.min(), xmax=y_pred_lr.max(), colors='k', lw=2)
        axs[1, 0].set_title('Linear Regression: Residuals')
        axs[1, 0].set_xlabel('Predicted')
        axs[1, 0].set_ylabel('Residuals')

        # Actual vs. Predicted for Random Forest
        axs[0, 1].scatter(y_test, y_pred_rf, color='green', alpha=0.5)
        axs[0, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        axs[0, 1].set_title('Random Forest: Actual vs. Predicted')
        axs[0, 1].set_xlabel('Actual')
        axs[0, 1].set_ylabel('Predicted')

        # Residuals for Random Forest
        residuals_rf = y_test - y_pred_rf
        axs[1, 1].scatter(y_pred_rf, residuals_rf, color='orange', alpha=0.5)
        axs[1, 1].hlines(y=0, xmin=y_pred_rf.min(), xmax=y_pred_rf.max(), colors='k', lw=2)
        axs[1, 1].set_title('Random Forest: Residuals')
        axs[1, 1].set_xlabel('Predicted')
        axs[1, 1].set_ylabel('Residuals')

        plt.tight_layout()
        plt.show()

    # # Randomized Search CV for hyperparameter tuning
    # # Simplifying the parameter grid and reducing the number of iterations for demonstration
    # simplified_param_dist = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [None, 10],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2]
    # }

    # # Initializing the Randomized Search CV object with simplified parameters
    # simplified_random_search = RandomizedSearchCV(estimator=random_forest, param_distributions=simplified_param_dist, n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=-1, scoring='neg_mean_absolute_error')

    # # Fitting it to the data
    # simplified_random_search.fit(X_train, y_train)

    # # Best parameters and score from the simplified search
    # simplified_best_params = simplified_random_search.best_params_
    # simplified_best_score = -simplified_random_search.best_score_  # Note: scores are negative in sklearn

    # print(simplified_best_params, simplified_best_score)
    return mae_lr, rmse_lr, mae_rf, rmse_rf

def make_log_df(df, plot = False):
    # Log-transforming selected features and the target variable
    df['log_Pressure_0.2'] = np.log1p(df['Pressure 0.2'])
    df['log_Pressure_0.4'] = np.log1p(df['Pressure 0.4'])
    df['log_Pressure_0.6'] = np.log1p(df['Pressure 0.6'])
    df['log_Pressure_0.8'] = np.log1p(df['Pressure 0.8'])
    df['log_Pressure_1.0'] = np.log1p(df['Pressure 1.0'])
    df['log_Pressure_1.2'] = np.log1p(df['Pressure 1.2'])
    df['log_Median_Velocity'] = np.log1p(df['Median Velocity'])

    if plot:
        # Plot log-transformed features vs Age
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        sns.scatterplot(y='log_Pressure_0.2', x='Age', data=df, ax=axs[0])
        axs[0].set_title('Log-transformed Pressure 0.2 vs. Age')
        sns.scatterplot(y='log_Pressure_0.8', x='Age', data=df, ax=axs[1])
        axs[1].set_title('Log-transformed Pressure 0.8 vs. Age')
        sns.scatterplot(y='log_Pressure_1.2', x='Age', data=df, ax=axs[2])
        axs[2].set_title('Log-transformed Pressure 1.2 vs. Age')
        plt.tight_layout()
        plt.show()

        # Plot log-transformed features vs SYS_BP
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        sns.scatterplot(y='log_Pressure_0.2', x='Median SYS_BP', data=df, ax=axs[0])
        axs[0].set_title('Log-transformed Pressure 0.2 vs. Median SYS_BP')
        sns.scatterplot(y='log_Pressure_0.8', x='Median SYS_BP', data=df, ax=axs[1])
        axs[1].set_title('Log-transformed Pressure 0.8 vs. Median SYS_BP')
        sns.scatterplot(y='log_Pressure_1.2', x='Median SYS_BP', data=df, ax=axs[2])
        axs[2].set_title('Log-transformed Pressure 1.2 vs. Median SYS_BP')
        plt.tight_layout()
        plt.show()

        # Plot log-transformed Median Velocity vs Age
        plt.figure(figsize=(8, 5))
        sns.scatterplot(y='log_Median_Velocity', x='Age', data=df)
        plt.title('Log-transformed Median Velocity vs. Age')
        plt.show()
    return df


def compare_log_and_linear(df, plot = False):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor

    # Assuming `df` is your DataFrame
    df = make_log_df(df)

    # Splitting both datasets into training and testing sets for all pressure columns
    features_original = ['Pressure 0.2', 'Pressure 0.8', 'Pressure 1.2']  # Example original features
    features_transformed = ['log_Pressure_0.2', 'log_Pressure_0.8', 'log_Pressure_1.2']  # Corresponding log-transformed features
    
    X_original = df[features_original]
    y_original = df['Median Velocity']
    
    X_transformed = df[features_transformed]
    y_transformed = df['log_Median_Velocity']
    
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_original, y_original, test_size=0.2, random_state=42)
    X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(X_transformed, y_transformed, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        # Original data
        model.fit(X_train_orig, y_train_orig)
        predictions_orig = model.predict(X_test_orig)
        
        # Log-transformed data
        model.fit(X_train_trans, y_train_trans)
        predictions_trans = model.predict(X_test_trans)
        
        # Evaluation
        mse_original = mean_squared_error(y_test_orig, predictions_orig)
        r2_original = r2_score(y_test_orig, predictions_orig)
        
        mse_transformed = mean_squared_error(np.expm1(y_test_trans), np.expm1(predictions_trans))
        r2_transformed = r2_score(np.expm1(y_test_trans), np.expm1(predictions_trans))
        
        print(f"{name} - Original Data - MSE: {mse_original:.4f}, R2: {r2_original:.4f}")
        print(f"{name} - Log-transformed Data - MSE: {mse_transformed:.4f}, R2: {r2_transformed:.4f}")
        
        if plot:
            # Actual vs. Predicted Plot
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            sns.scatterplot(x=y_test_orig, y=predictions_orig).set(title=f'{name} - Original Data: Actual vs. Predicted')
            plt.subplot(1, 2, 2)
            sns.scatterplot(x=np.expm1(y_test_trans), y=np.expm1(predictions_trans)).set(title=f'{name} - Log-transformed Data: Actual vs. Predicted')
            plt.show()
            
            # Residual Plot
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            sns.residplot(x=predictions_orig, y=y_test_orig, lowess=True).set(title=f'{name} - Original Data: Residuals')
            plt.subplot(1, 2, 2)
            sns.residplot(x=np.expm1(predictions_trans), y=np.expm1(y_test_trans), lowess=True).set(title=f'{name} - Log-transformed Data: Residuals')
            plt.show()
    
    return 0

def run_regression(df):
    """
    Runs a linear regression analysis on the inputted DataFrame and plots results.

    Args:
        df (DataFrame): The DataFrame to be analyzed
    
    Returns:
        0 if successful
    """
    collapsed_df = collapse_df(df)
    collapsed_df = make_log_df(collapsed_df)
    plot_stats(collapsed_df)
    # make_models(collapsed_df, plot=True)
    # compare_log_and_linear(collapsed_df, plot=True)


    # # Corrected aggregation method for 'SYS_BP'
    # collapsed_df = df.groupby('Participant').agg({'Corrected Velocity': 'median', 'Age': 'first', 'SYS_BP': 'median'}).reset_index()

    # # Create the interaction term
    # collapsed_df['Age*SYS_BP'] = collapsed_df['Age'] * collapsed_df['SYS_BP']

    # # Prepare X and Y for the model
    # X = collapsed_df[['Age', 'SYS_BP', 'Age*SYS_BP']]
    # X = sm.add_constant(X)  # adding a constant
    # Y = collapsed_df['Corrected Velocity']

    # # Fit the model
    # model = sm.OLS(Y, X).fit()

    # # Print the regression results
    # print(model.summary())

    # # Prepare the plots
    # plt.figure(figsize=(10, 5))

    # # Plot for Age vs Corrected Velocity
    # plt.subplot(1, 2, 1)  # First subplot
    # plt.scatter(collapsed_df['Age'], Y, label='Actual Data')
    # ages = np.linspace(collapsed_df['Age'].min(), collapsed_df['Age'].max(), 100)
    # # Predicting with only Age variable (ignoring SYS_BP and interaction for plotting simplicity)
    # predicted_values_age = model.predict(sm.add_constant(np.column_stack((ages, np.zeros_like(ages), np.zeros_like(ages)))))
    # plt.plot(ages, predicted_values_age, color='red', label='Fitted Line')
    # plt.xlabel('Age')
    # plt.ylabel('Corrected Velocity')
    # plt.title('Age vs Corrected Velocity')
    # plt.legend()

    # # Plot for SYS_BP vs Corrected Velocity
    # plt.subplot(1, 2, 2)  # Second subplot
    # plt.scatter(collapsed_df['SYS_BP'], Y, label='Actual Data')
    # sys_bps = np.linspace(collapsed_df['SYS_BP'].min(), collapsed_df['SYS_BP'].max(), 100)
    # # Predicting with only SYS_BP variable (ignoring Age and interaction for plotting simplicity)
    # predicted_values_sys_bp = model.predict(sm.add_constant(np.column_stack((np.zeros_like(sys_bps), sys_bps, np.zeros_like(sys_bps)))))
    # plt.plot(sys_bps, predicted_values_sys_bp, color='red', label='Fitted Line')
    # plt.xlabel('SYS_BP')
    # plt.ylabel('Corrected Velocity')
    # plt.title('SYS_BP vs Corrected Velocity')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    

    return 0

# def calculate_cdf_area(data, start=10, end=700):
#     # Generate a linear space from the min to max velocity in the range of interest
#     x = np.linspace(start, end, num=500)
    
#     # Calculate the CDF using the empirical data
#     cdf = np.interp(x, np.sort(data), np.linspace(0, 1, len(data)))
    
#     # Calculate the area under the CDF curve using Simpson's rule
#     area = simps(cdf, x)
#     return area

# def calculate_distance_score(data):
#     # Calculate the CDF area for the entire dataset
#     entire_dataset_area = calculate_cdf_area(data)

#     # Calculate the CDF area for each participant and compute the distance score
#     participant_scores = data.groupby('Participant')['Corrected Velocity'].apply(lambda x: entire_dataset_area - calculate_cdf_area(x))

#     return participant_scores



def calculate_cdf_area(data, start=10, end=700):
    data = data['Corrected Velocity']
    log_data = np.log1p(data)
    data_sorted = np.sort(data)
    log_data_sorted = np.sort(log_data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    p_log = 1. * np.arange(len(log_data)) / (len(log_data) - 1)
    
    # Interpolate to find CDF values at start and end points if necessary
    start_cdf = np.interp(start, data_sorted, p)
    start_cdf_log = np.interp(np.log1p(start), log_data_sorted, p_log)
    end_cdf = np.interp(end, data_sorted, p)
    end_cdf_log = np.interp(np.log1p(end), log_data_sorted, p_log)
    
    # Calculate the area under the CDF curve using Simpson's rule within the range
    x = np.linspace(start, end, num=500)  # More points for a smoother curve and more accurate integration
    x_log = np.linspace(np.log1p(start), np.log1p(end), num=500)
    cdf_values = np.interp(x, data_sorted, p)
    cdf_values_log = np.interp(x_log, log_data_sorted, p_log)
    area = simps(cdf_values, x)
    area_log = simps(cdf_values_log, x_log)
    
    return area, area_log


def main(verbose = False):
    if platform.system() == 'Windows':
        if 'gt8mar' in os.getcwd():
            path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\summary_df_test.csv'
        else:
            path = 'C:\\Users\\gt8ma\\capillary-flow\\results\\summary_df_test.csv'
    else:
        path = '/hpc/projects/capillary-flow/results/summary_df_test.csv'

    summary_df = pd.read_csv(path)
    summary_df = summary_df.drop(columns=['Capillary'])
    summary_df = summary_df.rename(columns={'Capillary_new': 'Capillary'})
    
    
    old_subset = summary_df[summary_df['Age'] > 50]
    # drop nan values
    old_subset_no_nan = old_subset.dropna(subset=['Corrected Velocity'])
    young_subset = summary_df[summary_df['Age'] <= 50]
    # drop nan values
    young_subset_no_nan = young_subset.dropna(subset=['Corrected Velocity'])

    # plot_histograms(summary_df, 'Age')
    # plot_histograms(summary_df, 'SYS_BP')
    stat, p = mannwhitneyu(old_subset_no_nan['Corrected Velocity'], young_subset_no_nan['Corrected Velocity'], alternative='two-sided')      # could also use 'less' or 'greater'
    print('Statistics=%.3f, p=%.5f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    
    # # print(summary_df.head())

    # plot_densities(summary_df)
    # plot_densities_pressure(summary_df)
    # plot_hist_pressure(summary_df, density=True)
    
    # create a subset of summary_df with no pressure values greater than 1.2
    summary_df_no_high_pressure = summary_df[summary_df['Pressure'] <= 1.2]
    old_nhp = summary_df_no_high_pressure[summary_df_no_high_pressure['Age'] > 50]
    young_nhp = summary_df_no_high_pressure[summary_df_no_high_pressure['Age'] <= 50]
    # plot_hist_pressure(summary_df_no_high_pressure, density=True)

    # plot_hist_specific_pressure(summary_df, 0.2, density=True, hist=False)
    # plot_hist_specific_pressure(summary_df, 0.8, density=True, hist=False)
    # plot_hist_specific_pressure(summary_df, 1.2, density=True, hist=False) 

    # plot_hist_comp_pressure(summary_df, density=True, hist=False)
    # plot_densities(summary_df_no_high_pressure)
    # plot_cdf(summary_df_no_high_pressure['Corrected Velocity'], subsets= [summary_df_no_high_pressure[summary_df_no_high_pressure['Age'] > 50]['Corrected Velocity'], summary_df_no_high_pressure[summary_df_no_high_pressure['Age'] <= 50]['Corrected Velocity']], labels=['Entire Dataset', 'Old', 'Young'], title = 'CDF Comparison of velocities by Age')    
    # plot_cdf(summary_df['Corrected Velocity'], subsets= [summary_df[summary_df['SYS_BP'] > 120]['Corrected Velocity'], summary_df[summary_df['SYS_BP'] <= 120]['Corrected Velocity']], labels=['Entire Dataset', 'High BP', 'Normal BP'], title = 'CDF Comparison by BP')
    # plot_cdf(summary_df_no_high_pressure['Corrected Velocity'], subsets= [summary_df_no_high_pressure[summary_df_no_high_pressure['SYS_BP'] > 120]['Corrected Velocity'], summary_df_no_high_pressure[summary_df_no_high_pressure['SYS_BP'] <= 120]['Corrected Velocity']], labels=['Entire Dataset', 'High BP', 'Normal BP'], title = 'CDF Comparison of velocities by BP nhp')
    # plot_cdf(summary_df['Corrected Velocity'], subsets= [summary_df[summary_df['Age'] > 50]['Corrected Velocity'], summary_df[summary_df['Age'] <= 50]['Corrected Velocity']], labels=['Entire Dataset', 'Old', 'Young'], title = 'CDF Comparison by Age')
    
    # plot cdf for high bp old, high bp young, low bp old, low bp young
    highBP_old = summary_df[(summary_df['SYS_BP'] > 120) & (summary_df['Age'] > 50)]['Corrected Velocity']
    highBP_young = summary_df[(summary_df['SYS_BP'] > 120) & (summary_df['Age'] <= 50)]['Corrected Velocity']
    normBP_old = summary_df[(summary_df['SYS_BP'] <= 120) & (summary_df['Age'] > 50)]['Corrected Velocity']
    normBP_young = summary_df[(summary_df['SYS_BP'] <= 120) & (summary_df['Age'] <= 50)]['Corrected Velocity']

    highBP_old_nhp = summary_df_no_high_pressure[(summary_df_no_high_pressure['SYS_BP'] > 120) & (summary_df_no_high_pressure['Age'] > 50)]['Corrected Velocity']
    highBP_young_nhp = summary_df_no_high_pressure[(summary_df_no_high_pressure['SYS_BP'] > 120) & (summary_df_no_high_pressure['Age'] <= 50)]['Corrected Velocity']
    normBP_old_nhp = summary_df_no_high_pressure[(summary_df_no_high_pressure['SYS_BP'] <= 120) & (summary_df_no_high_pressure['Age'] > 50)]['Corrected Velocity']
    normBP_young_nhp = summary_df_no_high_pressure[(summary_df_no_high_pressure['SYS_BP'] <= 120) & (summary_df_no_high_pressure['Age'] <= 50)]['Corrected Velocity']

    # plot_cdf(summary_df['Corrected Velocity'], subsets= [highBP_old, highBP_young, normBP_old, normBP_young], labels=['Entire Dataset', 'High BP Old', 'High BP Young', 'Normal BP Old', 'Normal BP Young'], title = 'CDF Comparison of velocities by Age and BP')
    # plot_cdf(summary_df_no_high_pressure['Corrected Velocity'], subsets= [highBP_old_nhp, highBP_young_nhp, normBP_old_nhp, normBP_young_nhp], labels=['Entire Dataset', 'High BP Old', 'High BP Young', 'Normal BP Old', 'Normal BP Young'], title = 'CDF Comparison of velocities by Age and BP')
    # plot_cdf_comp_pressure(summary_df)
    area, area_log = calculate_cdf_area(summary_df_no_high_pressure)
    print(area)
    area_scores = []
    for participant in summary_df_no_high_pressure['Participant'].unique():
        participant_df = summary_df_no_high_pressure[summary_df_no_high_pressure['Participant'] == participant]
        participant_area, participant_area_log = calculate_cdf_area(participant_df)
        print(f'Participant {participant} has a CDF area of {participant_area:.2f} and a log CDF area of {participant_area_log:.2f}')
        area_scores.append([participant, participant_area-area, participant_area_log-area_log])
    # plot area scores
    area_scores_df = pd.DataFrame(area_scores, columns=['Participant', 'Area Score', 'Log Area Score'])
    area_scores_df = area_scores_df.sort_values(by='Area Score', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(area_scores_df['Participant'], area_scores_df['Log Area Score'], width=0.5)
    plt.xlabel('Participant')
    plt.ylabel('Area Score')
    plt.title('Area Score for Each Participant')
    plt.xticks(rotation=45)
    plt.show()

    # add area scores to summary_df_no_high_pressure
    summary_df_no_high_pressure = summary_df_no_high_pressure.merge(area_scores_df, on='Participant', how='inner')









    # summary_metrics = calculate_metrics(summary_df['Corrected Velocity'])
    # print(summary_metrics)
    
    skewness = []
    kurtosis = []
    for participant in summary_df['Participant'].unique():
        participant_df = summary_df[summary_df['Participant'] == participant]
        participant_df_nhp = summary_df_no_high_pressure[summary_df_no_high_pressure['Participant'] == participant]
        participant_metrics = calculate_metrics(participant_df['Corrected Velocity'])
        skewness.append([participant,participant_metrics['skewness']])
        kurtosis.append([participant,participant_metrics['kurtosis']])
    
        # # Plot density
        # sns.kdeplot(summary_df['Corrected Velocity'], label='Entire Dataset', fill=True)
        # sns.kdeplot(participant_df['Corrected Velocity'], label=participant, fill=True, alpha=0.5)
        # plt.legend()
        # plt.title('Density Plot of Entire Dataset vs. Subset')
        # plt.show()

        # # Plot CDF
        # plot_cdf(summary_df['Corrected Velocity'], subsets= [participant_df['Corrected Velocity']], labels=['Entire Dataset', participant], title = f'CDF Comparison of velocities for {participant}')
        # plot_cdf(summary_df_no_high_pressure['Corrected Velocity'], subsets= [participant_df_nhp['Corrected Velocity']], labels=['Entire Dataset', participant], title = f'CDF Comparison of velocities for {participant} nhp', write=True)
        # plot_cdf_comp_pressure(participant_df)
        # plot_cdf_comp_pressure(participant_df_nhp)    
    
    # Plot median velocity by participant
    median_velocity_per_participant = summary_df_no_high_pressure.groupby('Participant')['Corrected Velocity'].median().sort_values()
    sorted_participant_indices = {participant: index for index, participant in enumerate(median_velocity_per_participant.index)}
    
    # plt.figure(figsize=(10, 6))
    # plt.bar(sorted_participant_indices.values(), median_velocity_per_participant.values, width=0.5)
    # plt.xlabel('Participant')
    # plt.ylabel('Median Corrected Velocity')
    # plt.title('Median Corrected Velocity for Each Participant')
    # plt.xticks(list(sorted_participant_indices.values()), list(sorted_participant_indices.keys()), rotation=45)
    # plt.show()
        
    run_regression(summary_df_no_high_pressure)
    

        
        

    
  
    # ####### Favorite Capillaries ######
    if platform.system() == 'Windows':
        if 'gt8mar' in os.getcwd():
            favorite_capillaries = pd.read_excel('C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\chosen_caps.xlsx', sheet_name='Sheet1')
        else:
            favorite_capillaries = pd.read_excel('C:\\Users\\gt8ma\\capillary-flow\\results\\velocities\\chosen_caps.xlsx', sheet_name='Sheet1')
    
    favorite_capillaries = favorite_capillaries.rename(columns={'Chosen Capillary': 'Capillary'})

    # slice summary_df into favorite capillaries if capillary, location, and participant match
    favorite_df = summary_df.merge(favorite_capillaries, on=['Participant', 'Location', 'Capillary'], how='inner')

    # save to csv
    # favorite_df.to_csv('C:\\Users\\gt8ma\\capillary-flow\\favorite_caps.csv', index=False)
    # print(favorite_df.columns)

    # remove part22 and part23
    favorite_df = favorite_df[~favorite_df['Participant'].isin(['part22', 'part23'])]
    
    # plot_histograms(favorite_df, 'Age')
    # plot_histograms(favorite_df, 'SYS_BP')
    
    # plot_loc_histograms(favorite_df, 'Age')
    # plot_loc_histograms(favorite_df, 'SYS_BP')
    # plot_densities(favorite_df)

    favorite_df_no_high_pressure = favorite_df[favorite_df['Pressure'] <= 1.2]
    # plot_hist_pressure(favorite_df_no_high_pressure, density=True)
    # plot_densities(favorite_df_no_high_pressure)
    # plot_cdf(favorite_df_no_high_pressure['Corrected Velocity'], subsets= [favorite_df_no_high_pressure[favorite_df_no_high_pressure['Age'] > 50]['Corrected Velocity'], favorite_df_no_high_pressure[favorite_df_no_high_pressure['Age'] <= 50]['Corrected Velocity']], labels=['Entire Dataset', 'Old', 'Young'], title = 'CDF Comparison by Age')
    # plot_cdf(favorite_df_no_high_pressure['Corrected Velocity'], subsets= [favorite_df_no_high_pressure[favorite_df_no_high_pressure['SYS_BP'] > 120]['Corrected Velocity'], favorite_df_no_high_pressure[favorite_df_no_high_pressure['SYS_BP'] <= 120]['Corrected Velocity']], labels=['Entire Dataset', 'High BP', 'Normal BP'], title = 'CDF Comparison by BP')

    highBP_old_fav_nhp = favorite_df_no_high_pressure[(favorite_df_no_high_pressure['SYS_BP'] > 120) & (favorite_df_no_high_pressure['Age'] > 50)]['Corrected Velocity']
    highBP_young_fav_nhp = favorite_df_no_high_pressure[(favorite_df_no_high_pressure['SYS_BP'] > 120) & (favorite_df_no_high_pressure['Age'] <= 50)]['Corrected Velocity']
    normBP_old_fav_nhp = favorite_df_no_high_pressure[(favorite_df_no_high_pressure['SYS_BP'] <= 120) & (favorite_df_no_high_pressure['Age'] > 50)]['Corrected Velocity']
    normBP_young_fav_nhp = favorite_df_no_high_pressure[(favorite_df_no_high_pressure['SYS_BP'] <= 120) & (favorite_df_no_high_pressure['Age'] <= 50)]['Corrected Velocity']

    # # plot_cdf(favorite_df_no_high_pressure['Corrected Velocity'], subsets= [highBP_old_fav, highBP_young_fav, normBP_old_fav, normBP_young_fav], labels=['Entire Dataset', 'High BP Old', 'High BP Young', 'Normal BP Old', 'Normal BP Young'], title = 'CDF Comparison by Age and BP')
    # plot_hist_specific_pressure(favorite_df, 0.2, density=True, hist=False)
    # plot_hist_specific_pressure(favorite_df, 0.8, density=True, hist=False)
    # plot_hist_specific_pressure(favorite_df, 1.2, density=True, hist=False)
    # plot_cdf_comp_pressure(favorite_df)

    # plot_hist_comp_pressure(summary_df_no_high_pressure, density=True, hist=False)

    favorite_metrics = calculate_metrics(favorite_df['Corrected Velocity'])

    # plot velocities for each participant:
    for participant in favorite_df['Participant'].unique():
        favorite_df_copy = favorite_df.copy()
        participant_df = favorite_df_copy[favorite_df_copy['Participant'] == participant]

        # plot_velocities(participant_df, write = True)
        # plot_densities_individual(summary_df, participant_df, participant)
        # plot_densities_pressure_individual(summary_df, participant_df, participant)



        # Group the data by 'Capillary'
        grouped_df = participant_df.groupby('Capillary')
        # Get the unique capillary names
        capillaries = participant_df['Capillary'].unique()
        participant = participant_df['Participant'].unique()[0]
        


        # Plot each capillary's data in separate subplots
        for i, capillary in enumerate(capillaries):
            print(f'Participant: {participant}, Capillary: {capillary}')
            capillary_data = grouped_df.get_group(capillary)
            capillary_data = capillary_data.copy()
            # decreases = capillary_data['Pressure'].diff() < 0
            
            # print(decreases)  

            # create column for "Up/down" in pressure by calling all videos after the maximum pressure 'down'
            capillary_data.loc[:, 'Up/Down'] = 'Up'
            max_pressure = capillary_data['Pressure'].max()
            max_index = capillary_data['Pressure'].idxmax()
            capillary_data.loc[max_index:, 'Up/Down'] = 'Down'

            # create function to fit a curve to the up and down data, respectively
            data_up = capillary_data[capillary_data['Up/Down'] == 'Up']
            data_down = capillary_data[capillary_data['Up/Down'] == 'Down']
            curve_up = plot_and_calculate_area(data_up, plot = False, normalize = False)
            curve_down = plot_and_calculate_area(data_down, plot = False, normalize = False)
            hysterisis = curve_up + curve_down
            # print(f'Participant: {participant}, Capillary: {capillary}, Hysterisis: {hysterisis}')
            
            # add hysterisis to the favorite_df
            favorite_df.loc[(favorite_df['Participant'] == participant) & (favorite_df['Capillary'] == capillary), 'Hysterisis'] = hysterisis
            
    # # Plot scatter of age vs hysterisis 
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x='Age', y='Hysterisis', data=favorite_df)
    # plt.title('Hysterisis vs Age')
    # plt.xlabel('Age')
    # plt.ylabel('Hysterisis')
    # plt.show()

    

           

                          
        


    return 0
    
if __name__ == '__main__':
    main()

    