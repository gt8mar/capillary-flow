import os, platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
import seaborn as sns
from scipy.integrate import simps, trapezoid
from scipy.stats import skew, kurtosis  

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
def plot_velocities(participant_df):
    # Group the data by 'Capillary'
    grouped_df = participant_df.groupby('Capillary')
    # Get the unique capillary names
    capillaries = participant_df['Capillary'].unique()
    participant = participant_df['Participant'].unique()[0]

    # Create subplots
    num_plots = len(capillaries)
    num_rows = (num_plots + 3) // 4  # Calculate the number of rows needed

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, 2 * num_rows), sharey=True, sharex=True)

    # Flatten the axes array to make it easier to iterate over
    axes = axes.flatten()

    # Plot each capillary's data in separate subplots
    for i, capillary in enumerate(capillaries):
        capillary_data = grouped_df.get_group(capillary)
        ax = axes[i]
        # ax.plot(capillary_data['Pressure'], capillary_data['Corrected Velocity'], marker='o', linestyle='-', label='Velocity')
        # Label all points which decrease in pressure with a red dot
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
        # Plot data up
        ax.plot(data_up['Pressure'], data_up['Corrected Velocity'],
                marker = 'o', linestyle = '-', label='Increase in Pressure')  # Use 'ro' for red dots
        # Plot data down in purple
        ax.plot(data_down['Pressure'], data_down['Corrected Velocity'], color= 'purple', 
                marker = 'o', linestyle = '-', label='Decrease in Pressure')  # Use 'ro' for red dots
        ax.set_xlabel('Pressure (psi)')
        ax.set_ylabel('Velocity (um/s)')
        ax.set_title(f'{participant} Capillary {capillary}')
        ax.grid(True)
        
        # Add a legend to the subplot
        ax.legend()

    # If there are unused subplots, remove them
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    # Adjust spacing between subplots to prevent label overlap
    plt.tight_layout()
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
            print(f'the binned velocities are: {df['Corrected Velocity']} to {df["Velocity Binned"]}')
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
    

    # plot_histograms(summary_df, 'Age')
    # plot_histograms(summary_df, 'SYS_BP')
    # # print(summary_df.head())

    summary_metrics = calculate_metrics(summary_df['Corrected Velocity'])
    print(summary_metrics)
    skewness = []
    kurtosis = []
    for participant in summary_df['Participant'].unique():
        participant_df = summary_df[summary_df['Participant'] == participant]
        participant_metrics = calculate_metrics(participant_df['Corrected Velocity'])
        skewness.append([participant,participant_metrics['skewness']])
        kurtosis.append([participant,participant_metrics['kurtosis']])
    
        # # Plot density
        # sns.kdeplot(summary_df['Corrected Velocity'], label='Entire Dataset', fill=True)
        # sns.kdeplot(participant_df['Corrected Velocity'], label=participant, fill=True, alpha=0.5)
        # plt.legend()
        # plt.title('Density Plot of Entire Dataset vs. Subset')
        # plt.show()
    
    # # Subset data into old vs young
    # old_df = summary_df[summary_df['Age'] > 50]
    # young_df = summary_df[summary_df['Age'] <= 50]
    # # Plot density
    # sns.kdeplot(summary_df['Corrected Velocity'], label='Entire Dataset', fill=True)
    # sns.kdeplot(old_df['Corrected Velocity'], label='old', fill=True, alpha=0.5)
    # sns.kdeplot(young_df['Corrected Velocity'], label='young', fill=True, alpha=0.5)
    # plt.legend()
    # plt.title('Density Plot of Entire Dataset vs. Subset')
    # plt.show()

    # # Subset data into low BP vs high BP
    # normBP_df = summary_df[summary_df['SYS_BP'] <= 120]
    # highBP_df = summary_df[summary_df['SYS_BP'] > 120]
    # print(f'the participants with high BP are: {highBP_df["Participant"].unique()}')

    # # Plot density
    # sns.kdeplot(summary_df['Corrected Velocity'], label='Entire Dataset', fill=True)
    # sns.kdeplot(normBP_df['Corrected Velocity'], label='normal', fill=True, alpha=0.5)
    # sns.kdeplot(highBP_df['Corrected Velocity'], label='high BP', fill=True, alpha=0.5)
    # plt.legend()
    # plt.title('Density Plot of Entire Dataset vs. Subset')
    # plt.show()

    # # Plot density of old high BP vs young high BP vs old low BP vs young low BP
    # old_highBP_df = old_df[old_df['SYS_BP'] > 120]
    # young_highBP_df = young_df[young_df['SYS_BP'] > 120]
    # old_normBP_df = old_df[old_df['SYS_BP'] <= 120]
    # young_normBP_df = young_df[young_df['SYS_BP'] <= 120]

    # # Plot density
    # sns.kdeplot(summary_df['Corrected Velocity'], label='Entire Dataset', fill=True)
    # sns.kdeplot(old_highBP_df['Corrected Velocity'], label='old high BP', fill=True, alpha=0.5)
    # sns.kdeplot(young_highBP_df['Corrected Velocity'], label='young high BP', fill=True, alpha=0.5)
    # sns.kdeplot(old_normBP_df['Corrected Velocity'], label='old normal BP', fill=True, alpha=0.5)
    # sns.kdeplot(young_normBP_df['Corrected Velocity'], label='young normal BP', fill=True, alpha=0.5)
    # plt.legend()
    # plt.title('Density Plot of Entire Dataset vs. Subset')
    # plt.show()

    # # compare high BP old vs young
    # # Plot density
    # sns.kdeplot(summary_df['Corrected Velocity'], label='Entire Dataset', fill=True)
    # sns.kdeplot(old_highBP_df['Corrected Velocity'], label='old high BP', fill=True, alpha=0.5)
    # sns.kdeplot(young_highBP_df['Corrected Velocity'], label='young high BP', fill=True, alpha=0.5)
    # plt.legend()
    # plt.title('Density Plot of high BP participants')
    # plt.show()

    # # compare low BP old vs young
    # # Plot density
    # sns.kdeplot(summary_df['Corrected Velocity'], label='Entire Dataset', fill=True)
    # sns.kdeplot(old_normBP_df['Corrected Velocity'], label='old normal BP', fill=True, alpha=0.5)
    # sns.kdeplot(young_normBP_df['Corrected Velocity'], label='young normal BP', fill=True, alpha=0.5)
    # plt.legend()
    # plt.title('Density Plot of normal BP participants')
    # plt.show()

    # Plot median velocity by participant
    median_velocity_per_participant = summary_df.groupby('Participant')['Corrected Velocity'].median().sort_values()
    sorted_participant_indices = {participant: index for index, participant in enumerate(median_velocity_per_participant.index)}
    
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_participant_indices.values(), median_velocity_per_participant.values, width=0.5)
    plt.xlabel('Participant')
    plt.ylabel('Median Corrected Velocity')
    plt.title('Median Corrected Velocity for Each Participant')
    plt.xticks(list(sorted_participant_indices.values()), list(sorted_participant_indices.keys()), rotation=45)
    plt.show()
        
    


    # plt.figure(figsize=(10, 6))
    # sns.barplot(x = [x[0] for x in skewness], y = [x[1] for x in skewness])
    # plt.title('Skewness by Participant')
    # plt.xlabel('Participant')
    # plt.ylabel('Skewness')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # sns.barplot(x = [x[0] for x in kurtosis], y = [x[1] for x in kurtosis])
    # plt.title('Kurtosis by Participant')
    # plt.xlabel('Participant')
    # plt.ylabel('Kurtosis')
    # plt.show()

        
        

    
  
    # ####### Favorite Capillaries ######
    # favorite_capillaries = pd.read_excel('C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\chosen_caps.xlsx', sheet_name='Sheet1')
    favorite_capillaries = pd.read_excel('C:\\Users\\gt8ma\\capillary-flow\\results\\chosen_caps.xlsx', sheet_name='Sheet1')
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

    favorite_metrics = calculate_metrics(favorite_df['Corrected Velocity'])

    # plot velocities for each participant:
    for participant in favorite_df['Participant'].unique():
        favorite_df_copy = favorite_df.copy()
        participant_df = favorite_df_copy[favorite_df_copy['Participant'] == participant]
        # plot_velocities(participant_df)
    
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

    