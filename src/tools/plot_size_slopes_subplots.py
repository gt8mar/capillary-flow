import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted  # for natural sorting of participant names
import time

def plot_size_slopes():
    file_path = "C:\\Users\\Luke\\Documents\\capillary-flow\\temp\\slopes.csv"

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None, names=['Name', 'Slope'])

    df = df.dropna()

    # Extract participant and direction from the 'Name' column
    df['Participant'] = df['Name'].str.extract(r'(part\d+)', expand=False)
    df['Direction'] = df['Name'].str.extract(r'(inc|dec)', expand=False)

    # Create a new column for sorting purposes
    df['SortOrder'] = df['Direction'] + '_' + df['Participant']

    # Sort the DataFrame by the 'SortOrder' column
    df = df.sort_values(by='SortOrder', key=lambda x: natsorted(x))

    # Create subplots with two columns
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Plot for increasing values (left subplot)
    inc_df = df[df['Direction'] == 'inc']
    sns.violinplot(ax=axs[0], x='Participant', y='Slope', data=inc_df, order=natsorted(inc_df['Participant'].unique()))
    axs[0].set_title('Increasing Values')
    axs[0].set_xlabel('Participant')
    axs[0].set_ylabel('Slope Values')

    # Plot for decreasing values (right subplot)
    dec_df = df[df['Direction'] == 'dec']
    sns.violinplot(ax=axs[1], x='Participant', y='Slope', data=dec_df, order=natsorted(dec_df['Participant'].unique()))
    axs[1].set_title('Decreasing Values')
    axs[1].set_xlabel('Participant')
    axs[1].set_ylabel('Slope Values')

    # Set x-axis ticks for each subplot based on the data
    axs[0].set_xticks(range(len(inc_df['Participant'].unique())))
    axs[0].set_xticklabels(natsorted(inc_df['Participant'].unique()), rotation=45, ha='right')

    axs[1].set_xticks(range(len(dec_df['Participant'].unique())))
    axs[1].set_xticklabels(natsorted(dec_df['Participant'].unique()), rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    
def plot_slope_variance():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('C:\\Users\\Luke\\Documents\\capillary-flow\\slopes.csv', header=None, names=['Name', 'Slope'])

    # Extract participant and type (inc/dec) information from the 'Name' column
    df['Participant'] = df['Name'].str.extract(r'part(\d+)')
    df['Type'] = df['Name'].str.extract(r'([a-zA-Z]+)_')

    # Filter data for increasing and decreasing slopes
    inc_data = df[df['Type'] == 'inc']
    dec_data = df[df['Type'] == 'dec']

    # Sort the DataFrame based on the participant number
    inc_data = inc_data.sort_values(by='Participant')
    dec_data = dec_data.sort_values(by='Participant')

    # Calculate variance of slope values for each participant
    inc_variance = inc_data.groupby('Participant')['Slope'].var()
    dec_variance = dec_data.groupby('Participant')['Slope'].var()

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    ax1.scatter(inc_variance.index, inc_variance.values, label='Variance of Inc Slopes')
    ax1.set_ylabel('Variance')
    ax1.set_title('Variance of Slope Values for Increasing Data')
    ax1.legend()

    ax2.scatter(dec_variance.index, dec_variance.values, label='Variance of Dec Slopes', color='orange')
    ax2.set_xlabel('Participant')
    ax2.set_ylabel('Variance')
    ax2.set_title('Variance of Slope Values for Decreasing Data')
    ax2.legend()

    plt.show()

if __name__ == "__main__":
    ticks = time.time()
    plot_size_slopes()
    plot_slope_variance()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))