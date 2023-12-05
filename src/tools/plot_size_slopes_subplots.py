import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted  # for natural sorting of participant names
import time

def plot_size_slopes():
    # Assuming your CSV file is named 'your_file.csv'
    file_path = 'C:\\Users\\Luke\\Documents\\capillary-flow\\slopes.csv'

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
    print(df)

    # Create a violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='SortOrder', y='Slope', data=df, order=natsorted(df['SortOrder'].unique()))
    plt.xlabel('Participant and Direction')
    plt.ylabel('Slope Values')
    plt.title('Violin Plot of Slope Values by Participant and Direction')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
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