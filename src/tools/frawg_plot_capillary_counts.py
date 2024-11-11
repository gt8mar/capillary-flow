import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PIX_UM = 0.8


# Read the data
data = pd.read_csv('D:\\frog\\counted_kymos_CalFrog4.csv')
# add fps column to data
data['FPS'] = data['Condition'].apply(lambda x: x.split('Lankle')[0])
# Read in capillary_count data
makai_count_100 = pd.read_csv('D:\\frog\\rbc_count\\manual\\frame_interval_100.csv')
makai_count_130 = pd.read_csv('D:\\frog\\rbc_count\\manual\\frame_interval_130.csv')
makai_count_220 = pd.read_csv('D:\\frog\\rbc_count\\manual\\frame_interval_220.csv')
makai_velocity_folder = 'D:\\frog\\counts_velocity_renamed'

makai_velocity_df = pd.DataFrame(columns=['FPS', 'Capillary', 'Makai_Velocity'])
for folder in os.listdir(makai_velocity_folder):
    if folder.startswith('fps'):
        fps_folder = os.path.join(makai_velocity_folder, folder)
        fps = folder.split('_')[-1]
        if fps == '220-2':
            fps = '220'
        for csv_file in os.listdir(fps_folder):
            if csv_file.endswith('.csv'):
                capillary = csv_file.split('.')[0].split('_')[-1]
                velocity_data = pd.read_csv(os.path.join(fps_folder, csv_file))
                length_per_3_frames = velocity_data['Length'].mean()
                # velocity = length_per_3_frames * int(fps) / 3 * (1/PIX_UM)
                velocity = length_per_3_frames  / 3 * (1/PIX_UM)

                makai_velocity_df = makai_velocity_df.append({'FPS': fps, 'Capillary': capillary, 'Makai_Velocity': velocity}, ignore_index=True)
                

# make new column in data where we divide the "Classified Velocity" by the FPS of that row
data['Classified_Velocity'] = data['Classified_Velocity'] / data['FPS'].astype(int)


# slice data to only use capillaries b, d, g, k, o
makai_data_1 = makai_velocity_df[makai_velocity_df['Capillary'].isin(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'])]
data_1 = data[data['Capillary'].isin(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'])]
sliced_data = data[data['Capillary'].isin(['b', 'd', 'g', 'k', 'o'])]
sliced_makai_data = makai_velocity_df[makai_velocity_df['Capillary'].isin(['b', 'd', 'g', 'k', 'o'])]
merged = pd.merge(sliced_data, sliced_makai_data, on=['FPS', 'Capillary'], how='inner')

# normalize the makai velocity data to have the same mean velocity as the classified data
# sliced_makai_data['Makai_Velocity'] = sliced_makai_data['Makai_Velocity'] * sliced_data['Classified_Velocity'].mean() / sliced_makai_data['Makai_Velocity'].mean()

# Function to plot velocity measurements
def plot_velocity_by_capillary(data):
    plt.figure(figsize=(15, 8))
    
    # Create a scatter plot
    sns.scatterplot(x='Capillary', y='Classified_Velocity', hue='Condition', data=data_1, palette='deep') # Velocity (um/s)
    sns.scatterplot(x='Capillary', y='Makai_Velocity', hue='FPS', data=makai_data_1, palette='dark') # Makai Velocity (um/s)
    
    plt.title('Velocity Measurements by Capillary and Condition', fontsize=16)
    plt.xlabel('Capillary', fontsize=12)
    plt.ylabel('Velocity (um/s)', fontsize=12)

    # make y axis log scale (optional)
    # data['Classified_Velocity'] = data['Classified_Velocity'] + 10
    # plt.yscale('log')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust legend
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    # plt.savefig('velocity_by_capillary.png')
    plt.show()
    plt.close()

def plot_velocity_by_makaivelocity(data, log_scale = False):
    plt.figure(figsize=(15, 8))
    sns.scatterplot(x='Makai_Velocity', y='Classified_Velocity', hue='FPS', data=data, palette='deep') # Velocity (um/s)
    plt.title('Velocity Measurements by Makai and Classified', fontsize=16)
    # make log scale
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    plt.xlabel('Makai Velocity (um/s)', fontsize=12)
    plt.ylabel('Classified Velocity (um/s)', fontsize=12)
    plt.legend(title='FPS', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


    return 0


# Call the function
plot_velocity_by_capillary(data)
plot_velocity_by_makaivelocity(merged)

print("Velocity plot has been saved as 'velocity_by_capillary.png'.")

# Original plotting function (kept for reference)
def plot_capillary_analysis(data):
    grouped = data.groupby('Capillary')

    for capillary, group in grouped:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Capillary {capillary}', fontsize=16)

        # Plot for Counts
        sns.barplot(x='Condition', y='Counts', data=group, ax=ax1)
        ax1.set_title('Counts by Condition')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_ylabel('Counts')

        # Plot for Velocity
        sns.barplot(x='Condition', y='Velocity (um/s)', data=group, ax=ax2)
        ax2.set_title('Velocity by Condition')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_ylabel('Velocity (um/s)')

        plt.tight_layout()
        # plt.savefig(f'capillary_{capillary}_analysis.png')
        plt.show()
        plt.close()

# Uncomment the line below if you want to run the original analysis as well
# plot_capillary_analysis(data)

print("Analysis complete. Plot images have been saved.")