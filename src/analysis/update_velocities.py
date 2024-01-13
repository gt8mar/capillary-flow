import os, platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_velocities(df, write = True, verbose = False):
    # Group the data by 'Capillary'
    grouped_df = df.groupby('Capillary')
    # Get the unique capillary names
    capillaries = df['Capillary'].unique()

    # Get the participant and location
    participant = df['Participant'][0]
    location = df['Location'][0]

    # Create subplots
    num_plots = len(capillaries)
    num_rows = (num_plots + 3) // 4  # Round up to the nearest integer

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(10, 2 * num_rows), sharey=True, sharex=True)

    # Flatten the 2x2 subplot array to make it easier to iterate over
    axes = axes.flatten()

    # Plot each capillary's data in separate subplots
    for i, capillary in enumerate(capillaries):
        capillary_data = grouped_df.get_group(capillary)
        ax = axes[i]
        ax.plot(capillary_data['Pressure'], capillary_data['Corrected Velocity'], marker='o', linestyle='-')
        # Label all points which decrease in pressure with a red dot
        ax.plot(capillary_data.loc[capillary_data['Pressure'].diff() < 0, 'Pressure'],
                capillary_data.loc[capillary_data['Pressure'].diff() < 0, 'Corrected Velocity'],
                marker='o', linestyle='-', color='purple')
        # Label all points which are false, false, false with a gray dot
        ax.plot(capillary_data.loc[(capillary_data['Correct'] == 'f') & 
                                   (capillary_data['Zero'] == 'f') &
                                   (capillary_data['Max'] == 'f'), 'Pressure'],
                capillary_data.loc[(capillary_data['Correct'] == 'f') & 
                                   (capillary_data['Zero'] == 'f') &
                                   (capillary_data['Max'] == 'f'), 'Corrected Velocity'],
                marker='o', linestyle='', color='gray')
        # Label all points which are false, false, false and too slow with a blue dot
        ax.plot(capillary_data.loc[(capillary_data['Correct'] == 'f') & 
                                   (capillary_data['Zero'] == 'f') &
                                   (capillary_data['Max'] == 'f') &
                                   (capillary_data['Notes'] == 'too slow'), 'Pressure'],
                capillary_data.loc[(capillary_data['Correct'] == 'f') & 
                                   (capillary_data['Zero'] == 'f') &
                                   (capillary_data['Max'] == 'f') &
                                   (capillary_data['Notes'] == 'too slow'), 'Corrected Velocity'],
                marker='o', linestyle='', color='blue')
        # Label all points which are false, false, true and too fast with a red dot
        ax.plot(capillary_data.loc[(capillary_data['Correct'] == 'f') & 
                                   (capillary_data['Zero'] == 'f') &
                                   (capillary_data['Max'] == 'f') &
                                   (capillary_data['Notes'] == 'too fast'), 'Pressure'],
                capillary_data.loc[(capillary_data['Correct'] == 'f') & 
                                   (capillary_data['Zero'] == 'f') &
                                   (capillary_data['Max'] == 'f') &
                                   (capillary_data['Notes'] == 'too fast'), 'Corrected Velocity'],
                marker='o', linestyle='', color='red')
        ax.set_xlabel('Pressure (psi)')
        ax.set_ylabel('Velocity (um/s)')
        ax.set_title(f'Capillary {capillary}')
        ax.grid(True)

    # If there are unused subplots, remove them
    for i in range(num_plots, num_rows * 2):
        fig.delaxes(axes[i])

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    plt.suptitle(f'Participant {participant}, Location {location}', y=0.98)

    if write:
        if platform.system() == 'Windows':
            output_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\corrected'
            os.makedirs(output_folder, exist_ok=True)
            plt.savefig(os.path.join(output_folder, f"{participant}_{location}_velocity_vs_pressure_per_cap_corrected.png"), bbox_inches='tight', dpi=400)
        else:
            output_folder = '/hpc/projects/capillary-flow/results/velocities/corrected'
            os.makedirs(output_folder, exist_ok=True)            
            plt.savefig(os.path.join(output_folder, f"{participant}_{location}_velocity_vs_pressure_per_cap_corrected.png"), bbox_inches='tight', dpi=400)    
    if verbose:
        plt.show()
    else:
        plt.close()
    
    """
    --------------------------------- Plot the data on the same graph ---------------------------------------------------
    """
    
    fig, ax = plt.subplots()
    for name, group in grouped_df:
        ax.plot(group['Pressure'], group['Corrected Velocity'], marker='o', linestyle='', ms=12, label=name)
    
    ax.set_xlabel('Pressure (psi)')
    ax.set_ylabel('Velocity (um/s)')
    ax.set_title('Velocity vs. Pressure for each Capillary')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    # Set title to be participant and location:
    plt.suptitle(f'Participant {participant}, Location {location}', y=0.98)

    if write:
        if platform.system() == 'Windows':
            output_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\corrected'
            os.makedirs(output_folder, exist_ok=True)
            plt.savefig(os.path.join(output_folder, f"{participant}_{location}_velocity_vs_pressure_corrected.png"), bbox_inches='tight', dpi=400)
        else:
            output_folder = '/hpc/projects/capillary-flow/results/velocities/corrected'
            os.makedirs(output_folder, exist_ok=True)
            plt.savefig(os.path.join(output_folder, f"{participant}_{location}_velocity_vs_pressure_corrected.png"), bbox_inches='tight', dpi=400)
    if verbose:
        plt.show()
    else:
        plt.close()
    return 0


def update_velocities(csv_path):
    """
    This function updates the velocities in a csv file.

    Args:
        csv_path (str): the path to the csv file to be updated
    Returns:
        None
    Saves:
        csv file with updated velocities
    """
    print(f'Updating velocities in {csv_path}')
    # Read the csv file
    df = pd.read_csv(csv_path)

    Maxes = []
    # Iterate through each row
    for i in range(len(df)):
        # If the velocity is 0
        if df['Correct'][i] == 't':
            if df['Max'][i] == 't':
                Maxes.append([i, df['Velocity'][i]])
    print(Maxes)

    # max is the average of the maxes:
    max = np.mean([Max[1] for Max in Maxes])
    print(f'max = {max}')

    # make new column for corrected velocities
    df['Corrected Velocity'] = df['Velocity']

    for i in range(len(df)):
        if df['Max'][i] == 't':
            df['Corrected Velocity'][i] = max
        if df['Zero'][i] == 't':
            df['Corrected Velocity'][i] = 0

    print(df)

    # plot the corrected velocities organized by capillary:
    plot_velocities(df, write = True, verbose = False)

    # # Get the velocity column
    # velocities = df['Velocity']

    # # Get the velocity column as a numpy array
    # velocities_np = velocities.to_numpy()

    # # Get the number of rows
    # n_rows = velocities_np.shape[0]

    # # Get the number of columns
    # n_cols = velocities_np.shape[1]

    # # Iterate through the rows
    # for i in range(n_rows):
    #     # Iterate through the columns
    #     for j in range(n_cols):
    #         # If the velocity is 0
    #         if velocities_np[i,j] == 0:
    #             # Set the velocity to the previous velocity
    #             velocities_np[i,j] = velocities_np[i,j-1]

    # # Save the updated velocities
    # df['velocity'] = velocities_np
    # df.to_csv(csv_path, index=False)

                    

if __name__ == '__main__':
    # Usage example
    if platform.system() == 'Windows':
        velocities_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results\\velocities'
        for csv_file in os.listdir(velocities_folder):
            if csv_file.endswith('Copy.csv'):
                csv_file_path = os.path.join(velocities_folder, csv_file)
                update_velocities(csv_file_path)
    else:
        csv_file_path = '/hpc/projects/capillary-flow/results/velocities'
        update_velocities(csv_file_path)
