import time
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_area_hist():
    size_data_dir = 'C:\\Users\\Luke\\Documents\\capillary-flow\\size_data'

    participants = {}

    for file in os.listdir(size_data_dir):
        participant = file.split("_")[0]

        if participant not in participants:
            participants[participant] = []

        with open(os.path.join(size_data_dir, file), 'r') as f:
            f.readline()
            for line in f:
                participants[participant].append(float(line.split(",")[3]))

    num_plots = len(participants)
    num_cols = 3
    num_rows = -(-num_plots // num_cols)  # Calculate the number of rows needed, rounding up

    fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(15, 5*num_rows))
    axs = axs.flatten()  # Flatten the 2D array of subplots to a 1D array

    for i, participant in enumerate(participants):
        axs[i].hist(participants[participant], bins=10)
        axs[i].set_title(participant)

    # Remove any unused subplots
    for j in range(num_plots, num_rows * num_cols):
        fig.delaxes(axs[j])

    fig.suptitle('Histogram of Area Values')
    plt.show()

def plot_size_variance():
    size_data_dir = 'C:\\Users\\Luke\\Documents\\capillary-flow\\size_data'

    participants = {}

    for file in os.listdir(size_data_dir):
        participant = file.split("_")[0]

        if participant not in participants:
            participants[participant] = []

        with open(os.path.join(size_data_dir, file), 'r') as f:
            f.readline()
            for line in f:
                participants[participant].append(float(line.split(",")[3]))

    fig, ax = plt.subplots()
    for participant in participants:
        variance = np.var(participants[participant])
        ax.scatter(participant, variance)
    plt.title('Variance of Area Values')
    plt.show()




if __name__ == "__main__":
    ticks = time.time()
    #plot_area_hist()
    plot_size_variance()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))