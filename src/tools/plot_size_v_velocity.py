import os
import time
import matplotlib.pyplot as plt
import pandas as pd

def match_size_vel():
    size_dir = 'C:\\Users\\Luke\\Documents\\capillary-flow\\size_data'
    vel_dir = 'C:\\Users\\Luke\\Documents\\capillary-flow\\velocity_csv'

    size_vel_data = []

    for size_csv in os.listdir(size_dir):
        size_part = size_csv.split("_")[0]
        size_date = size_csv.split("_")[1]
        size_loc = size_csv.split("_")[2]
        for vel_csv in os.listdir(vel_dir):
            vel_part = vel_csv.split("_")[1]
            vel_date = vel_csv.split("_")[2]
            vel_loc = vel_csv.split("_")[3]
            if size_part == vel_part and size_date == vel_date and size_loc == vel_loc:
                print(size_csv)
                size_df = pd.read_csv(os.path.join(size_dir, size_csv))
                size_data = size_df.values.tolist()
                vel_df = pd.read_csv(os.path.join(vel_dir, vel_csv))
                vel_data = vel_df.values.tolist()

                for size_row in size_data:
                    size_capnum = int(f'{size_row[5]:02}') 
                    size_vidnum = 'vid' + f'{size_row[6]:02}'
                    for vel_row in vel_data:
                        vel_capnum = int(''.join(filter(str.isdigit, vel_row[5])))
                        vel_vidnum = vel_row[3]
                        if size_capnum == vel_capnum and size_vidnum == vel_vidnum:
                            size_vel_data.append([size_part, size_date, size_loc, size_vidnum, size_capnum, size_row[4], size_row[3], vel_row[6]])
    size_vel_data.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    size_vel_df = pd.DataFrame(size_vel_data, columns=['participant', 'date', 'location', 'vidnum', 'capnum', 'pressure', 'area', 'velocity'])
    size_vel_df.to_csv('C:\\Users\\Luke\\Documents\\capillary-flow\\size_vel_data.csv', index=False)

def plot_size_vel():
    size_vel_csv = 'C:\\Users\\Luke\\Documents\\capillary-flow\\size_vel_data.csv'
    size_vel_data = pd.read_csv(size_vel_csv)

    area = size_vel_data['area']
    velocity = size_vel_data['velocity']
    participants = size_vel_data['participant']

    unique_participants = set(participants)

    fig, axes = plt.subplots((len(unique_participants) + 2) // 3, 3, sharex=True, sharey=True)

    for i, participant in enumerate(unique_participants):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        mask = (participants == participant)
        ax.scatter(area[mask], velocity[mask])

        ax.set_xlabel('Area (pixels)')
        ax.set_ylabel('Velocity (um/s)')
        ax.set_title(participant)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ticks = time.time()
    #match_size_vel()
    plot_size_vel()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))