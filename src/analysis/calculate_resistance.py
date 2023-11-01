import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def get_resistances():
    csv_dir = 'C:\\Users\\Luke\\Documents\\capillary-flow\\velocity_csv'

    resistances = {}

    for file in os.listdir(csv_dir):
        print(file)
        participant = file.split("_")[1]
        if participant not in resistances:
            resistances[participant] = []

        df = pd.read_csv(os.path.join(csv_dir, file))[1:]
        data = df.values.tolist()

        data.sort(key=lambda x: (x[5], x[3]))
        unique_capnums = set([row[5] for row in data])

        for capnum in unique_capnums:
            velocity_initial = None
            velocity_final = None
            print(capnum)
            for entry in data:
                if entry[4] == 0.2 and entry[5] == capnum:
                    capnum = entry[5]
                    velocity_initial = float(entry[6])
                elif entry[4] == 1.2 and capnum == entry[5]:
                    velocity_final = float(entry[6])
                    break
            
            if velocity_final is not None and velocity_initial is not None:
                print(velocity_final)
                print(velocity_initial)
                resistance = velocity_final - velocity_initial
                resistances[participant].append(resistance)

    max_len = max([len(resistances[participant]) for participant in resistances])
    padded_resistances = {key: resistances[key] + [None] * (max_len - len(resistances[key])) for key in resistances}

    #save to csv
    df = pd.DataFrame.from_dict(padded_resistances)
    df.to_csv('C:\\Users\\Luke\\Documents\\capillary-flow\\resistances.csv', index=False)
            
def plot_resistances():
    resistances_fp = 'C:\\Users\\Luke\\Documents\\capillary-flow\\resistances.csv'
    data = pd.read_csv(resistances_fp)

    #metadata_dir = '/hpc/projects/capillary-flow/metadata'
    metadata_dir = 'C:\\Users\\Luke\\Documents\\capillary-flow\\metadata'

    participants = []

    # get birthdates
    age_dict = {}
    for file in os.listdir(metadata_dir):
        participant = file.split("_")[0]
        date = file.split("_")[1][:-5]
        date = pd.to_datetime(date, format='%y%m%d')

        if participant not in participants:
            participants.append(participant)
            birthday = pd.read_excel(os.path.join(metadata_dir, file))['Birthday'][1]
            birthday = pd.to_datetime(birthday, format='%Y%m%d')

            age = (date - birthday).days / 365

            age_dict[participant] = age
            
    participants.sort(key=lambda x: age_dict[x])

    fig, ax = plt.subplots()

    for i, participant in enumerate(participants):
        if participant in data:
            y = data[participant].dropna().values.tolist()
            age = [age_dict[participant]]*len(y)
            ax.scatter(age, y, label=participant)

    plt.title('Resistance by Age')
    plt.xlabel('Age')
    plt.ylabel('Resistance')

    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ticks = time.time()
    #get_resistances()
    plot_resistances()  
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))