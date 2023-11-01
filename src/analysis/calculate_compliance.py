import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def get_compliances():
    size_data_dir = '/hpc/projects/capillary-flow/results/size/size_data'

    compliances = {}
    for file in os.listdir(size_data_dir):
        participant = file.split("_")[0]
        date = file.split("_")[1]
        location = file.split("_")[2]

        df = pd.read_csv(os.path.join(size_data_dir, file))[1:]
        data = df.values.tolist()
        # sort by capnum and vidnum
        data.sort(key=lambda x: (x[5], x[6]))

        if participant not in compliances:
            compliances[participant] = []

        unique_capnums = set([row[5] for row in data])

        for capnum in unique_capnums:
            area_initial = None
            area_final = None

            # get the area_initial and area_final values at 0.2 and 1.2 psi
            for entry in data:
                if entry[4] == 0.2 and entry[5] == capnum:
                    area_initial = float(entry[3])  
                elif entry[4] == 1.2 and entry[5] == capnum:
                    area_final = float(entry[3]) 
                    break

            # If both initial and final are found
            if area_initial is not None and area_final is not None:
                compliance = (area_final - area_initial) # divided by delta P = 1.2 - 0.2 = 1
                compliances[participant].append(compliance)

    max_len = max([len(compliances[participant]) for participant in compliances])
    padded_compliances = {key: compliances[key] + [None] * (max_len - len(compliances[key])) for key in compliances}

    # save to csv
    df = pd.DataFrame.from_dict(padded_compliances)
    df.to_csv('/hpc/projects/capillary-flow/results/size/compliances.csv', index=False)

def plot_compliances():
    #compliances_fp = '/hpc/projects/capillary-flow/results/size/compliances.csv'
    compliances_fp = 'C:\\Users\\Luke\\Documents\\capillary-flow\\compliances.csv'
    data = pd.read_csv(compliances_fp)

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

    plt.title('Compliance by Age')
    plt.xlabel('Age')
    plt.ylabel('Compliance')

    plt.tight_layout()
    plt.legend()
    plt.show()

def main():
    #get_compliances()
    plot_compliances()


"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))