import time
import os
import csv
import shutil

def main(path):
    individual_caps_folder = os.path.join(path, 'individual_caps')
    name_csv = os.path.join(individual_caps_folder, 'cap_names.csv')

    renamed_folder = os.path.join(path, 'individual_caps_renamed')
    os.makedirs(renamed_folder, exist_ok=True)

    with open(name_csv, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            original_filename = row[0] + '.png'
            source_path = os.path.join(individual_caps_folder, original_filename)

            new_filename = row[0][:-2] + str(row[1]) + '.png'
            destination_path = os.path.join(renamed_folder, new_filename)
            shutil.copyfile(source_path, destination_path)

if __name__ == "__main__":
    ticks = time.time()
    umbrella_folder = 'J:\\frog\\data'
    for date in os.listdir(umbrella_folder):
        if not date.startswith('24'):
            continue
        if date == 'archive':
            continue
        for frog in os.listdir(os.path.join(umbrella_folder, date)):
            if frog.startswith('STD'):
                continue
            if not frog.startswith('Frog'):
                continue
            for side in os.listdir(os.path.join(umbrella_folder, date, frog)):
                if side.startswith('STD'):
                    continue
                if side == 'archive':
                    continue
                print('Processing: ' + date + ' ' + frog + ' ' + side)
                path = os.path.join(umbrella_folder, date, frog, side)
                main(path)
    #main(path = 'E:\\frog\\24-2-14 WkSl\\Frog4\\Right')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))