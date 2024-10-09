import time
import os
import csv
import shutil

def main(frog_side_path):
    print('Renaming individual capillaries in ' + frog_side_path)
    individual_caps_folder = os.path.join(frog_side_path, 'individual_caps')
    name_csv = os.path.join(individual_caps_folder, 'cap_names.csv')

    renamed_folder = os.path.join(frog_side_path, 'individual_caps_renamed')
    os.makedirs(renamed_folder, exist_ok=True)
    with open(name_csv, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            original_filename = row[0] + '.png'
            source_path = os.path.join(individual_caps_folder, original_filename)

            new_filename = row[0][:-2] + str(row[1]) + '.png' # remove the old capillary number and add the new name (letter)
            destination_path = os.path.join(renamed_folder, new_filename)
            shutil.copyfile(source_path, destination_path)

if __name__ == "__main__":
    ticks = time.time()
    umbrella_folder = '/hpc/projects/capillary-flow/frog/'
    for date in os.listdir(umbrella_folder):
        if not date.startswith('240729'):
            continue
        if date == 'archive':
            continue
        if date.endswith('alb'):
            continue
        for frog in os.listdir(os.path.join(umbrella_folder, date)):
            if frog.startswith('STD'):
                continue
            if not frog.startswith('Frog4'):
                continue
            for side in os.listdir(os.path.join(umbrella_folder, date, frog)):
                if side.startswith('STD'):
                    continue
                if not side.startswith('Left'): # only process the left side for now
                    continue
                if side == 'archive':
                    continue
                print('Processing: ' + date + ' ' + frog + ' ' + side)
                frog_side_path = os.path.join(umbrella_folder, date, frog, side)
                main(frog_side_path)
    #main(path = 'E:\\frog\\24-2-14 WkSl\\Frog4\\Right')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))