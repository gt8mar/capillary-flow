import time
import os
import cv2
from enumerate_capillaries2 import find_connected_components
import csv

def main(path):
    segmented_folder = os.path.join(path, 'segmented')
    output_folder = os.path.join(path, 'individual_caps')
    os.makedirs(output_folder, exist_ok = True)
    for file in os.listdir(segmented_folder):
        if file.endswith('.png'):
            file_image = cv2.imread(os.path.join(segmented_folder, file), cv2.IMREAD_GRAYSCALE)
            caps = find_connected_components(file_image)
            for i, cap in enumerate(caps):
                file = file.replace('SD_', '')
                file = file.replace('.png', '')
                cv2.imwrite(os.path.join(output_folder, file + '_' + str(i).zfill(2) + '.png'), cap)

    # save all file names to csv
    csv_file = os.path.join(output_folder, 'cap_names.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for file in os.listdir(output_folder):
            if file.endswith('.png'):
                file = file.strip('.png')
                csvwriter.writerow([file])

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
    #main(path = 'D:\\frog\\data\\240530\\Frog5\\Right')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))