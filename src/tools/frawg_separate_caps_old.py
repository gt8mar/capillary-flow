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
    main(path = 'E:\\frog\\24-2-14 WkSl\\Frog4\\Right')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))