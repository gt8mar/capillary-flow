import time
import os
import cv2
from enumerate_capillaries2 import find_connected_components

def main(path):
    segmented_folder = os.path.join(path, 'segmented')
    output_folder = os.path.join(path, 'individual_caps')
    os.makedirs(output_folder, exist_ok = True)
    for file in os.listdir(segmented_folder):
        if file.endswith('.png'):
            file_image = cv2.imread(os.path.join(segmented_folder, file), cv2.IMREAD_GRAYSCALE)
            caps = find_connected_components(file_image)
            for i, cap in enumerate(caps):
                cv2.imwrite(os.path.join(output_folder, file[4:-4] + '_' + str(i).zfill(2) + '.png'), cap)

if __name__ == "__main__":
    ticks = time.time()
    main(path = 'E:\\frawg\\gabbyanalysis')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))