import os
import time
import cv2
import numpy as np
import pandas as pd

def main(path):
    rbc_img_folder = os.path.join(path, 'rbc')
    df = pd.DataFrame(columns = ['Date', 'Video','Capillary', 'Position', 'RBC Count'])
    for img in os.listdir(rbc_img_folder):
        
        date = img.split(' ')[0]
        video = img.split(' ')[1].split('_')[0]
        capnum = img.split(' ')[1].split('_')[1]
        position = img.split(' ')[1].split('_')[2][:-5]

        image = cv2.imread(os.path.join(rbc_img_folder, img), cv2.IMREAD_GRAYSCALE)
        equalized_image = cv2.equalizeHist(image)
        _, binary_image = cv2.threshold(equalized_image, 20, 255, cv2.THRESH_BINARY_INV)
        _, labeled_regions = cv2.connectedComponents(binary_image)
        
        min_size = 1
        maxima_count = 0

        for label in range(1, np.max(labeled_regions) + 1):
            region_mask = np.uint8(labeled_regions == label)
            maxima_coords = np.argwhere(region_mask * (image == np.max(image * region_mask)))
            
            if len(maxima_coords) > 0:
                # Check if the connected component size is greater than min_size
                if np.sum(region_mask) > min_size:
                    maxima_count += 1
        
        #save
        new_data = pd.DataFrame([[date, video, capnum, position, maxima_count]], columns = df.columns)
        df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(os.path.join(rbc_img_folder, "rbc_counts.csv"), index=False)


if __name__ == '__main__':
    ticks = time.time()
    main(path = 'D:\\frawg\\Wake Sleep Pairs\\gabby_analysis')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))    