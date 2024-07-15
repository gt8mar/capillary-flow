import os
import time
import cv2
import numpy as np
import pandas as pd

def get_condition(filename):
    start_index = filename.find('WkSl')
    end_index = filename.find('Frog')
    start_index += len('WkSl')
    return filename[start_index:end_index].strip()

def subtract_avg(img):
    for row in range(img.shape[0]):
        avg = np.mean(img[row])
        img[row] = (255/2)*img[row]/(avg + 1)
    return img

def main(path, date, frog, side):
    kymograph_folder = os.path.join(path, 'kymographs')
    rbc_count_folder = os.path.join(path, 'rbc_count')
    os.makedirs(rbc_count_folder, exist_ok=True)
    
    csv_filename = os.path.join(rbc_count_folder, 'counts.csv')
    csv_data = []

    # Loop through all TIFF files in kymograph folder
    for filename in os.listdir(kymograph_folder):
        if filename.endswith('.tiff'):
            image_path = os.path.join(kymograph_folder, filename)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            
            img = subtract_avg(img)
            img = cv2.GaussianBlur(img, (5,5), 0)
            img = cv2.equalizeHist(img)
            _, binary_image = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY_INV)
            _, labeled_regions = cv2.connectedComponents(binary_image)
            
            min_size = 0.5 * img.shape[0]
            maxima_count = 0

            # Create overlay image with labels
            overlay_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for label in range(1, np.max(labeled_regions) + 1):

                region_mask = np.uint8(labeled_regions == label)
                maxima_coords = np.argwhere(region_mask * (img == np.max(img * region_mask)))
                
                if len(maxima_coords) > 0:
                    # Check if the connected component size is greater than min_size
                    if np.sum(region_mask) > min_size:
                        maxima_count += 1
                        overlay_img[region_mask > 0] = (0, 0, 255)  # Overlay in red color
            
            # Save the overlay image
            overlay_filename = os.path.join(rbc_count_folder, f'{os.path.splitext(filename)[0]}_overlay.tiff')
            cv2.imwrite(overlay_filename, overlay_img)

            # Save data for CSV
            capnum = filename[filename.find('.tiff') - 1]
            condition = get_condition(filename)
            csv_data.append({'Date': date, 'Frog': frog, 'Side': side, 'Condition': condition, 'Capillary': capnum, 'RBC Count': maxima_count})

    # Create DataFrame and save CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_filename, index=False)

                

    # Write edge counts to CSV file
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_filename, index=False)    



if __name__ == "__main__":
    ticks = time.time()
    umbrella_folder = 'J:\\frog\\data'
    for date in os.listdir(umbrella_folder):
        if not date.startswith('24'):
            continue
        if date == 'archive' or date != '240213': 
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
                main(path, date, frog, side)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))