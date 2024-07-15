import os
import time
import cv2
import numpy as np
import pandas as pd
from skimage.transform import radon

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

            theta = np.linspace(0., 180., max(img.shape), endpoint=False)
            sinogram = radon(img, theta=theta, circle=True)
            
            # Find the angle corresponding to the maximum peak
            peak_angle_index = np.argmax(np.sum(sinogram, axis=0))
            peak_angle = theta[peak_angle_index]

            # Calculate the corresponding slope (tan(angle))
            target_slope = np.tan(np.deg2rad(peak_angle))

            # Define a tolerance for slope detection
            epsilon = 0.1  # Example tolerance, adjust based on your needs

            # Calculate the range of slopes to consider
            slope_min = np.tan(np.deg2rad(peak_angle - epsilon))
            slope_max = np.tan(np.deg2rad(peak_angle + epsilon))

            # Count occurrences of slopes within the specified range
            slopes = np.tan(np.deg2rad(theta))  # Calculate all slopes corresponding to angles
            count = np.sum((slopes >= slope_min) & (slopes <= slope_max))

            # Save data for CSV
            capnum = filename[filename.find('.tiff') - 1]
            condition = get_condition(filename)
            csv_data.append({'Date': date, 'Frog': frog, 'Side': side, 'Condition': condition, 'Capillary': capnum, 'RBC Count': count})

            # Save the image with pruned contours overlayed
            output_image_path = os.path.join(rbc_count_folder, f'{filename}_sinogram.tiff')
            cv2.imwrite(output_image_path, sinogram)

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