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

def prune_contour(contour, epsilon_ratio=0.1):
    # Approximate the contour by a simpler polygon
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

def is_contour_closed(contour):
    return np.array_equal(contour[0], contour[-1])

def main(path, condition = 'WkSl'):
    kymograph_folder = os.path.join(path, 'kymographs')
    rbc_count_folder = os.path.join(path, 'rbc_count')
    os.makedirs(rbc_count_folder, exist_ok=True)
    csv_filename = os.path.join(rbc_count_folder, 'edge_counts.csv')
    csv_data = []

    # Loop through all TIFF files in kymograph folder
    for filename in os.listdir(kymograph_folder):
        if filename.endswith('.tiff'):
            image_path = os.path.join(kymograph_folder, filename)
            filename_no_zfill = filename.replace('.tiff', '').replace('_0', '_')
            capnum = filename_no_zfill.split('_')[-1]
            if condition == 'Calb':
                condition = filename_no_zfill.split('_')[0]
            else:
                condition = get_condition(filename)
            
            # Perform Canny edge detection
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = subtract_avg(img)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = cv2.equalizeHist(img)
            median_value = np.median(img)
            max_value = np.max(img)
            edges = cv2.Canny(img, 0.3*int(median_value), 0.3*int(max_value))

            # Find contours of the detected edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Prune contours to remove branches or curved components
            pruned_contours = [prune_contour(contour) for contour in contours]

            # Filter pruned contours based on length and closed condition
            height = img.shape[0]
            min_length = 0.5 * height
            filtered_contours = [contour for contour in pruned_contours 
                                if cv2.arcLength(contour, False) > min_length and not is_contour_closed(contour)]

            # Draw filtered contours on the original image
            cv2.drawContours(img, filtered_contours, -1, (255, 255, 255), 1)
            

            # Save the image with pruned contours overlayed
            output_image_path = os.path.join(rbc_count_folder, f'{filename}_edges.tiff')
            cv2.imwrite(output_image_path, img)
            
            # Count the number of pruned edges
            num_pruned_edges = len(pruned_contours)

            # Save data for CSV
            # capnum = filename[filename.find('.tiff') - 1]
            csv_data.append({'Date': date, 'Frog': frog, 'Side': side, 'Condition': condition, 'Capillary': capnum, 'RBC Count': num_pruned_edges})

    
    # Write edge counts to CSV file
    df = pd.DataFrame(csv_data)
    df = df.sort_values(by=['Capillary', 'Condition'])
    df.to_csv(csv_filename, index=False)    



if __name__ == "__main__":
    ticks = time.time()
    # umbrella_folder = 'J:\\frog\\data'
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
                if side == 'archive':
                    continue
                if not side.startswith('Left'): # only process the left side for now
                    continue
                print('Processing: ' + date + ' ' + frog + ' ' + side)
                path = os.path.join(umbrella_folder, date, frog, side)
                main(path, condition='Calb')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))