import os
import time
import cv2
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

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

def main(path):
    kymograph_folder = os.path.join(path, 'kymographs')
    rbc_count_folder = os.path.join(path, 'rbc_count')
    os.makedirs(rbc_count_folder, exist_ok=True)
    
    csv_filename = os.path.join(rbc_count_folder, 'edge_counts.csv')
    csv_data = []

    # Loop through all TIFF files in kymograph folder
    for filename in os.listdir(kymograph_folder):
        if filename.endswith('.tiff'):
            image_path = os.path.join(kymograph_folder, filename)

            # Perform Canny edge detection
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Compute the autocorrelation function for each pixel over time
            kymograph = img
            kymograph_columns = kymograph.shape[1]
            kymograph_rows = kymograph.shape[0]
            acf_matrix = np.array([np.correlate(kymograph[pixel, :], kymograph[pixel, :], mode='full') for pixel in range(int(kymograph_rows))])
            acf_matrix = acf_matrix[:, acf_matrix.shape[1] // 2:]  # Take the positive lags

            # Normalize the ACF for each pixel
            acf_matrix = acf_matrix / acf_matrix[:, 0][:, np.newaxis]

            # Average the autocorrelation functions across all pixels
            avg_acf = np.mean(acf_matrix, axis=0)

            # Find peaks in the averaged autocorrelation function
            peaks, _ = find_peaks(avg_acf, height=0.2)  # Adjust height threshold as needed

            # Calculate the period from the peaks
            if len(peaks) > 1:
                periods = np.diff(peaks)
                estimated_period = np.mean(periods)
            else:
                estimated_period = None

            # Plot the kymograph
            plt.figure(figsize=(12, 6))
            plt.imshow(kymograph, aspect='auto', cmap='gray')
            plt.colorbar(label='Pixel Intensity')
            plt.title('Synthetic Kymograph')
            plt.xlabel('Time')
            plt.ylabel('Pixel Position')
            plt.show()

            # Plot the averaged autocorrelation function and peaks
            plt.figure(figsize=(12, 6))
            plt.plot(avg_acf, label='Averaged Autocorrelation')
            plt.plot(peaks, avg_acf[peaks], "x", label='Peaks')
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
            plt.title('Averaged Autocorrelation Function from Kymograph')
            plt.legend()
            plt.grid(True)
            plt.show()


            
            
            
    
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
                main(path)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))