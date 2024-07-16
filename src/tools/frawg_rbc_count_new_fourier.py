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
            img = subtract_avg(img)

            # Prepare to collect peak data
            all_peaks = []

            # FFT for each row and find peaks
            for row in img:
                # Apply FFT
                fft_result = fft(row)
                fft_magnitude = np.abs(fft_result[:len(fft_result)//2])  # Take half, since FFT is symmetric
                
                # Find peaks in the FFT magnitude
                peaks, _ = find_peaks(fft_magnitude, height=10)  # You might need to adjust the height parameter
                
                # Collect peaks
                all_peaks.append(peaks)

                # Analyze consistency of peaks
                consistent_peaks = set(all_peaks[0])
                for peaks in all_peaks[1:]:
                    consistent_peaks.intersection_update(peaks)
            
                """# Plot a sample FFT result with peaks
                plt.figure(figsize=(10, 4))
                sample_fft_magnitude = np.abs(fft(img[0])[:len(img[0])//2])
                plt.plot(sample_fft_magnitude, label='FFT Magnitude')
                peaks, _ = find_peaks(sample_fft_magnitude, height=10)
                plt.scatter(peaks, sample_fft_magnitude[peaks], color='red', label='Peaks')
                plt.title('FFT of First Row with Peaks')
                plt.legend()
                plt.xlabel('Frequency')
                plt.ylabel('Magnitude')

                # Save the figure
                plt.savefig('fft_analysis.png', dpi=300)"""

            # Save the image with pruned contours overlayed
            """output_image_path = os.path.join(rbc_count_folder, f'{filename}_edges.tiff')
            cv2.imwrite(output_image_path, img)"""
            
            # Count the number of pruned edges
            csv_data.append({'Image': filename, 'RBC Count': consistent_peaks})

    
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