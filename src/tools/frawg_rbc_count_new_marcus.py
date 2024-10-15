import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import shift

# Load the data
data = np.loadtxt('D:\\frog\\test_kymo_claude.csv', delimiter=',')

def straighten_kymograph(kymograph, velocity):
    """
    Straighten the kymograph based on average velocity.
    
    Parameters:
    kymograph (ndarray): The original kymograph
    velocity (float): Average velocity in pixels per frame
    
    Returns:
    ndarray: The straightened kymograph
    """
    num_rows, num_cols = kymograph.shape
    straightened = np.zeros_like(kymograph)
    
    for i in range(num_rows):
        offset = int(i * velocity)
        straightened[i] = np.roll(kymograph[i], -offset)
    
    return straightened

def count_peaks_and_periodicity(signal_1d):
    """
    Count peaks and calculate periodicity in a 1D signal.
    
    Parameters:
    signal_1d (ndarray): 1D signal to analyze
    
    Returns:
    tuple: (number of peaks, average periodicity)
    """
    # Find peaks
    peaks, _ = signal.find_peaks(signal_1d, height=np.mean(signal_1d), distance=5)
    
    # Calculate periodicity
    if len(peaks) > 1:
        periods = np.diff(peaks)
        avg_periodicity = np.mean(periods)
    else:
        avg_periodicity = None
    
    return len(peaks), avg_periodicity

# Parameters
average_velocity = (660 * 0.8) / 220 # Adjust this based on your data: 660 um/s which is 220 fps and average_slope = (um_slope * 0.8) / fps
num_positions = data.shape[0]
time_points = data.shape[1]

# Straighten the kymograph
straightened_kymograph = straighten_kymograph(data, average_velocity)

# Analyze each position in the straightened kymograph
peak_counts = []
periodicities = []

for pos in range(num_positions):
    signal_1d = straightened_kymograph[pos, :]
    num_peaks, periodicity = count_peaks_and_periodicity(signal_1d)
    peak_counts.append(num_peaks)
    if periodicity is not None:
        periodicities.append(periodicity)

# Calculate average results
avg_peak_count = np.mean(peak_counts)
avg_periodicity = np.mean(periodicities)

# Plot results
plt.figure(figsize=(15, 10))

# Original kymograph
plt.subplot(2, 2, 1)
plt.imshow(data, cmap='gray', aspect='auto')
plt.title('Original Kymograph')
plt.ylabel('Position')

# Straightened kymograph
plt.subplot(2, 2, 2)
plt.imshow(straightened_kymograph, cmap='gray', aspect='auto')
plt.title('Straightened Kymograph')
plt.ylabel('Position')

# Peak counts
plt.subplot(2, 2, 3)
plt.plot(peak_counts)
plt.title('Number of Peaks Detected per Position')
plt.xlabel('Position')
plt.ylabel('Number of Peaks')

# Periodicities
plt.subplot(2, 2, 4)
plt.hist(periodicities, bins=20)
plt.title('Distribution of Periodicities')
plt.xlabel('Periodicity (frames)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Print results
print(f"Average number of peaks (cells) per position: {avg_peak_count:.2f}")
print(f"Average periodicity: {avg_periodicity:.2f} frames")
print(f"Estimated frequency: {1/avg_periodicity:.4f} cells/frame")