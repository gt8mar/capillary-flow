import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import shift
import cv2

# Load the data
data = np.loadtxt('D:\\frog\\test_kymo_claude.csv', delimiter=',')

def straighten_kymograph(kymograph, velocity):
    """
    Straighten the kymograph to create vertical lines.
    
    Parameters:
    kymograph (ndarray): The original kymograph
    velocity (float): Velocity in pixels per frame
    
    Returns:
    ndarray: The straightened kymograph
    """
    num_rows, num_cols = kymograph.shape
    straightened = np.zeros_like(kymograph)
    
    for i in range(num_rows):
        offset = int(round(i * velocity))
        straightened[i] = np.roll(kymograph[i], -offset)
    
    return straightened

def rotate_image_opencv(image, angle=45):
    """
    Rotate an image using OpenCV.
    
    Parameters:
    image (ndarray): Input image
    angle (float): Rotation angle in degrees
    
    Returns:
    ndarray: Rotated image
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    
    return rotated

# Method 2: Using scikit-image
from skimage import transform

def rotate_image_skimage(image, angle=45):
    """
    Rotate an image using scikit-image.
    
    Parameters:
    image (ndarray): Input image
    angle (float): Rotation angle in degrees
    
    Returns:
    ndarray: Rotated image
    """
    # Rotate the image
    rotated = transform.rotate(image, angle, resize=False, preserve_range=True)
    
    return rotated.astype(np.uint8)

def find_optimal_velocity(kymograph, velocity_range):
    """
    Find the optimal velocity for straightening.
    
    Parameters:
    kymograph (ndarray): The original kymograph
    velocity_range (tuple): Range of velocities to try (start, stop, step)
    
    Returns:
    float: Optimal velocity
    """
    best_velocity = 0
    max_vertical_sum = 0
    
    for velocity in np.arange(*velocity_range):
        straightened = straighten_kymograph(kymograph, velocity)
        vertical_sum = np.sum(np.std(straightened, axis=0))
        
        if vertical_sum > max_vertical_sum:
            max_vertical_sum = vertical_sum
            best_velocity = velocity
    
    return best_velocity

def autocorrelation(x):
    """Compute autocorrelation of 1D array."""
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]


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
# straightened_kymograph = straighten_kymograph(data, average_velocity)
straightened_kymograph = rotate_image_opencv(data, 5)

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

# Find optimal velocity
optimal_velocity = find_optimal_velocity(data, (0.1, 2, 0.1))
print(f"Optimal velocity: {optimal_velocity:.2f} pixels/frame")

# Straighten the kymograph
straightened_kymograph = straighten_kymograph(data, optimal_velocity)
straightened_kymograph = rotate_image_opencv(straightened_kymograph, 45)

# Compute average signal and its autocorrelation
avg_signal = np.mean(straightened_kymograph, axis=0)
auto_corr = autocorrelation(avg_signal)

# Find peaks in autocorrelation
peaks, _ = signal.find_peaks(auto_corr, height=0, distance=5)

if len(peaks) > 1:
    # Calculate periodicity
    periodicity = np.mean(np.diff(peaks))
    frequency = 1 / periodicity
    print(f"Estimated periodicity: {periodicity:.2f} frames")
    print(f"Estimated frequency: {frequency:.4f} cycles/frame")
else:
    print("Could not estimate periodicity - not enough peaks found.")

# Plotting
plt.figure(figsize=(15, 10))

# Original kymograph
plt.subplot(2, 2, 1)
plt.imshow(data, cmap='gray', aspect='auto')
plt.title('Original Kymograph')
plt.ylabel('Position')

# Straightened kymograph
plt.subplot(2, 2, 2)
plt.imshow(straightened_kymograph, cmap='gray', aspect='auto')
plt.title(f'Straightened Kymograph (v={optimal_velocity:.2f})')
plt.ylabel('Position')

# Average signal
plt.subplot(2, 2, 3)
plt.plot(avg_signal)
plt.title('Average Signal Across Positions')
plt.xlabel('Time')
plt.ylabel('Average Intensity')

# Autocorrelation
plt.subplot(2, 2, 4)
plt.plot(auto_corr)
plt.plot(peaks, auto_corr[peaks], "x")
plt.title('Autocorrelation of Average Signal')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')

plt.tight_layout()
plt.show()