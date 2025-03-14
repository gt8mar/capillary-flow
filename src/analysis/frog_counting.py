import glob, os
import numpy as np
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, convolve
from scipy.ndimage import gaussian_filter1d
from matplotlib.font_manager import FontProperties

FPS = 130 #113.9 #227.8 #169.3
PIX_UM = 0.8 #2.44 #1.74
source_sans = FontProperties(fname='C:\\Users\\ejerison\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')


def setup_plotting_style():
    """Set up consistent plotting style according to coding standards."""
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 5,
        'lines.linewidth': 0.5,
        'figure.figsize': (2.4, 2.0)
    })



def overlay_x_values_on_kymograph(image, x_values, color='r', linewidth=1):
    """
    Overlays vertical lines at specified x-values on a rotated kymograph image.
    
    Parameters
    ----------
    image : 2D numpy array
        The rotated kymograph image array (rows = y, columns = x).
    x_values : array-like
        The x (column) coordinates where RBC lines have been detected.
    color : str, optional
        Color of the overlaid lines. Default is 'r' (red).
    linewidth : float, optional
        Thickness of the overlaid lines. Default is 1.
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap='gray', aspect='auto')
    
    # Overlay vertical lines at each detected RBC position
    for x in x_values:
        plt.axhline(y=x, color=color, linewidth=linewidth)
    
    plt.title("Rotated Kymograph with Detected RBC Positions")
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.show()

def main(path_to_kymograph, counts_df):
    # -----------------------------
    # Step 1: Load the Kymograph, velocity, and estimated counts
    # -----------------------------
    # The kymograph is assumed to be a 2D grayscale image (rows=frames, cols=position)
    kymograph = io.imread(path_to_kymograph)  # or png, etc.
    # slice into bottom half of the kymograph
    kymograph = kymograph[kymograph.shape[0]//2:, :]
    kymograph = kymograph.astype(float)  # ensure floating point
    # get other data from counts_df based on the filename
    counts_row = counts_df[counts_df['Image_Path'] == os.path.basename(path_to_kymograph)]
    image_name = os.path.basename(path_to_kymograph)
    rbc_velocity_um_s = counts_row['Classified_Velocity'].values[0]
    rbc_count_est = counts_row['Counts'].values[0]
    # print(f"{image_name} RBC velocity: {rbc_velocity_um_s} um/s, RBC count: {rbc_count_est}")

    # -----------------------------
    # Step 2: Determine Rotation Angle
    # -----------------------------
    # Calculate the average velocity of RBCs (in pixels/frame)
    rbc_velocity = rbc_velocity_um_s / (PIX_UM * FPS)

    # Angle in radians:
    theta_radians = np.arctan(rbc_velocity)
    # Convert to degrees:
    theta_degrees = np.degrees(theta_radians)

    # Rotate image so RBC lines are vertical:
    # If RBCs tilt towards the right, you likely need a negative angle to correct.
    rotated = transform.rotate(kymograph, -theta_degrees, resize=True)

    # After rotation, RBC traces should be more vertical.

    # -----------------------------
    # Step 3: Average Pixels Along Each Row
    # -----------------------------
    # Now that RBC lines are vertical, we can compress each row into a single intensity value
    # by averaging across columns. This gives a 1D intensity profile.
    profile = rotated.mean(axis=1)  # average each row

    # 'profile' is now a 1D signal (intensity vs row index)
    # If RBCs are vertical lines, each cell should produce a local intensity dip or peak depending on contrast.

    # -----------------------------
    # Step 4: Optional Preprocessing (Convolution)
    # -----------------------------
    # Convolution with a small smoothing kernel can help reduce noise and sharpen edges.
    # For example, a simple Gaussian-like kernel:
    # kernel_size = 5
    # kernel = np.ones(kernel_size) / kernel_size
    # smoothed_profile = convolve(profile, kernel, mode='same')

    # Alternatively, you can use more sophisticated filters (like scipy.ndimage.gaussian_filter1d)
    # from scipy.ndimage import gaussian_filter1d
    # smoothed_profile = gaussian_filter1d(profile, sigma=2)

    # -----------------------------
    # Step 5: Threshold or Peak Detection with Adaptive Parameters
    # -----------------------------
    # If RBCs appear as darker lines (lower intensity), invert the profile:
    inverted = -profile

    # Calculate expected distance between peaks based on estimated count
    expected_distance = rotated.shape[0] / max(1, rbc_count_est)  # Avoid division by zero
    prominence_threshold = 0.05
    # # Adjust prominence threshold based on estimated count NOTE: Not really needed
    # if rbc_count_est > 5:  # Many RBCs
    #     prominence_threshold = 0.03  # Lower threshold for dense samples
    #     print(f"Using lower prominence threshold ({prominence_threshold}) for high density sample")
    # elif rbc_count_est < 5:  # Few RBCs
    #     prominence_threshold = 0.05  # Higher threshold for sparse samples
    #     print(f"Using higher prominence threshold ({prominence_threshold}) for low density sample")
    # else:  # Medium density
    #     prominence_threshold = 0.05  # Default threshold
    #     print(f"Using default prominence threshold ({prominence_threshold}) for medium density sample")

    # Initial peak detection with adaptive parameters
    peaks, properties = find_peaks(inverted, 
                                  prominence=prominence_threshold,
                                  distance=max(1, expected_distance * 0.7),  # Minimum expected spacing
                                  height=None)
    
    initial_count = len(peaks)
    print(f"Initial RBC count: {initial_count} (Estimated: {rbc_count_est})")
    
    # # -----------------------------
    # # Step 6: Validation and Correction #NOTE: Not really needed
    # # -----------------------------
    # # Compare detected count with estimated count
    # if rbc_count_est > 0:  # Only if we have a valid estimate
    #     count_ratio = initial_count / rbc_count_est
        
    #     # If detected count is significantly different from estimated count
    #     if count_ratio < 0.7 or count_ratio > 1.3:  # More than 30% difference
    #         print(f"Warning: Detected count ({initial_count}) differs significantly from estimated ({rbc_count_est})")
            
    #         # Try different parameters
    #         if initial_count < rbc_count_est:
    #             # We're missing peaks, reduce prominence threshold
    #             adjusted_prominence = prominence_threshold * 0.7
    #             print(f"Reducing prominence threshold to {adjusted_prominence} to detect more peaks")
    #             peaks, _ = find_peaks(inverted, 
    #                                  prominence=adjusted_prominence, 
    #                                  distance=max(1, expected_distance * 0.6))
    #         else:
    #             # Too many peaks, increase prominence threshold
    #             adjusted_prominence = prominence_threshold * 1.3
    #             print(f"Increasing prominence threshold to {adjusted_prominence} to detect fewer peaks")
    #             peaks, _ = find_peaks(inverted, 
    #                                  prominence=adjusted_prominence, 
    #                                  distance=max(1, expected_distance * 0.8))
            
    #         print(f"Adjusted RBC count: {len(peaks)}")
    
    # Final RBC count
    rbc_count = len(peaks)
    print(f"Final number of RBCs detected: {rbc_count}")

    # -----------------------------
    # Step 7: Visual Inspection
    # -----------------------------
    setup_plotting_style()

    plt.figure(figsize=(2.4,2))
    plt.plot(profile, label='Smoothed Profile')
    plt.plot(peaks, profile[peaks], 'rx', label='Detected RBCs')
    plt.xlabel('Row Index', fontproperties=source_sans)
    plt.ylabel('Intensity', fontproperties=source_sans)
    plt.title(f'Vertical Intensity Profile After Rotation (Detected: {rbc_count}, Estimated: {rbc_count_est})', fontproperties=source_sans)
    plt.legend(prop=source_sans)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Step 8: Overlay Detected RBCs on Kymograph
    # -----------------------------
    overlay_x_values_on_kymograph(rotated, peaks)
    
    return rbc_count

if __name__ == '__main__':
    counts_df = pd.read_csv('D:\\frog\\counted_kymos_CalFrog4.csv')
    for path in glob.glob('D:\\frog\\kymographs\\*.tiff'):
        main(path, counts_df)