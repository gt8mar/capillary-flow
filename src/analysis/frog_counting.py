import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, convolve

FPS = 130 #113.9 #227.8 #169.3
PIX_UM = 0.8 #2.44 #1.74

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

def main():
    # -----------------------------
    # Step 1: Load the Kymograph
    # -----------------------------
    # Replace with your actual kymograph image path
    # The kymograph is assumed to be a 2D grayscale image (rows=frames, cols=position)
    kymograph = io.imread('D:\\frog\\kymographs\\CalFrog4fps130Lankle_kymograph_0i.tiff')  # or png, etc.
    # slice into bottom half of the kymograph
    kymograph = kymograph[kymograph.shape[0]//2:, :]
    kymograph = kymograph.astype(float)  # ensure floating point

    # -----------------------------
    # Step 2: Determine Rotation Angle
    # -----------------------------
    # Calculate the average velocity of RBCs (in pixels/frame)
    rbc_velocity_um_s = 439.289416
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
    # Step 5: Threshold or Peak Detection
    # -----------------------------
    # If RBCs appear as darker lines (lower intensity), invert the profile:
    inverted = -profile

    # Find peaks in the inverted profile (which correspond to valleys in the original):
    peaks, properties = find_peaks(inverted, prominence=0.05, distance=1, height=None)
    # Adjust 'prominence' and 'distance' to ensure you correctly identify RBCs

    rbc_count = len(peaks)
    print("Number of RBCs detected:", rbc_count)

    # -----------------------------
    # Step 6: Visual Inspection
    # -----------------------------
    plt.figure(figsize=(10,5))
    plt.plot(profile, label='Smoothed Profile')
    plt.plot(peaks, profile[peaks], 'rx', label='Detected RBCs')
    plt.xlabel('Row Index')
    plt.ylabel('Intensity')
    plt.title('Vertical Intensity Profile After Rotation')
    plt.legend()
    plt.show()

    # -----------------------------
    # Step 7: Overlay Detected RBCs on Kymograph
    # -----------------------------
    overlay_x_values_on_kymograph(rotated, peaks)

if __name__ == '__main__':
    main()