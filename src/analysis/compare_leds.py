import os
import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
from matplotlib.font_manager import FontProperties

# Import paths from config
from src.config import PATHS, load_source_sans

source_sans = load_source_sans()

def compare_images(image1, image2, plot=True):
    """
    Compare two images using multiple metrics and optionally visualize the difference.
    
    Parameters:
    image1, image2 (numpy.ndarray): Input images to compare (should be same size)
    plot (bool): Whether to display comparison visualizations
    
    Returns:
    dict: Dictionary containing comparison metrics
    """
    # Load font using config helper function
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Calculate various comparison metrics
    mse = np.mean((image1 - image2) ** 2)
    rmse = np.sqrt(mse)
    
    # Normalize images for correlation calculation
    img1_norm = (image1 - np.mean(image1)) / np.std(image1)
    img2_norm = (image2 - np.mean(image2)) / np.std(image2)
    correlation = np.mean(img1_norm * img2_norm)
    
    # Calculate SSIM
    ssim_score = ssim(image1, image2)
    
    # Calculate difference image
    diff_image = np.abs(image1 - image2)
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0))
        axes[0, 0].imshow(image1, cmap='gray')
        if source_sans:
            axes[0, 0].set_title('Image 1', fontproperties=source_sans)
        else:
            axes[0, 0].set_title('Image 1')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(image2, cmap='gray')
        if source_sans:
            axes[0, 1].set_title('Image 2', fontproperties=source_sans)
        else:
            axes[0, 1].set_title('Image 2')
        axes[0, 1].axis('off')
        
        im = axes[1, 0].imshow(diff_image, cmap='hot')
        if source_sans:
            axes[1, 0].set_title('Difference Map', fontproperties=source_sans)
        else:
            axes[1, 0].set_title('Difference Map')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Intensity histogram
        axes[1, 1].hist(image1.ravel(), bins=50, alpha=0.5, label='Image 1')
        axes[1, 1].hist(image2.ravel(), bins=50, alpha=0.5, label='Image 2')
        if source_sans:
            axes[1, 1].set_title('Intensity Histograms', fontproperties=source_sans)
        else:
            axes[1, 1].set_title('Intensity Histograms')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Use path from config for saving
        results_path = os.path.join(PATHS['cap_flow'], 'results')
        snr_path = os.path.join(results_path, 'snr')
        # Ensure directory exists
        os.makedirs(snr_path, exist_ok=True)
        save_path = os.path.join(snr_path, 'image_comparison.png')
        plt.savefig(save_path, dpi=400)
        plt.close()
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'Correlation': correlation,
        'SSIM': ssim_score,
        'Max_Difference': np.max(diff_image),
        'Mean_Difference': np.mean(diff_image)
    }

def compare_line_profiles(profile1, profile2, plot=True):
    """
    Compare two line profiles and calculate relevant metrics.
    
    Parameters:
    profile1, profile2 (numpy.ndarray): 1D arrays containing line profile data
    plot (bool): Whether to display comparison plot
    
    Returns:
    dict: Dictionary containing comparison metrics
    """
    # Load font using config helper function
    source_sans = load_source_sans()
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    if len(profile1) != len(profile2):
        raise ValueError("Line profiles must have the same length")
    
    # Calculate basic statistics
    mse = np.mean((profile1 - profile2) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate Pearson correlation
    correlation, _ = pearsonr(profile1, profile2)
    
    # Calculate peak metrics
    peak1_idx = np.argmax(profile1)
    peak2_idx = np.argmax(profile2)
    peak_shift = abs(peak1_idx - peak2_idx)
    peak_intensity_diff = abs(np.max(profile1) - np.max(profile2))
    
    if plot:
        plt.figure(figsize=(4.8, 3.0))
        x = np.arange(len(profile1))
        plt.plot(x, profile1, label='Profile 1', alpha=0.7)
        plt.plot(x, profile2, label='Profile 2', alpha=0.7)
        plt.plot(x, np.abs(profile1 - profile2), label='Absolute Difference', 
                linestyle='--', alpha=0.5)
        
        if source_sans:
            plt.xlabel('Position', fontproperties=source_sans)
            plt.ylabel('Intensity', fontproperties=source_sans)
            plt.title('Line Profile Comparison', fontproperties=source_sans)
        else:
            plt.xlabel('Position')
            plt.ylabel('Intensity')
            plt.title('Line Profile Comparison')
            
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Use path from config for saving
        results_path = os.path.join(PATHS['cap_flow'], 'results')
        snr_path = os.path.join(results_path, 'snr')
        # Ensure directory exists
        os.makedirs(snr_path, exist_ok=True)
        save_path = os.path.join(snr_path, 'line_profile_comparison.png')
        plt.savefig(save_path, dpi=400)
        plt.close()
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'Correlation': correlation,
        'Peak_Shift': peak_shift,
        'Peak_Intensity_Difference': peak_intensity_diff,
        'Area_Difference': np.abs(np.sum(profile1) - np.sum(profile2))
    }

def calculate_image_metrics(img):
        # Contrast metrics
        contrast = np.std(img)
        dynamic_range = np.max(img) - np.min(img)
        
        # Calculate local contrast using gradient magnitude
        gx = np.gradient(img, axis=1)
        gy = np.gradient(img, axis=0)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        local_contrast = np.mean(gradient_magnitude)
        
        # Signal-to-noise ratio (SNR)
        # Assuming noise is the standard deviation in a relatively uniform region
        # Using the top 10% most uniform regions based on local variance
        kernel_size = 5
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(img, kernel_size)
        local_var = uniform_filter(img**2, kernel_size) - local_mean**2
        uniform_mask = local_var < np.percentile(local_var, 10)
        noise_std = np.std(img[uniform_mask])
        signal = np.mean(img)
        snr = signal / noise_std if noise_std != 0 else float('inf')
        
        return {
            'Contrast': contrast,
            'Dynamic_Range': dynamic_range,
            'Local_Contrast': local_contrast,
            'SNR': snr,
            'Mean_Intensity': np.mean(img),
            'Median_Intensity': np.median(img),
            'Std_Dev': np.std(img)
        }

def analyze_image_quality(image1, image2, names=('Image 1', 'Image 2'), plot=True):
    """
    Compare two images with focus on contrast and image quality metrics.
    
    Parameters:
    image1, image2 (numpy.ndarray): Input images to compare
    names (tuple): Names of the images for labeling
    plot (bool): Whether to display comparison visualizations
    
    Returns:
    dict: Dictionary containing quality metrics
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    metrics1 = calculate_image_metrics(image1)
    metrics2 = calculate_image_metrics(image2)
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Images with intensity distributions
        for ax, img, name, metrics in zip(axes[0], [image1, image2], names, [metrics1, metrics2]):
            im = ax.imshow(img, cmap='gray')
            ax.set_title(f'{name}\nContrast: {metrics["Contrast"]:.2f}\nSNR: {metrics["SNR"]:.2f}')
            plt.colorbar(im, ax=ax)
        
        # Intensity histograms
        axes[1, 0].hist(image1.ravel(), bins=50, alpha=0.7, label=names[0])
        axes[1, 0].hist(image2.ravel(), bins=50, alpha=0.7, label=names[1])
        axes[1, 0].set_title('Intensity Distributions')
        axes[1, 0].set_xlabel('Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Local contrast visualization
        gx1, gy1 = np.gradient(image1)
        gx2, gy2 = np.gradient(image2)
        grad_mag1 = np.sqrt(gx1**2 + gy1**2)
        grad_mag2 = np.sqrt(gx2**2 + gy2**2)
        
        combined_grad = np.abs(grad_mag1 - grad_mag2)
        im = axes[1, 1].imshow(combined_grad, cmap='hot')
        axes[1, 1].set_title('Local Contrast Difference Map')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        # plt.show()
        plt.close()
    
    return {
        'Image1_Metrics': metrics1,
        'Image2_Metrics': metrics2,
        'Contrast_Difference': metrics2['Contrast'] - metrics1['Contrast'],
        'SNR_Difference': metrics2['SNR'] - metrics1['SNR'],
        'Dynamic_Range_Difference': metrics2['Dynamic_Range'] - metrics1['Dynamic_Range']
    }

def old_method():
    pass
    # if plot:
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # # Profile comparison
        # x = np.arange(len(profile1))
        # ax1.plot(x, profile1, label=f'{names[0]}', alpha=0.7)
        # ax1.plot(x, profile2, label=f'{names[1]}', alpha=0.7)
        # ax1.set_title('Line Profile Comparison')
        # ax1.set_xlabel('Position')
        # ax1.set_ylabel('Intensity')
        # ax1.grid(True, alpha=0.3)
        # ax1.legend(loc = 'upper right')
        
        # # Add contrast and SNR annotations
        # for i, (name, metrics) in enumerate(zip(names, [metrics1, metrics2])):    
        #     ax1.text(0.02, 0.98 - i*0.15, 
        #             f'{name}:\nSNR: {metrics["SNR"]:.2f}',      # \nContrast: {metrics["Contrast"]:.2f}
        #             transform=ax1.transAxes,
        #             verticalalignment='top')
        
        # # Normalized profiles for shape comparison
        # norm_profile1 = (profile1 - np.min(profile1)) / (np.max(profile1) - np.min(profile1))
        # norm_profile2 = (profile2 - np.min(profile2)) / (np.max(profile2) - np.min(profile2))
        # ax2.plot(x, norm_profile1, label=f'{names[0]} (normalized)', alpha=0.7)
        # ax2.plot(x, norm_profile2, label=f'{names[1]} (normalized)', alpha=0.7)
        # ax2.set_title('Normalized Profiles')
        # ax2.set_xlabel('Position')
        # ax2.set_ylabel('Normalized Intensity')
        # ax2.grid(True, alpha=0.3)
        # ax2.legend(loc='upper right')
        
        # plt.tight_layout()
        # # plt.show()
        # plt.close()
    return 0

def analyze_line_profile_quality(profile1, profile2, names=('Two LEDs', 'One LED'), plot=True, window_size=15, polyorder=3):
    """
    Compare two line profiles with focus on signal quality and contrast.
    
    Parameters:
    profile1, profile2 (numpy.ndarray): 1D arrays containing line profile data
    names (tuple): Names of the profiles for labeling
    plot (bool): Whether to display comparison plot
    
    Returns:
    dict: Dictionary containing quality metrics
    """
    # Load font using config helper function
    source_sans = load_source_sans()
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

    if len(profile1) != len(profile2):
        raise ValueError("Line profiles must have the same length")
    
    def calculate_profile_metrics(profile):
        # Signal metrics
        peak_value = np.max(profile)
        background = np.percentile(profile, 10)  # Assuming bottom 10% is background
        contrast = peak_value - background
        
        # Calculate noise as std dev in the background region
        background_mask = profile < np.percentile(profile, 20)
        noise = np.std(profile[background_mask])
        snr = contrast / noise if noise != 0 else float('inf')
        
        # Calculate FWHM
        half_max = (peak_value + background) / 2
        above_half = profile >= half_max
        edges = np.where(np.diff(above_half))[0]
        fwhm = edges[-1] - edges[0] if len(edges) >= 2 else None
        
        return {
            'Peak_Value': peak_value,
            'Background': background,
            'Contrast': contrast,
            'SNR': snr,
            'FWHM': fwhm,
            'Mean': np.mean(profile),
            'Std_Dev': np.std(profile)
        }
    
    metrics1 = calculate_profile_metrics(profile1)
    metrics2 = calculate_profile_metrics(profile2)
    
    if plot:
        # Use standard figure size from coding standards
        fig, ax1 = plt.subplots(1, 1, figsize=(4.8, 3.0))
        
        # Profile comparison
        x = np.arange(len(profile1))
        ax1.plot(x, profile1, label=f'{names[0]}', alpha=0.7)
        ax1.plot(x, profile2, label=f'{names[1]}', alpha=0.7)
        
        # Apply font safely
        if source_sans:
            ax1.set_title('Line Profile Comparison', fontproperties=source_sans)
            ax1.set_xlabel('Position', fontproperties=source_sans)
            ax1.set_ylabel('Intensity', fontproperties=source_sans)
        else:
            ax1.set_title('Line Profile Comparison')
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Intensity')
            
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Add contrast and SNR annotations
        for i, (name, metrics) in enumerate(zip(names, [metrics1, metrics2])):    
            ax1.text(0.02, 0.98 - i*0.15, 
                f'{name}:\nSNR: {metrics["SNR"]:.2f}',      # \nContrast: {metrics["Contrast"]:.2f}
                transform=ax1.transAxes,
                verticalalignment='top')
        
        plt.tight_layout()
        
        # Use path from config for saving
        results_path = os.path.join(PATHS['cap_flow'], 'results')
        snr_path = os.path.join(results_path, 'snr')
        # Ensure directory exists
        os.makedirs(snr_path, exist_ok=True)
        save_path = os.path.join(snr_path, 'line_profile_comparisonsnr.png')
        plt.savefig(save_path, dpi=400)
        plt.close()
    
    return {
        f'{names[0]}_Metrics': metrics1,
        f'{names[1]}_Metrics': metrics2,
        'Contrast_Difference': metrics2['Contrast'] - metrics1['Contrast'],
        'SNR_Difference': metrics2['SNR'] - metrics1['SNR'],
        'FWHM_Difference': metrics2['FWHM'] - metrics1['FWHM'] if (metrics1['FWHM'] and metrics2['FWHM']) else None
    }

def normalize_and_enhance_profiles(profile1, profile2, names=('Two LEDs', 'One LED'), plot=True):
    """
    Normalize and enhance contrast of line profiles for better comparison.
    
    Parameters:
    profile1, profile2 (numpy.ndarray): 1D arrays containing line profile data
    names (tuple): Names of the profiles for labeling
    plot (bool): Whether to display and save the comparison plot
    
    Returns:
    tuple: Tuple containing normalized profiles (norm_profile1, norm_profile2)
    """
    # Load font using config helper function
    source_sans = load_source_sans()
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    if len(profile1) != len(profile2):
        raise ValueError("Line profiles must have the same length")
    
    # Min-max normalization
    norm_profile1 = (profile1 - np.min(profile1)) / (np.max(profile1) - np.min(profile1))
    norm_profile2 = (profile2 - np.min(profile2)) / (np.max(profile2) - np.min(profile2))
    
    # Enhance contrast (histogram equalization for 1D)
    def enhance_contrast_1d(profile):
        # Convert to integers for histogram binning (0-255 range)
        scaled = (profile * 255).astype(np.uint8)
        # Calculate histogram
        hist, bins = np.histogram(scaled, 256, [0, 256])
        # Calculate cumulative distribution function
        cdf = hist.cumsum()
        # Normalize CDF
        cdf_normalized = cdf * 255 / cdf[-1]
        # Use linear interpolation to map values
        enhanced = np.interp(scaled, np.arange(256), cdf_normalized)
        # Return to 0-1 range
        return enhanced / 255.0
    
    enhanced_profile1 = enhance_contrast_1d(norm_profile1)
    enhanced_profile2 = enhance_contrast_1d(norm_profile2)
    
    if plot:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8, 6.0))
        
        # Plot normalized profiles
        x = np.arange(len(profile1))
        ax1.plot(x, norm_profile1, label=f'{names[0]} (normalized)', alpha=0.7)
        ax1.plot(x, norm_profile2, label=f'{names[1]} (normalized)', alpha=0.7)
        
        # Apply font safely
        if source_sans:
            ax1.set_title('Normalized Line Profiles', fontproperties=source_sans)
            ax1.set_xlabel('Position', fontproperties=source_sans)
            ax1.set_ylabel('Normalized Intensity', fontproperties=source_sans)
        else:
            ax1.set_title('Normalized Line Profiles')
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Normalized Intensity')
        
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Plot enhanced profiles
        ax2.plot(x, enhanced_profile1, label=f'{names[0]} (enhanced)', alpha=0.7)
        ax2.plot(x, enhanced_profile2, label=f'{names[1]} (enhanced)', alpha=0.7)
        
        # Apply font safely
        if source_sans:
            ax2.set_title('Contrast Enhanced Line Profiles', fontproperties=source_sans)
            ax2.set_xlabel('Position', fontproperties=source_sans)
            ax2.set_ylabel('Enhanced Intensity', fontproperties=source_sans)
        else:
            ax2.set_title('Contrast Enhanced Line Profiles')
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Enhanced Intensity')
        
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Use path from config for saving
        results_path = os.path.join(PATHS['cap_flow'], 'results')
        snr_path = os.path.join(results_path, 'snr')
        # Ensure directory exists
        os.makedirs(snr_path, exist_ok=True)
        save_path = os.path.join(snr_path, 'normalized_enhanced_profiles.png')
        plt.savefig(save_path, dpi=400)
        plt.close()
    
    return norm_profile1, norm_profile2, enhanced_profile1, enhanced_profile2

def normalize_with_snr(profile1, profile2, names=('Two LEDs', 'One LED'), offset=0.2, plot=True):
    """
    Normalize line profiles, offset them for better visualization, and calculate SNR.
    
    Parameters:
    profile1, profile2 (numpy.ndarray): 1D arrays containing line profile data
    names (tuple): Names of the profiles for labeling
    offset (float): Vertical offset to apply between normalized profiles
    plot (bool): Whether to display and save the comparison plot
    
    Returns:
    tuple: Tuple containing normalized profiles (norm_profile1, norm_profile2)
    """
    # Load font using config helper function
    source_sans = load_source_sans()
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    if len(profile1) != len(profile2):
        raise ValueError("Line profiles must have the same length")
    
    # Min-max normalization
    norm_profile1 = (profile1 - np.min(profile1)) / (np.max(profile1) - np.min(profile1))
    norm_profile2 = (profile2 - np.min(profile2)) / (np.max(profile2) - np.min(profile2))
    
    # Apply offset to the second profile for better visualization
    norm_profile2_offset = norm_profile2 + offset
    
    # Calculate SNR for each profile using same method as in analyze_line_profile_quality
    def calculate_snr(profile):
        peak_value = np.max(profile)
        background = np.percentile(profile, 10)  # Assuming bottom 10% is background
        contrast = peak_value - background
        
        # Calculate noise as std dev in the background region
        background_mask = profile < np.percentile(profile, 20)
        noise = np.std(profile[background_mask])
        snr = contrast / noise if noise != 0 else float('inf')
        
        return snr, contrast, noise
    
    snr1, contrast1, noise1 = calculate_snr(profile1)
    snr2, contrast2, noise2 = calculate_snr(profile2)
    
    if plot:
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        
        x = np.arange(len(profile1))
        ax.plot(x, norm_profile1, label=f'{names[0]}', alpha=0.7)
        ax.plot(x, norm_profile2_offset, label=f'{names[1]} (offset)', alpha=0.7)
        
        # Apply font safely
        if source_sans:
            ax.set_title('Normalized Line Profiles with Offset', fontproperties=source_sans)
            ax.set_xlabel('Position', fontproperties=source_sans)
            ax.set_ylabel('Normalized Intensity', fontproperties=source_sans)
        else:
            ax.set_title('Normalized Line Profiles with Offset')
            ax.set_xlabel('Position')
            ax.set_ylabel('Normalized Intensity')
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Add SNR annotations
        for i, (name, snr, contrast) in enumerate(zip(names, [snr1, snr2], [contrast1, contrast2])):    
            ax.text(0.02, 0.98 - i*0.15, 
                f'{name}:\nSNR: {snr:.2f}\nContrast: {contrast:.2f}',
                transform=ax.transAxes,
                verticalalignment='top')
        
        plt.tight_layout()
        
        # Use path from config for saving
        results_path = os.path.join(PATHS['cap_flow'], 'results')
        snr_path = os.path.join(results_path, 'snr')
        # Ensure directory exists
        os.makedirs(snr_path, exist_ok=True)
        save_path = os.path.join(snr_path, 'normalized_profiles_with_snr.png')
        plt.savefig(save_path, dpi=400)
        plt.close()
    
    return {
        f'{names[0]}_Normalized': norm_profile1,
        f'{names[1]}_Normalized': norm_profile2,
        f'{names[0]}_SNR': snr1,
        f'{names[1]}_SNR': snr2,
        f'{names[0]}_Contrast': contrast1,
        f'{names[1]}_Contrast': contrast2,
        'SNR_Difference': snr2 - snr1,
        'Contrast_Difference': contrast2 - contrast1
    }

def calculate_snr_with_smoothing(profile, window_size=11, plot=False, name='Profile'):
    """
    Calculate SNR by smoothing the profile and measuring variations around the smoothed line.
    This is useful for absorptive features (dark) on a light background.
    
    Parameters:
    profile (numpy.ndarray): 1D array containing line profile data
    window_size (int): Size of the window for smoothing (must be odd)
    plot (bool): Whether to display and save a visualization of the smoothing and SNR
    name (str): Name of the profile for plotting
    
    Returns:
    dict: Dictionary containing SNR metrics
    """
    # Load font using config helper function
    source_sans = load_source_sans()
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Apply smoothing using a moving average
    from scipy.signal import savgol_filter
    
    # Smooth the profile
    smoothed_profile = savgol_filter(profile, window_size, 3)  # window size, polynomial order
    
    # plot smoothed profile
    plt.plot(smoothed_profile)
    # plt.show()
    plt.close()
    
    # Calculate residuals (noise)
    residuals = profile - smoothed_profile
    
    # Calculate signal and noise metrics
    signal_power = np.std(smoothed_profile)
    noise_power = np.std(residuals)
    
    # Calculate SNR
    snr = signal_power / noise_power if noise_power != 0 else float('inf')
    
    # Identify peaks (absorptive features are valleys in the profile)
    from scipy.signal import find_peaks
    
    # IMPORTANT CHANGE: Use smoothed profile for valley detection
    # Invert smoothed profile for valley detection
    inverted_smoothed = np.max(smoothed_profile) - smoothed_profile
    
    # Use a more locally-adaptive approach for valley detection
    # First, detect all potential valleys with a lower threshold
    threshold = np.percentile(inverted_smoothed, 60)  # More inclusive threshold
    min_prominence = np.std(inverted_smoothed) * 0.5  # Lower prominence for smoothed profile
    min_width = 3  # Minimum width for valleys
    
    # Find peaks with more relaxed criteria on the smoothed profile
    valleys, valley_properties = find_peaks(inverted_smoothed, 
                                 height=threshold,
                                 distance=int(window_size/3),  # Allow even closer peaks 
                                 prominence=min_prominence,
                                 width=min_width)  # Require minimum width
    
    # Use local regions of interest approach if we have specific knowledge of where peaks should be
    roi_regions = []
    
    # Check if profile length matches expected length for these specific regions
    if len(profile) > 750:
        # Define regions of interest based on domain knowledge
        roi_regions = [(300, 420), (600, 750)]
    
    # If we have defined ROIs, use a specialized approach for each ROI
    if roi_regions and len(valleys) > 0:
        roi_valleys = []
        
        for i, (start, end) in enumerate(roi_regions):
            # Keep valleys that fall within this ROI
            region_valleys = valleys[(valleys >= start) & (valleys <= end)]
            
            # If we found valleys in this region
            if len(region_valleys) > 0:
                # For first ROI (i==0), use a more robust approach to find the two valleys
                if i == 0:
                    # For the first ROI (300-420), use a more robust approach to find the two valleys
                    # Get the smoothed profile segment in this region with some padding
                    roi_start = max(0, start-20)
                    roi_end = min(len(smoothed_profile), end+20)
                    roi_profile = smoothed_profile[roi_start:roi_end]
                    roi_x = np.arange(roi_start, roi_end)
                    
                    # Find all local minima in this segment
                    from scipy.signal import argrelextrema
                    local_mins_idx = argrelextrema(roi_profile, np.less, order=5)[0]
                    
                    # If we found minima
                    if len(local_mins_idx) > 0:
                        # Convert to absolute indices
                        local_mins = roi_start + local_mins_idx
                        
                        # Keep only minima within the actual ROI
                        roi_mins = local_mins[(local_mins >= start) & (local_mins <= end)]
                        
                        # If we found minima in the ROI
                        if len(roi_mins) > 0:
                            # Calculate depths (how deep each valley is)
                            baseline = np.max(smoothed_profile[start:end])
                            depths = baseline - smoothed_profile[roi_mins]
                            
                            # If we have exactly one minimum, we need to find a second one
                            if len(roi_mins) == 1:
                                # Look for a second minimum that might not be as pronounced
                                # Use a smaller order parameter for more sensitive detection
                                detailed_mins_idx = argrelextrema(roi_profile, np.less, order=3)[0]
                                detailed_mins = roi_start + detailed_mins_idx
                                
                                # Filter to ROI and remove the already found minimum
                                filtered_mins = detailed_mins[(detailed_mins >= start) & 
                                                             (detailed_mins <= end) & 
                                                             (detailed_mins != roi_mins[0])]
                                
                                if len(filtered_mins) > 0:
                                    # Calculate the distance from the first minimum
                                    distances = np.abs(filtered_mins - roi_mins[0])
                                    
                                    # Prioritize minima that are:
                                    # 1. Far enough from the first minimum (at least 30 points)
                                    # 2. Deep enough (at least 20% as deep as the main minimum)
                                    far_enough = distances >= 30
                                    
                                    if np.any(far_enough):
                                        filtered_mins = filtered_mins[far_enough]
                                        # Among the far enough points, find the deepest one
                                        depths2 = baseline - smoothed_profile[filtered_mins]
                                        second_min = filtered_mins[np.argmax(depths2)]
                                        # Add both valleys
                                        roi_valleys.extend([roi_mins[0], second_min])
                                    else:
                                        # If no point is far enough, just use the best we have
                                        second_min = filtered_mins[np.argmax(distances)]
                                        roi_valleys.extend([roi_mins[0], second_min])
                                else:
                                    # If we couldn't find a second minimum, just use the one we have
                                    roi_valleys.append(roi_mins[0])
                            elif len(roi_mins) >= 2:
                                # We have at least two minima, select the two deepest ones
                                sorted_indices = np.argsort(depths)[::-1]  # Descending order
                                
                                # But also ensure they're sufficiently separated (at least 30 points)
                                candidates = roi_mins[sorted_indices]
                                
                                # Try to find two valleys that are at least 30 points apart
                                selected = [candidates[0]]
                                
                                for idx in candidates[1:]:
                                    if np.all(np.abs(np.array(selected) - idx) >= 30):
                                        selected.append(idx)
                                        if len(selected) == 2:
                                            break
                                
                                # If we couldn't find two well-separated valleys, take the deepest
                                # and the second one that's farthest from it
                                if len(selected) < 2 and len(candidates) > 1:
                                    distances = np.abs(candidates[1:] - candidates[0])
                                    second_valley = candidates[1:][np.argmax(distances)]
                                    selected = [candidates[0], second_valley]
                                
                                roi_valleys.extend(selected[:2])  # Take up to 2 valleys
                        else:
                            # If no minima in ROI, check if any are close to the boundary
                            nearby = local_mins[(local_mins >= start-10) & (local_mins <= end+10)]
                            if len(nearby) > 0:
                                roi_valleys.extend(nearby[:2])  # Take up to 2 nearby valleys
                    # In case no valleys were found with the above method
                    if len(roi_valleys) == 0 and len(region_valleys) > 0:
                        roi_valleys.extend(region_valleys[:2])  # Fall back to original method
                else:
                    # For other ROIs, get prominences for these valleys
                    if len(region_valleys) > 1:
                        region_prominences = [valley_properties['prominences'][list(valleys).index(v)] for v in region_valleys]
                        
                        # Sort valleys by prominence
                        sorted_indices = np.argsort(region_prominences)[::-1]  # Descending order
                        
                        # Keep up to 2 most prominent valleys in each region (for W shape)
                        top_valleys = region_valleys[sorted_indices[:min(2, len(region_valleys))]]
                        roi_valleys.extend(top_valleys)
                    else:
                        # If only one valley, just add it
                        roi_valleys.extend(region_valleys)
        
        # If we found valleys in the ROIs, use them, otherwise keep the original valleys
        if len(roi_valleys) > 0:
            valleys = np.array(sorted(roi_valleys))
    
    # If we don't have specific ROIs or didn't find valleys in them,
    # and if we still have too many valleys, use clustering to find groups
    if len(roi_regions) == 0 and len(valleys) > 4:
        from sklearn.cluster import DBSCAN
        
        # Reshape for clustering
        X = valleys.reshape(-1, 1)
        
        # Use DBSCAN to cluster close valleys
        clustering = DBSCAN(eps=window_size, min_samples=1).fit(X)
        labels = clustering.labels_
        
        # Find the center of each cluster
        clustered_valleys = []
        for label in np.unique(labels):
            cluster_points = X[labels == label].flatten()
            
            # For each cluster, keep the point with highest prominence
            prominences = [valley_properties['prominences'][list(valleys).index(v)] for v in cluster_points]
            best_idx = np.argmax(prominences)
            clustered_valleys.append(cluster_points[best_idx])
        
        valleys = np.array(sorted(clustered_valleys))
    
    # Plot valleys on the smoothed profile for better visualization
    if plot:
        plt.figure(figsize=(4.8, 3.0))
        x = np.arange(len(profile))
        
        # Plot both original and smoothed profiles
        plt.plot(x, profile, label='Original Profile', alpha=0.7)
        plt.plot(x, smoothed_profile, label='Smoothed Profile', linewidth=1.5, alpha=0.8)
        
        # Mark valleys with different markers for clarity
        if len(valleys) > 0:
            plt.plot(valleys, smoothed_profile[valleys], 'ro', markersize=6, label=f'Detected Valleys ({len(valleys)})')
            
            # Add valley indices as text labels
            for i, idx in enumerate(valleys):
                plt.text(idx, smoothed_profile[idx] - 2, f'v{i+1}', fontsize=8, 
                         ha='center', va='top', color='darkred')
        
        # Mark ROI regions if defined
        for i, (start, end) in enumerate(roi_regions):
            if start < len(profile) and end < len(profile):
                plt.axvspan(start, end, alpha=0.2, color='gray', label=f'ROI {i+1}' if i==0 else None)
        
        plt.title(f'Valley Detection: {name}')
        plt.xlabel('Position')
        plt.ylabel('Intensity')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Use path from config for saving
        results_path = os.path.join(PATHS['cap_flow'], 'results')
        snr_path = os.path.join(results_path, 'snr')
        # Ensure directory exists
        os.makedirs(snr_path, exist_ok=True)
        save_path = os.path.join(snr_path, f'detected_valleys_{name.replace(" ", "_").lower()}.png')
        plt.savefig(save_path, dpi=400)
        # plt.show()
        plt.close()
    
    # Calculate peak-to-noise ratio for each detected valley
    peak_to_noise_ratios = []
    valley_depths = []
    valley_widths = []
    valley_areas = []
    
    # Calculate overall baseline for the profile (could be max or a high percentile)
    overall_baseline = np.percentile(smoothed_profile, 95)
    
    for i, valley_idx in enumerate(valleys):
        # Analyze each valley more thoroughly
        
        # Find local region around the valley (±20 points or until next valley)
        valley_start = valley_idx
        valley_end = valley_idx
        
        # Expand left until we find a local maximum or reach the beginning
        for j in range(valley_idx-1, max(0, valley_idx-50), -1):
            if j < 0 or j >= len(smoothed_profile):
                break
            if smoothed_profile[j] <= smoothed_profile[valley_start]:
                valley_start = j
            else:
                # Stop if we found a point higher than the previous
                if smoothed_profile[j] > smoothed_profile[j+1]:
                    break
        
        # Expand right until we find a local maximum or reach the end
        for j in range(valley_idx+1, min(len(smoothed_profile), valley_idx+50)):
            if j < 0 or j >= len(smoothed_profile):
                break
            if smoothed_profile[j] <= smoothed_profile[valley_end]:
                valley_end = j
            else:
                # Stop if we found a point higher than the previous
                if smoothed_profile[j] > smoothed_profile[j-1]:
                    break
        
        # Make sure we don't overlap with other valleys' regions
        if i > 0:
            prev_valley = valleys[i-1]
            valley_start = max(valley_start, (prev_valley + valley_idx) // 2)
        if i < len(valleys) - 1:
            next_valley = valleys[i+1]
            valley_end = min(valley_end, (next_valley + valley_idx) // 2)
        
        # Expand a bit more to capture the local baseline
        extended_start = max(0, valley_start - 10)
        extended_end = min(len(smoothed_profile), valley_end + 10)
        
        # Calculate local baseline as the maximum in the extended region
        local_region = smoothed_profile[extended_start:extended_end]
        local_baseline = np.max(local_region)
        
        # Calculate valley properties
        valley_depth = local_baseline - smoothed_profile[valley_idx]
        
        # Estimate FWHM (Full Width at Half Maximum)
        half_depth = local_baseline - valley_depth/2
        half_width_indices = np.where(smoothed_profile[valley_start:valley_end] <= half_depth)[0]
        valley_width = len(half_width_indices) if len(half_width_indices) > 0 else 1
        
        # Calculate approximate area of the valley
        valley_area = np.sum(local_baseline - smoothed_profile[valley_start:valley_end])
        
        # Record measurements
        valley_depths.append(valley_depth)
        valley_widths.append(valley_width)
        valley_areas.append(valley_area)
        
        # Calculate SNR with respect to noise
        peak_to_noise = valley_depth / noise_power if noise_power != 0 else float('inf')
        peak_to_noise_ratios.append(peak_to_noise)
    
    # Calculate average metrics
    avg_peak_to_noise = np.mean(peak_to_noise_ratios) if peak_to_noise_ratios else 0
    avg_valley_depth = np.mean(valley_depths) if valley_depths else 0
    
    # Calculate a feature contrast score as the ratio of average valley depth to average profile level
    avg_profile = np.mean(smoothed_profile)
    feature_contrast = avg_valley_depth / avg_profile if avg_profile != 0 else 0
    
    # Calculate a valley quality score that combines depth, width and noise level
    valley_quality_scores = [depth * width / noise_power for depth, width in zip(valley_depths, valley_widths)]
    avg_valley_quality = np.mean(valley_quality_scores) if valley_quality_scores else 0
    
    if plot:
        plt.rcParams.update({
            'pdf.fonttype': 42, 'ps.fonttype': 42,
            'font.size': 7, 'axes.labelsize': 7,
            'xtick.labelsize': 6, 'ytick.labelsize': 6,
            'legend.fontsize': 5, 'lines.linewidth': 0.5
        })
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8, 6.0))
        
        # Plot original and smoothed profiles
        x = np.arange(len(profile))
        ax1.plot(x, profile, label='Original Profile', alpha=0.7)
        ax1.plot(x, smoothed_profile, label='Smoothed Profile', alpha=0.7, linewidth=1.5)
        
        # Mark valleys on the plot with SNR values
        for i, valley_idx in enumerate(valleys):
            ax1.plot(valley_idx, profile[valley_idx], 'ro', markersize=4)
            ax1.text(valley_idx, profile[valley_idx] - 5, f'SNR: {peak_to_noise_ratios[i]:.1f}', 
                     fontsize=6, ha='center', color='darkred')
            
            # Highlight the valley region
            valley_start = max(0, valley_idx - valley_widths[i])
            valley_end = min(len(profile), valley_idx + valley_widths[i])
            ax1.axvspan(valley_start, valley_end, alpha=0.2, color='red')
        
        # Apply font safely
        if source_sans:
            ax1.set_title(f'{name}: Profile with Valley SNR Analysis', fontproperties=source_sans)
            ax1.set_xlabel('Position', fontproperties=source_sans)
            ax1.set_ylabel('Intensity', fontproperties=source_sans)
        else:
            ax1.set_title(f'{name}: Profile with Valley SNR Analysis')
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Intensity')
        
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Plot residuals
        ax2.plot(x, residuals, label='Residuals (Noise)', alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax2.axhline(y=noise_power, color='g', linestyle='--', alpha=0.5, label=f'Noise σ: {noise_power:.2f}')
        ax2.axhline(y=-noise_power, color='g', linestyle='--', alpha=0.5)
        
        # Mark valleys on residuals plot
        for valley_idx in valleys:
            ax2.axvline(x=valley_idx, color='r', linestyle=':', alpha=0.5)
        
        # Apply font safely
        if source_sans:
            ax2.set_title(f'Noise Analysis (Avg Valley SNR: {avg_peak_to_noise:.2f})', fontproperties=source_sans)
            ax2.set_xlabel('Position', fontproperties=source_sans)
            ax2.set_ylabel('Intensity Difference', fontproperties=source_sans)
        else:
            ax2.set_title(f'Noise Analysis (Avg Valley SNR: {avg_peak_to_noise:.2f})')
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Intensity Difference')
        
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Use path from config for saving
        results_path = os.path.join(PATHS['cap_flow'], 'results')
        snr_path = os.path.join(results_path, 'snr')
        # Ensure directory exists
        os.makedirs(snr_path, exist_ok=True)
        save_path = os.path.join(snr_path, f'valley_snr_analysis_{name.replace(" ", "_").lower()}.png')
        plt.savefig(save_path, dpi=400)
        # plt.show()
        plt.close()
    
    return {
        'Overall_SNR': snr,
        'Signal_Power': signal_power,
        'Noise_Power': noise_power,
        'Detected_Valleys': len(valleys),
        'Avg_Valley_SNR': avg_peak_to_noise,
        'Valley_SNR_Values': peak_to_noise_ratios,
        'Valley_Depths': valley_depths,
        'Valley_Widths': valley_widths,
        'Valley_Areas': valley_areas,
        'Feature_Contrast': feature_contrast,
        'Valley_Quality_Score': avg_valley_quality,
        'Valley_Indices': valleys.tolist()
    }

def calculate_valley_snr(profile1, profile2, window_size=15, names=('Two LEDs', 'One LED')):
    """
    Calculate SNR based on valley depths divided by noise, with separate plots for each profile.
    Specifically designed to find 4 valleys in the profiles and calculate their depths and SNR.
    
    Parameters:
    profile1, profile2 (numpy.ndarray): 1D arrays containing line profile data
    window_size (int): Size of the window for smoothing (must be odd)
    names (tuple): Names of the profiles for labeling
    
    Returns:
    tuple: Dictionaries containing SNR metrics for each profile
    """
    # Load font using config helper function
    source_sans = load_source_sans()
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Function to process a single profile
    def process_profile(profile, name):
        # Apply smoothing using Savitzky-Golay filter
        from scipy.signal import savgol_filter
        smoothed_profile = savgol_filter(profile, window_size, 3)  # window size, polynomial order
        
        # Calculate residuals (noise)
        residuals = profile - smoothed_profile
        noise_power = np.std(residuals)
        
        # Find valleys in the smoothed profile
        from scipy.signal import find_peaks
        
        # Invert smoothed profile for valley detection
        inverted_smoothed = np.max(smoothed_profile) - smoothed_profile
        
        # Define regions of interest based on looking at the profile
        # For One LED profile, we're interested in specific regions
        roi_regions = []
        if name == 'One LED':
            # Define specific regions to look for valleys based on visual inspection
            roi_regions = [(283, 380), (363, 420), (591, 657), (657, 750)] 
        elif name == 'Two LEDs':
            # Define regions for the Two LEDs profile
            roi_regions = [(283, 380), (363, 420), (591, 657), (657, 750)] 
        
        # Set parameters for valley detection
        prominence = np.std(inverted_smoothed) * 0.5  # TODO Lower prominence to catch more valleys
        width = 3  # Minimum width for valleys
        distance = 20  # Reduced distance to allow closer valleys
        
        # Find peaks in inverted profile (valleys in original)
        valleys, valley_properties = find_peaks(inverted_smoothed, 
                                    prominence=prominence,
                                    width=width,
                                    distance=distance)
        
        # Use regions of interest to refine valley detection
        if roi_regions:
            roi_valleys = []
            
            # Find valleys in each ROI
            for start, end in roi_regions:
                # Ensure we're within valid array bounds
                start = max(0, min(start, len(smoothed_profile)-1))
                end = max(0, min(end, len(smoothed_profile)-1))
                
                # Get segment of inverted profile in this ROI
                roi_segment = inverted_smoothed[start:end]
                
                # Find local peaks within this segment
                # Use lower prominence for focused detection
                local_prominence = np.std(roi_segment) * 0.3
                local_peaks, _ = find_peaks(roi_segment,
                                          prominence=local_prominence,
                                          width=width)
                
                # Convert local indices to global
                local_peaks = local_peaks + start
                
                # If we found peaks, take the most prominent one
                if len(local_peaks) > 0:
                    # Calculate prominences for these peaks
                    local_prominences = [inverted_smoothed[i] for i in local_peaks]
                    # Take the most prominent peak in this region
                    best_peak = local_peaks[np.argmax(local_prominences)]
                    roi_valleys.append(best_peak)
            
            # Use ROI valleys if we found any, otherwise keep the original valleys
            if roi_valleys:
                valleys = np.array(sorted(roi_valleys))
        
        # If we found more or fewer than expected valleys, adjust accordingly
        if len(valleys) != 4:
            print(f"Found {len(valleys)} valleys in {name}, expected 4.")
            
            # If we found more valleys, keep the 4 most prominent
            if len(valleys) > 4:
                prominences = valley_properties['prominences']
                idx = np.argsort(prominences)[::-1]  # Descending order
                valleys = valleys[idx[:4]]  # Keep top 4
            
            # If we found fewer, we'll work with what we have
        
        # Calculate valley maximums (local baseline for each valley)
        valley_maxs = []
        valley_depths = []
        valley_baseline_indices = []
        
        for i, valley_idx in enumerate(valleys):
            # Look for local maximum before and after the valley
            # Define search regions (±30 points or until next/previous valley)
            left_limit = max(0, valley_idx - 30)
            right_limit = min(len(smoothed_profile) - 1, valley_idx + 30)
            
            # Adjust limits based on neighboring valleys
            if i > 0:
                prev_valley = valleys[i-1]
                left_limit = max(left_limit, (prev_valley + valley_idx) // 2)
            if i < len(valleys) - 1:
                next_valley = valleys[i+1]
                right_limit = min(right_limit, (next_valley + valley_idx) // 2)
            
            # Get profile segment around valley
            left_segment = smoothed_profile[left_limit:valley_idx]
            right_segment = smoothed_profile[valley_idx+1:right_limit+1]
            
            # Find local maximums (baseline) on left and right
            if len(left_segment) > 0:
                left_max_idx = left_limit + np.argmax(left_segment)
            else:
                left_max_idx = left_limit
                
            if len(right_segment) > 0:
                right_max_idx = valley_idx + 1 + np.argmax(right_segment)
            else:
                right_max_idx = right_limit
            
            # Use higher of the two maximums as baseline
            left_max = smoothed_profile[left_max_idx]
            right_max = smoothed_profile[right_max_idx]
            
            if left_max >= right_max:
                baseline_idx = left_max_idx
                baseline = left_max
            else:
                baseline_idx = right_max_idx
                baseline = right_max
            
            # Calculate valley depth
            valley_depth = baseline - smoothed_profile[valley_idx]
            
            valley_maxs.append(baseline)
            valley_depths.append(valley_depth)
            valley_baseline_indices.append(baseline_idx)
        
        # Calculate SNR for each valley
        valley_snrs = [depth / noise_power for depth in valley_depths]
        
        # Mean values
        mean_depth = np.mean(valley_depths) if valley_depths else 0
        mean_snr = np.mean(valley_snrs) if valley_snrs else 0
        
        # Create plot
        plt.figure(figsize=(4.8, 3.0))
        x = np.arange(len(profile))
        
        # Plot original and smoothed profiles
        plt.plot(x, profile, label='Original Profile', alpha=0.7, color='lightgray')
        plt.plot(x, smoothed_profile, label='Smoothed Profile', linewidth=1.5, color='blue')
        
        # Mark valleys
        plt.plot(valleys, smoothed_profile[valleys], 'ro', markersize=5, label='Valley Minimums')
        
        # Mark baseline points for each valley
        plt.plot(valley_baseline_indices, [smoothed_profile[idx] for idx in valley_baseline_indices], 
                'go', markersize=5, label='Valley Baselines')
        
        # Draw lines showing valley depths
        for i, (valley_idx, baseline_idx) in enumerate(zip(valleys, valley_baseline_indices)):
            plt.plot([valley_idx, valley_idx], 
                    [smoothed_profile[valley_idx], smoothed_profile[baseline_idx]], 
                    'r--', linewidth=0.8)
            
            # # Add text showing SNR
            # plt.text(valley_idx, smoothed_profile[valley_idx] - 2, 
            #         f"SNR: {valley_snrs[i]:.1f}", fontsize=10, ha='center', 
            #         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Highlight ROI regions
        for i, (start, end) in enumerate(roi_regions):
            if start < len(profile) and end < len(profile):
                plt.axvspan(start, end, alpha=0.1, color='gray')
        
        # Apply title and labels
        if source_sans:
            plt.title(f'{name} Profile: Valley Analysis', fontproperties=source_sans)
            plt.xlabel('Position', fontproperties=source_sans)
            plt.ylabel('Intensity', fontproperties=source_sans)
        else:
            plt.title(f'{name} Profile: Valley Analysis')
            plt.xlabel('Position')
            plt.ylabel('Intensity')
        
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Save plot
        results_path = os.path.join(PATHS['cap_flow'], 'results')
        snr_path = os.path.join(results_path, 'snr')
        os.makedirs(snr_path, exist_ok=True)
        save_path = os.path.join(snr_path, f'valley_analysis_{name.replace(" ", "_").lower()}.png')
        plt.savefig(save_path, dpi=400)
        plt.close()
        
        # Print results
        print(f"\n{name} Analysis Results:")
        print(f"Number of valleys detected: {len(valleys)}")
        print(f"Noise level: {noise_power:.4f}")
        print(f"Valley details:")
        for i, (idx, depth, snr) in enumerate(zip(valleys, valley_depths, valley_snrs)):
            print(f"  Valley {i+1} (at position {idx}):")
            print(f"    Depth: {depth:.4f}")
            print(f"    SNR: {snr:.2f}")
        print(f"Mean valley depth: {mean_depth:.4f}")
        print(f"Mean valley SNR: {mean_snr:.2f}")
        
        return {
            'Valleys': valleys.tolist(),
            'Valley_Depths': valley_depths,
            'Valley_SNRs': valley_snrs,
            'Noise_Power': noise_power,
            'Mean_Depth': mean_depth,
            'Mean_SNR': mean_snr,
            'Baseline_Indices': valley_baseline_indices,
            'Smoothed_Profile': smoothed_profile
        }
    
    # Process each profile
    results1 = process_profile(profile1, names[0])
    results2 = process_profile(profile2, names[1])
    
    return results1, results2

def flatten_background_and_enhance(profile, window_size=51, plot=False, name='Profile'):
    """
    Flatten the background of a profile and enhance contrast for better valley detection and SNR calculation.
    This is especially useful when dealing with profiles of different brightness levels.
    
    Parameters:
    profile (numpy.ndarray): 1D array containing line profile data
    window_size (int): Size of the window for background estimation (must be odd and larger than valleys)
    plot (bool): Whether to display and save a visualization of the processing steps
    name (str): Name of the profile for plotting
    
    Returns:
    numpy.ndarray: Background-flattened and contrast-enhanced profile
    """
    # Load font using config helper function
    source_sans = load_source_sans()
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Step 1: Estimate background using a large window rolling maximum filter
    from scipy.ndimage import maximum_filter1d
    background = maximum_filter1d(profile, size=window_size)
    
    # Step 2: Flatten by dividing by the background (normalization)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    flattened = profile / (background + epsilon)
    
    # Step 3: Apply contrast enhancement
    # Scale to 0-1 range
    normalized = (flattened - np.min(flattened)) / (np.max(flattened) - np.min(flattened) + epsilon)
    
    # Apply contrast stretching/enhancement
    # Use a gamma correction to enhance dark features
    gamma = 0.7  # Gamma < 1 enhances dark features (valleys)
    enhanced = np.power(normalized, gamma)
    
    # Step 4: Invert so valleys become peaks (optional, depends on next steps)
    # inverted = 1 - enhanced
    
    if plot:
        plt.rcParams.update({
            'pdf.fonttype': 42, 'ps.fonttype': 42,
            'font.size': 7, 'axes.labelsize': 7,
            'xtick.labelsize': 6, 'ytick.labelsize': 6,
            'legend.fontsize': 5, 'lines.linewidth': 0.5
        })
        
        fig, axes = plt.subplots(3, 1, figsize=(4.8, 9.0))
        x = np.arange(len(profile))
        
        # Original vs Background
        axes[0].plot(x, profile, label='Original Profile', alpha=0.7)
        axes[0].plot(x, background, label='Estimated Background', alpha=0.7, linewidth=1.5)
        
        # Apply font safely
        if source_sans:
            axes[0].set_title(f'{name}: Original and Background', fontproperties=source_sans)
            axes[0].set_xlabel('Position', fontproperties=source_sans)
            axes[0].set_ylabel('Intensity', fontproperties=source_sans)
        else:
            axes[0].set_title(f'{name}: Original and Background')
            axes[0].set_xlabel('Position')
            axes[0].set_ylabel('Intensity')
        
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best')
        
        # Flattened
        axes[1].plot(x, flattened, label='Background Flattened', alpha=0.7)
        
        # Apply font safely
        if source_sans:
            axes[1].set_title(f'{name}: Background Flattened', fontproperties=source_sans)
            axes[1].set_xlabel('Position', fontproperties=source_sans)
            axes[1].set_ylabel('Normalized Intensity', fontproperties=source_sans)
        else:
            axes[1].set_title(f'{name}: Background Flattened')
            axes[1].set_xlabel('Position')
            axes[1].set_ylabel('Normalized Intensity')
        
        axes[1].grid(True, alpha=0.3)
        
        # Enhanced
        axes[2].plot(x, enhanced, label='Contrast Enhanced', alpha=0.7)
        
        # Apply font safely
        if source_sans:
            axes[2].set_title(f'{name}: Contrast Enhanced (γ={gamma})', fontproperties=source_sans)
            axes[2].set_xlabel('Position', fontproperties=source_sans)
            axes[2].set_ylabel('Enhanced Intensity', fontproperties=source_sans)
        else:
            axes[2].set_title(f'{name}: Contrast Enhanced (γ={gamma})')
            axes[2].set_xlabel('Position')
            axes[2].set_ylabel('Enhanced Intensity')
        
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Use path from config for saving
        results_path = os.path.join(PATHS['cap_flow'], 'results')
        snr_path = os.path.join(results_path, 'snr')
        # Ensure directory exists
        os.makedirs(snr_path, exist_ok=True)
        save_path = os.path.join(snr_path, f'background_flattened_{name.replace(" ", "_").lower()}.png')
        plt.savefig(save_path, dpi=400)
        # plt.show()
        plt.close()
    
    return enhanced

def compare_flattened_profiles(profile1, profile2, names=('Two LEDs', 'One LED'), window_size=51, plot=True):
    """
    Compare two profiles after background flattening and contrast enhancement.
    
    Parameters:
    profile1, profile2 (numpy.ndarray): 1D arrays containing line profile data
    names (tuple): Names of the profiles for labeling
    window_size (int): Size of the window for background estimation
    plot (bool): Whether to display and save the comparison
    
    Returns:
    tuple: (flattened_profile1, flattened_profile2)
    """
    # Load font using config helper function
    source_sans = load_source_sans()
    
    # Process both profiles
    enhanced1 = flatten_background_and_enhance(profile1, window_size=window_size, plot=False, name=names[0])
    enhanced2 = flatten_background_and_enhance(profile2, window_size=window_size, plot=False, name=names[1])
    
    if plot:
        plt.rcParams.update({
            'pdf.fonttype': 42, 'ps.fonttype': 42,
            'font.size': 7, 'axes.labelsize': 7,
            'xtick.labelsize': 6, 'ytick.labelsize': 6,
            'legend.fontsize': 5, 'lines.linewidth': 0.5
        })
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8, 6.0))
        
        # Original profiles
        x1 = np.arange(len(profile1))
        x2 = np.arange(len(profile2))
        ax1.plot(x1, profile1, label=f'{names[0]} Original', alpha=0.7)
        ax1.plot(x2, profile2, label=f'{names[1]} Original', alpha=0.7)
        
        # Apply font safely
        if source_sans:
            ax1.set_title('Original Profiles Comparison', fontproperties=source_sans)
            ax1.set_xlabel('Position', fontproperties=source_sans)
            ax1.set_ylabel('Intensity', fontproperties=source_sans)
        else:
            ax1.set_title('Original Profiles Comparison')
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Intensity')
        
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Enhanced profiles
        ax2.plot(x1, enhanced1, label=f'{names[0]} Enhanced', alpha=0.7)
        ax2.plot(x2, enhanced2, label=f'{names[1]} Enhanced', alpha=0.7)
        
        # Apply font safely
        if source_sans:
            ax2.set_title('Background-Flattened and Enhanced Profiles', fontproperties=source_sans)
            ax2.set_xlabel('Position', fontproperties=source_sans)
            ax2.set_ylabel('Enhanced Intensity', fontproperties=source_sans)
        else:
            ax2.set_title('Background-Flattened and Enhanced Profiles')
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Enhanced Intensity')
        
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        plt.tight_layout()
        
        # Use path from config for saving
        results_path = os.path.join(PATHS['cap_flow'], 'results')
        snr_path = os.path.join(results_path, 'snr')
        # Ensure directory exists
        os.makedirs(snr_path, exist_ok=True)
        save_path = os.path.join(snr_path, 'flattened_profiles_comparison.png')
        plt.savefig(save_path, dpi=400)
        # plt.show()
        plt.close()
    
    return enhanced1, enhanced2

def main():
    # Use paths from config
    cap_flow_path = PATHS['cap_flow']
    user_path = os.path.dirname(cap_flow_path)
    image_folder = os.path.join(user_path, 'Desktop', 'data', 'calibration', '241213_led_sides')
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
    # print(image_paths)
    # print(image_paths[0])
    # print(image_paths[1])
    right_image_path = os.path.join(image_folder, 'rightpng.png')
    both_image_path = os.path.join(image_folder, 'bothpng.png')
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    both_image = cv2.imread(both_image_path, cv2.IMREAD_GRAYSCALE)
    # image1 = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    # image2 = cv2.imread(image_paths[1], cv2.IMREAD_GRAYSCALE)
    # image2_translated = cv2.imread(image_paths[2], cv2.IMREAD_GRAYSCALE)

    # # Compare images
    # results = compare_images(image1, image2)
    # print(results)

    # # Compare image metrics
    # quality_results = analyze_image_quality(image1, image2)
    # print(quality_results)

    # Load line profiles with left right offset
    profile1 = both_image[574, 74:-74]
    profile2 = right_image[634, 0:-148]

    # Calculate SNR using valley method - NEW ANALYSIS
    print("\n========== VALLEY-BASED SNR ANALYSIS ==========")
    valley_results1, valley_results2 = calculate_valley_snr(profile1, profile2, window_size=15)

    # Print results for valley analysis
    print(valley_results1)
    print(valley_results2)



    # #-----------------------------------------------------------------------------------------
    # # Old ANALYSES BELOW
    # #-----------------------------------------------------------------------------------------    
    
    # # Compare line profiles - ORIGINAL ANALYSES BELOW
    # profile_results = compare_line_profiles(profile1, profile2)
    # print(profile_results)

    # # Analyze line profile quality
    # profile_quality = analyze_line_profile_quality(profile1, profile2)
    # print(profile_quality)
    
    # #--------------------------------------------------------------------
    # # Computational enhancement below
    # #--------------------------------------------------------------------

    # # Normalize and enhance profiles
    # normalized_profiles = normalize_and_enhance_profiles(profile1, profile2)
    # print("Normalized and enhanced profiles created and saved")
    
    # # Normalize with SNR
    # normalized_snr = normalize_with_snr(profile1, profile2)
    # print("Normalized profiles with SNR created and saved")
    # print(normalized_snr)
    
    # # Calculate SNR with smoothing on original profiles
    # print("\n========== SNR ANALYSIS ON ORIGINAL PROFILES ==========")
    # smoothed_snr1 = calculate_snr_with_smoothing(profile1, window_size=15, plot=True, name='Two LEDs')
    # smoothed_snr2 = calculate_snr_with_smoothing(profile2, window_size=15, plot=True, name='One LED')
    
    # # Print detailed SNR results for original profiles
    # print("\n================ DETAILED SNR RESULTS ================")
    # print(f"Number of valleys detected:")
    # print(f"  Two LEDs: {smoothed_snr1['Detected_Valleys']}")
    # print(f"  One LED: {smoothed_snr2['Detected_Valleys']}")
    
    # print("\nOverall profile SNR (signal variation / noise):")
    # print(f"  Two LEDs: {smoothed_snr1['Overall_SNR']:.2f}")
    # print(f"  One LED: {smoothed_snr2['Overall_SNR']:.2f}")
    # print(f"  Ratio (One/Two): {smoothed_snr2['Overall_SNR'] / smoothed_snr1['Overall_SNR']:.2f}")
    
    # print("\nAverage Valley SNR (valley depth / noise):")
    # print(f"  Two LEDs: {smoothed_snr1['Avg_Valley_SNR']:.2f}")
    # print(f"  One LED: {smoothed_snr2['Avg_Valley_SNR']:.2f}")
    # print(f"  Ratio (One/Two): {smoothed_snr2['Avg_Valley_SNR'] / smoothed_snr1['Avg_Valley_SNR']:.2f}")
    
    # print("\nValley Quality Score (combines depth, width and noise):")
    # print(f"  Two LEDs: {smoothed_snr1['Valley_Quality_Score']:.2f}")
    # print(f"  One LED: {smoothed_snr2['Valley_Quality_Score']:.2f}")
    # print(f"  Ratio (One/Two): {smoothed_snr2['Valley_Quality_Score'] / smoothed_snr1['Valley_Quality_Score']:.2f}")
    
    # # Print individual valley details for Two LEDs
    # print("\nTwo LEDs - Individual Valley Details:")
    # for i, idx in enumerate(smoothed_snr1['Valley_Indices']):
    #     print(f"  Valley {i+1} (at position {idx}):")
    #     print(f"    SNR: {smoothed_snr1['Valley_SNR_Values'][i]:.2f}")
    #     print(f"    Depth: {smoothed_snr1['Valley_Depths'][i]:.2f}")
    #     print(f"    Width: {smoothed_snr1['Valley_Widths'][i]}")
    #     print(f"    Area: {smoothed_snr1['Valley_Areas'][i]:.2f}")
    
    # # Print individual valley details for One LED
    # print("\nOne LED - Individual Valley Details:")
    # for i, idx in enumerate(smoothed_snr2['Valley_Indices']):
    #     print(f"  Valley {i+1} (at position {idx}):")
    #     print(f"    SNR: {smoothed_snr2['Valley_SNR_Values'][i]:.2f}")
    #     print(f"    Depth: {smoothed_snr2['Valley_Depths'][i]:.2f}")
    #     print(f"    Width: {smoothed_snr2['Valley_Widths'][i]}")
    #     print(f"    Area: {smoothed_snr2['Valley_Areas'][i]:.2f}")
    
    # print("\nNoise Measurements:")
    # print(f"  Two LEDs noise: {smoothed_snr1['Noise_Power']:.4f}")
    # print(f"  One LED noise: {smoothed_snr2['Noise_Power']:.4f}")
    # print(f"  Noise Ratio (One/Two): {smoothed_snr2['Noise_Power'] / smoothed_snr1['Noise_Power']:.2f}")
    
    # print("====================================================\n")
    
    # # Now perform background flattening and analyze the flattened profiles
    # print("\n========== BACKGROUND FLATTENING AND CONTRAST ENHANCEMENT ==========")
    # # Process and compare the flattened profiles
    # flattened1, flattened2 = compare_flattened_profiles(profile1, profile2, window_size=51, plot=True)
    
    # # Calculate SNR on flattened profiles
    # print("\n========== SNR ANALYSIS ON FLATTENED PROFILES ==========")
    
    # # Try different smoothing window sizes for comparison
    # smoothing_windows = [15, 25, 35]
    # flat_snr_results = []
    
    # print("\n========== TESTING DIFFERENT SMOOTHING LEVELS ==========")
    # for window in smoothing_windows:
    #     print(f"\nUsing smoothing window size: {window}")
    #     flat_snr1 = calculate_snr_with_smoothing(flattened1, window_size=window, plot=True, 
    #                                            name=f'Two LEDs (Smoothing={window})')
    #     flat_snr2 = calculate_snr_with_smoothing(flattened2, window_size=window, plot=True, 
    #                                            name=f'One LED (Smoothing={window})')
        
    #     print(f"Two LEDs - Detected valleys: {flat_snr1['Detected_Valleys']}")
    #     print(f"One LED - Detected valleys: {flat_snr2['Detected_Valleys']}")
    #     print(f"Two LEDs - Avg Valley SNR: {flat_snr1['Avg_Valley_SNR']:.2f}")
    #     print(f"One LED - Avg Valley SNR: {flat_snr2['Avg_Valley_SNR']:.2f}")
        
    #     flat_snr_results.append((window, flat_snr1, flat_snr2))
    
    # # Select the optimal window size based on results (using the last one for detailed analysis)
    # optimal_window = smoothing_windows[-1]
    # flat_snr1 = flat_snr_results[-1][1]
    # flat_snr2 = flat_snr_results[-1][2]
    
    # # Print detailed SNR results for flattened profiles with optimal smoothing
    # print(f"\n========== DETAILED SNR RESULTS (FLATTENED PROFILES, SMOOTHING={optimal_window}) ==========")
    # print(f"Number of valleys detected:")
    # print(f"  Two LEDs: {flat_snr1['Detected_Valleys']}")
    # print(f"  One LED: {flat_snr2['Detected_Valleys']}")
    
    # print("\nOverall profile SNR (signal variation / noise):")
    # print(f"  Two LEDs: {flat_snr1['Overall_SNR']:.2f}")
    # print(f"  One LED: {flat_snr2['Overall_SNR']:.2f}")
    # print(f"  Ratio (One/Two): {flat_snr2['Overall_SNR'] / flat_snr1['Overall_SNR']:.2f}")
    
    # print("\nAverage Valley SNR (valley depth / noise):")
    # print(f"  Two LEDs: {flat_snr1['Avg_Valley_SNR']:.2f}")
    # print(f"  One LED: {flat_snr2['Avg_Valley_SNR']:.2f}")
    # print(f"  Ratio (One/Two): {flat_snr2['Avg_Valley_SNR'] / flat_snr1['Avg_Valley_SNR']:.2f}")
    
    # print("\nValley Quality Score (combines depth, width and noise):")
    # print(f"  Two LEDs: {flat_snr1['Valley_Quality_Score']:.2f}")
    # print(f"  One LED: {flat_snr2['Valley_Quality_Score']:.2f}")
    # print(f"  Ratio (One/Two): {flat_snr2['Valley_Quality_Score'] / flat_snr1['Valley_Quality_Score']:.2f}")
    
    # # Print individual valley details with optimal smoothing
    # print("\nTwo LEDs - Individual Valley Details:")
    # for i, idx in enumerate(flat_snr1['Valley_Indices']):
    #     print(f"  Valley {i+1} (at position {idx}):")
    #     print(f"    SNR: {flat_snr1['Valley_SNR_Values'][i]:.2f}")
    #     print(f"    Depth: {flat_snr1['Valley_Depths'][i]:.2f}")
    #     print(f"    Width: {flat_snr1['Valley_Widths'][i]}")
    #     print(f"    Area: {flat_snr1['Valley_Areas'][i]:.2f}")
    
    # print("\nOne LED - Individual Valley Details:")
    # for i, idx in enumerate(flat_snr2['Valley_Indices']):
    #     print(f"  Valley {i+1} (at position {idx}):")
    #     print(f"    SNR: {flat_snr2['Valley_SNR_Values'][i]:.2f}")
    #     print(f"    Depth: {flat_snr2['Valley_Depths'][i]:.2f}")
    #     print(f"    Width: {flat_snr2['Valley_Widths'][i]}")
    #     print(f"    Area: {flat_snr2['Valley_Areas'][i]:.2f}")
    
    # # Print comparison of original vs flattened with optimal smoothing
    # print("\n========== IMPROVEMENT FROM FLATTENING ==========")
    # print(f"Overall SNR improvement:")
    # print(f"  Two LEDs: {flat_snr1['Overall_SNR'] / smoothed_snr1['Overall_SNR']:.2f}x")
    # print(f"  One LED: {flat_snr2['Overall_SNR'] / smoothed_snr2['Overall_SNR']:.2f}x")
    
    # print(f"\nAverage Valley SNR improvement:")
    # print(f"  Two LEDs: {flat_snr1['Avg_Valley_SNR'] / smoothed_snr1['Avg_Valley_SNR']:.2f}x")
    # print(f"  One LED: {flat_snr2['Avg_Valley_SNR'] / smoothed_snr2['Avg_Valley_SNR']:.2f}x")
    
    # print(f"\nValley Quality Score improvement:")
    # print(f"  Two LEDs: {flat_snr1['Valley_Quality_Score'] / smoothed_snr1['Valley_Quality_Score']:.2f}x")
    # print(f"  One LED: {flat_snr2['Valley_Quality_Score'] / smoothed_snr2['Valley_Quality_Score']:.2f}x")
    
    # print("====================================================\n")
    
    # # Create a visualization of the smoothing window comparison
    # plt.figure(figsize=(4.8, 4.0))
    
    # # Plot SNR values by window size
    # window_sizes = [result[0] for result in flat_snr_results]
    # two_led_snrs = [result[1]['Avg_Valley_SNR'] for result in flat_snr_results]
    # one_led_snrs = [result[2]['Avg_Valley_SNR'] for result in flat_snr_results]
    
    # plt.plot(window_sizes, two_led_snrs, 'o-', label='Two LEDs')
    # plt.plot(window_sizes, one_led_snrs, 's-', label='One LED')
    
    # plt.xlabel('Smoothing Window Size')
    # plt.ylabel('Average Valley SNR')
    # plt.title('Effect of Smoothing Window Size on Valley SNR')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    
    # # Use path from config for saving
    # results_path = os.path.join(PATHS['cap_flow'], 'results')
    # snr_path = os.path.join(results_path, 'snr')
    # # Ensure directory exists
    # os.makedirs(snr_path, exist_ok=True)
    # save_path = os.path.join(snr_path, 'smoothing_window_comparison.png')
    # plt.savefig(save_path, dpi=400)
    # # plt.show()
    # plt.close()

if __name__ == '__main__':
    main()