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
        # Ensure directory exists
        os.makedirs(results_path, exist_ok=True)
        save_path = os.path.join(results_path, 'image_comparison.png')
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
        # Ensure directory exists
        os.makedirs(results_path, exist_ok=True)
        save_path = os.path.join(results_path, 'line_profile_comparison.png')
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
        plt.show()
    
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
        # plt.show()
    return 0

def analyze_line_profile_quality(profile1, profile2, names=('Two LEDs', 'One LED'), plot=True):
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
        # Ensure directory exists
        os.makedirs(results_path, exist_ok=True)
        save_path = os.path.join(results_path, 'line_profile_comparisonsnr.png')
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
        # Ensure directory exists
        os.makedirs(results_path, exist_ok=True)
        save_path = os.path.join(results_path, 'normalized_enhanced_profiles.png')
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
        # Ensure directory exists
        os.makedirs(results_path, exist_ok=True)
        save_path = os.path.join(results_path, 'normalized_profiles_with_snr.png')
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

    # Compare line profiles
    profile_results = compare_line_profiles(profile1, profile2)
    print(profile_results)

    # Analyze line profile quality
    profile_quality = analyze_line_profile_quality(profile1, profile2)
    print(profile_quality)
    
    # Normalize and enhance profiles
    normalized_profiles = normalize_and_enhance_profiles(profile1, profile2)
    print("Normalized and enhanced profiles created and saved")
    
    # Normalize with SNR
    normalized_snr = normalize_with_snr(profile1, profile2)
    print("Normalized profiles with SNR created and saved")
    print(normalized_snr)

if __name__ == '__main__':
    main()