"""
Filename: src/analysis/compare_leds_3.py
----------------------------------------

Compare SNR of one‑dimensional line‑profiles using manual window selection
for valley detection. This method consistently produces the most reliable
SNR measurements for capillary flow images.

This version calculates noise sigma by excluding user-defined valley windows.

By: Marcus Forst
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import savgol_filter
from scipy.ndimage import binary_dilation
from typing import List, Tuple, Optional

# Import paths from config
from src.config import PATHS, load_source_sans

source_sans = load_source_sans()

# Ensure the results directory exists
output_folder = os.path.join(PATHS['cap_flow'], 'results', 'snr')
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------
# MANUAL WINDOW VALLEY DETECTION -------------------------------------
# ---------------------------------------------------------------------
def valley_depths_manual(profile: np.ndarray,
                         windows: List[Tuple[int, int]],
                         baseline_win: int = 60) -> np.ndarray:
    """Computes depths of valleys within explicitly defined pixel windows.

    Args:
        profile: 1-D numpy array representing the intensity profile.
        windows: A list of tuples, where each tuple (start, stop) defines the
                 pixel indices for a window to search for a valley.
        baseline_win: The half-width (in pixels) of the region on each side
                      of the detected valley used to calculate the baseline
                      intensity. The total baseline region width will be
                      2 * baseline_win.

    Returns:
        A 1-D numpy array containing the calculated depth for each valley found
        within the specified windows. Depth is calculated as
        (baseline_intensity - valley_intensity).
    """
    depths = []
    for (lo, hi) in windows:
        seg = profile[lo:hi]
        if seg.size == 0:  # Handle empty segment
            continue
        valley_rel = np.argmin(seg)                # index *within* seg
        valley_abs = lo + valley_rel               # absolute index
        valley_int = profile[valley_abs]

        # baseline = mean of left + right flanks, each baseline_win wide
        left_slice = profile[max(0, valley_abs - baseline_win): valley_abs]
        right_slice = profile[valley_abs + 1: min(profile.size, valley_abs + 1 + baseline_win)]
        
        baseline_points = np.concatenate([left_slice, right_slice])
        if baseline_points.size == 0: # Handle case where baseline region is empty
             # Fallback: Use mean of the whole profile if no baseline points found
             baseline_int = np.mean(profile)
        else:
            baseline_int = np.mean(baseline_points)

        depths.append(baseline_int - valley_int)   # positive number
    
    return np.asarray(depths)


# ---------------------------------------------------------------------
# NOISE CALCULATION --------------------------------------------------
# ---------------------------------------------------------------------
def calculate_noise_sigma(profile: np.ndarray,
                          windows: List[Tuple[int, int]],
                          trend_win: int = 101,
                          trend_poly: int = 3) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates noise sigma excluding specified valley windows.

    Args:
        profile: 1-D numpy array representing the intensity profile.
        windows: A list of tuples defining regions (valleys) to exclude
                 from the noise calculation.
        trend_win: The window length (odd integer) for the Savitzky-Golay filter
                   used for detrending.
        trend_poly: The polynomial order for the Savitzky-Golay filter.

    Returns:
        A tuple containing:
        - sigma_noise (float): The estimated noise standard deviation, calculated
                               from the detrended profile excluding the specified
                               windows.
        - trend (np.ndarray): The calculated low-frequency trend.
        - detrended (np.ndarray): The profile after subtracting the trend.
        - mask (np.ndarray): A boolean mask where True indicates pixels used
                             for noise calculation (i.e., outside the windows).
    """
    # remove low‑frequency illumination trend
    trend = savgol_filter(profile, trend_win, trend_poly, mode="interp")
    detrended = profile - trend

    # Create a mask to exclude the window regions from noise calculation
    mask = np.ones_like(profile, dtype=bool)
    for (lo, hi) in windows:
        mask[lo:hi] = False

    # Calculate noise sigma using only the data outside the windows
    if np.any(mask): # Ensure there are points outside windows
        sigma_noise = detrended[mask].std(ddof=1)
    else: # If windows cover the entire profile
        sigma_noise = detrended.std(ddof=1) # Fallback to using all points

    # Handle potential NaN/Inf if std dev is zero or mask is empty
    if not np.isfinite(sigma_noise):
        sigma_noise = 0.0 # Or raise an error, or use a very small number

    return sigma_noise, trend, detrended, mask


# ---------------------------------------------------------------------
# SNR CALCULATION ----------------------------------------------------
# ---------------------------------------------------------------------
def snr_from_valleys(profile: np.ndarray,
                    depths: np.ndarray,
                    windows: List[Tuple[int, int]],
                    sigma_noise: Optional[float] = None) -> float:
    """Combines valley depths and noise estimate (excluding windows) for SNR.

    Args:
        profile: 1-D numpy array representing the intensity profile.
        depths: A 1-D numpy array of valley depths, typically calculated by
                `valley_depths_manual`.
        windows: A list of tuples defining regions (valleys) that were excluded
                 from the noise calculation.
        sigma_noise: The noise standard deviation. If None, it will be
                     calculated by calling `calculate_noise_sigma`, excluding
                     the provided `windows`.

    Returns:
        The calculated Signal-to-Noise Ratio (SNR), defined as the mean of
        `depths` divided by `sigma_noise`. Returns 0.0 if depths are empty
        or sigma_noise is zero.
    """
    if sigma_noise is None:
        sigma_noise, _, _, _ = calculate_noise_sigma(profile, windows)

    # Handle case where depths might be empty or sigma_noise is zero
    if depths.size == 0 or sigma_noise == 0 or not np.isfinite(sigma_noise):
        return 0.0
        
    return depths.mean() / sigma_noise


# ---------------------------------------------------------------------
# VISUALIZATION -----------------------------------------------------
# ---------------------------------------------------------------------
def visualize_valley_detection(profile: np.ndarray, 
                              windows: List[Tuple[int, int]],
                              baseline_win: int,
                              output_filename: str,
                              title: str = "Valley Detection"):
    """Visualizes manual valley detection, saving the plot.

    Generates a plot showing the intensity profile, the manually defined
    windows, the detected valley within each window, the baseline calculation
    regions and level, and the calculated depth. The final SNR (calculated
    using noise estimated *outside* the windows) is included in the title.
    The figure is saved to `output_filename` and then closed.

    Args:
        profile: 1-D numpy array of the intensity profile.
        windows: List of (start, stop) tuples defining valley search windows.
        baseline_win: Half-width for baseline calculation.
        output_filename: Full path where the plot image should be saved.
        title: Base title for the plot.
    """
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 6, 'lines.linewidth': 0.8
    })
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    x = np.arange(profile.size)
    ax.plot(x, profile, 'k-', lw=1, label="Intensity Profile")
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Calculate depths first to use in SNR calculation for the title
    depths_arr = valley_depths_manual(profile, windows, baseline_win)
    
    # Calculate SNR using noise estimated *outside* these windows
    sigma_noise, _, _, _ = calculate_noise_sigma(profile, windows)
    snr = depths_arr.mean() / sigma_noise if depths_arr.size > 0 and sigma_noise != 0 and np.isfinite(sigma_noise) else 0.0

    valley_positions = []
    baselines = []
    
    # Now plot the details
    for i, (lo, hi) in enumerate(windows):
        color = colors[i % len(colors)]
        ax.axvspan(lo, hi, alpha=0.2, color=color, label=f'Window {i+1}' if i==0 else "_nolegend_")
        
        seg = profile[lo:hi]
        if seg.size == 0: continue
        valley_rel = np.argmin(seg)
        valley_abs = lo + valley_rel
        valley_int = profile[valley_abs]
        valley_positions.append(valley_abs)
        
        left_slice = profile[max(0, valley_abs - baseline_win): valley_abs]
        right_slice = profile[valley_abs + 1: min(profile.size, valley_abs + 1 + baseline_win)]
        baseline_points = np.concatenate([left_slice, right_slice])
        
        if baseline_points.size == 0:
             baseline_int = np.mean(profile) # Fallback
        else:
            baseline_int = np.mean(baseline_points)
        baselines.append(baseline_int)
        
        ax.plot(valley_abs, valley_int, 'o', color=color, ms=6, label='Valley' if i==0 else "_nolegend_")
        # Only add label for the first instance of baseline points/lines
        label_baseline_pts = 'Baseline Points' if i==0 else "_nolegend_"
        label_baseline_line = 'Baseline Level' if i==0 else "_nolegend_"
        ax.plot(range(max(0, valley_abs - baseline_win), valley_abs), 
                left_slice, '.', color=color, ms=3, alpha=0.7, label=label_baseline_pts)
        ax.plot(range(valley_abs + 1, min(profile.size, valley_abs + 1 + baseline_win)), 
                right_slice, '.', color=color, ms=3, alpha=0.7) # No label for right part
        ax.hlines(baseline_int, max(0, valley_abs - baseline_win), 
                  min(profile.size - 1, valley_abs + baseline_win), 
                  color=color, linestyle='--', lw=1, label=label_baseline_line)
        
        # Use pre-calculated depth for consistency
        depth = depths_arr[i] if i < len(depths_arr) else (baseline_int - valley_int)
        ax.vlines(valley_abs, valley_int, baseline_int, color=color, linestyle='-', lw=1.2, label='Depth' if i==0 else "_nolegend_")
        ax.annotate(f"Depth: {depth:.1f}", 
                    xy=(valley_abs, (valley_int + baseline_int)/2),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=7, color=color)
    
    ax.set_title(f"{title} - SNR: {snr:.2f}", fontproperties=source_sans)
    ax.set_xlabel("Position (pixels)", fontproperties=source_sans)
    ax.set_ylabel("Intensity (a.u.)", fontproperties=source_sans)
    # Create legend avoiding duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicates
    ax.legend(by_label.values(), by_label.keys(), loc='best', prop=source_sans)
    fig.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close(fig) # Close the figure to free memory
    
    # Return calculated values
    return depths_arr, valley_positions, baselines, snr


def visualize_noise_calculation(profile: np.ndarray, 
                                sigma_noise: float, 
                                trend: np.ndarray, 
                                detrended: np.ndarray, 
                                mask: np.ndarray,
                                output_filename: str,
                                title: str = "Noise Calculation"):
    """Visualizes the noise calculation process, saving the plot.

    Generates a two-panel plot:
    1. The original profile and the calculated trend line.
    2. The detrended profile, highlighting the regions *excluded* from the
       noise calculation (based on the mask) and showing the ±1 sigma noise band
       calculated from the *included* regions.
    The figure is saved to `output_filename` and then closed.

    Args:
        profile: 1-D numpy array of the original intensity profile.
        sigma_noise: The calculated noise standard deviation.
        trend: The calculated trend line.
        detrended: The profile after trend subtraction.
        mask: Boolean array where True indicates points used for noise calculation.
        output_filename: Full path where the plot image should be saved.
        title: Base title for the plot.
    """
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 6, 'lines.linewidth': 0.8
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x = np.arange(profile.size)
    
    # Plot 1: Profile and Trend
    axes[0].plot(x, profile, 'k-', lw=1, label='Original Profile')
    axes[0].plot(x, trend, 'r-', lw=1, label='Trend (SavGol)')
    axes[0].set_ylabel("Intensity (a.u.)", fontproperties=source_sans)
    axes[0].set_title(title, fontproperties=source_sans)
    axes[0].legend(loc='best', prop=source_sans)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Plot 2: Detrended Profile and Noise
    # Plot points used for noise calculation
    axes[1].plot(x[mask], detrended[mask], 'b.', ms=2, alpha=0.6, label='Data Used for Noise σ')
    # Plot points excluded from noise calculation
    axes[1].plot(x[~mask], detrended[~mask], 'rx', ms=3, alpha=0.7, label='Data Excluded (Windows)')
    
    axes[1].axhline(0, color='k', ls='-', lw=0.5)
    axes[1].axhspan(-sigma_noise, sigma_noise, color='blue', alpha=0.2, 
                   label=f'Noise Band (±σ = {sigma_noise:.2f})')
    axes[1].set_xlabel("Position (pixels)", fontproperties=source_sans)
    axes[1].set_ylabel("Detrended Intensity", fontproperties=source_sans)
    axes[1].legend(loc='best', prop=source_sans)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close(fig) # Close the figure to free memory


def visualize_snr_comparison(profiles: List[np.ndarray], 
                            names: List[str], 
                            windows: List[Tuple[int, int]], 
                            baseline_win: int, 
                            output_filename: str):
    """Compares SNR between profiles using manual windows, saving the plot.

    Generates a two-panel bar chart:
    1. Comparison of the final SNR values for each profile.
    2. Comparison of the average valley depth (signal) and noise sigma
       for each profile.
    Results are sorted by SNR in descending order. The figure is saved to
    `output_filename` and then closed.

    Args:
        profiles: A list of 1-D numpy arrays, each an intensity profile.
        names: A list of strings corresponding to the names of the profiles.
        windows: The list of (start, stop) tuples defining valley windows.
        baseline_win: The half-width used for baseline calculation.
        output_filename: Full path where the plot image should be saved.
    """
    results = []
    for profile, name in zip(profiles, names):
        depths = valley_depths_manual(profile, windows, baseline_win)
        # Calculate noise excluding the windows for this specific profile
        sigma_noise, _, _, _ = calculate_noise_sigma(profile, windows)
        # Recalculate SNR using the specific noise for this profile
        snr = depths.mean() / sigma_noise if depths.size > 0 and sigma_noise != 0 and np.isfinite(sigma_noise) else 0.0
        results.append({
            'name': name,
            'depths': depths, # Store individual depths if needed later
            'avg_depth': depths.mean() if depths.size > 0 else 0.0,
            'sigma_noise': sigma_noise,
            'snr': snr
        })
    
    results.sort(key=lambda x: x['snr'], reverse=True)
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 6
    })
    
    fig = plt.figure(figsize=(8, 7))
    
    # Bar chart of SNRs
    ax1 = fig.add_subplot(2, 1, 1)
    plot_names = [r['name'] for r in results]
    snrs = [r['snr'] for r in results]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax1.bar(plot_names, snrs, color=colors[:len(plot_names)])
    
    ax1.set_title("SNR Comparison", fontproperties=source_sans)
    ax1.set_ylabel("SNR", fontproperties=source_sans)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    max_snr = max(snrs) if snrs else 1
    ax1.set_ylim(0, max_snr * 1.15) # Adjust y-limit for labels
    for bar, snr in zip(bars, snrs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_snr*0.02,
                f'{snr:.2f}', ha='center', fontsize=8)
    
    # Bar chart of depth/noise
    ax2 = fig.add_subplot(2, 1, 2)
    x = np.arange(len(plot_names))
    width = 0.35
    depths_mean = [r['avg_depth'] for r in results]
    noises = [r['sigma_noise'] for r in results]
    
    rects1 = ax2.bar(x - width/2, depths_mean, width, label='Avg. Depth', color='darkblue')
    rects2 = ax2.bar(x + width/2, noises, width, label='Noise σ', color='firebrick')
    
    ax2.set_xlabel("Profile", fontproperties=source_sans)
    ax2.set_ylabel("Intensity", fontproperties=source_sans)
    ax2.set_title("Average Valley Depth vs. Noise", fontproperties=source_sans)
    ax2.set_xticks(x)
    ax2.set_xticklabels(plot_names)
    ax2.legend(prop=source_sans)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels, check for valid data before calculating max_val
    valid_depths = [d for d in depths_mean if np.isfinite(d)]
    valid_noises = [n for n in noises if np.isfinite(n)]
    if valid_depths or valid_noises:
         max_val = max(valid_depths + valid_noises) if (valid_depths + valid_noises) else 1
    else:
        max_val = 1
        
    ax2.set_ylim(0, max_val * 1.15) # Adjust y-limit for labels
    for i, (depth, noise) in enumerate(zip(depths_mean, noises)):
        # Check if values are finite before plotting text
        if np.isfinite(depth):
             ax2.text(i - width/2, depth + max_val*0.02, f'{depth:.1f}', ha='center', fontsize=7)
        if np.isfinite(noise):
             ax2.text(i + width/2, noise + max_val*0.02, f'{noise:.1f}', ha='center', fontsize=7)
    
    fig.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close(fig) # Close the figure to free memory
    
    print("\nSNR Comparison Results (Noise calculated excluding windows):")
    print("-----------------------------------------------------------")
    for r in results:
        print(f"{r['name']}: SNR = {r['snr']:.2f} (Avg Depth = {r['avg_depth']:.1f}, Noise σ = {r['sigma_noise']:.1f})")
    
    return results


def main():
    """Main execution function to load data, run analysis, and generate plots."""
    # Use paths from config
    cap_flow_path = PATHS['cap_flow']
    user_path = os.path.dirname(cap_flow_path)
    # Construct paths relative to the user's directory structure expected
    # Adjust this path if the data is located elsewhere
    image_folder = os.path.join(user_path, 'Desktop', 'data', 'calibration', '241213_led_sides') 
    output_folder = os.path.join(cap_flow_path, 'results', 'snr')
    os.makedirs(output_folder, exist_ok=True) # Ensure output folder exists
    
    # --- Configuration ---
    profile_rows = {'Two LEDs': 574, 'One LED': 634} # Row index for line profile
    profile_cols = {'Two LEDs': (74,-74), 'One LED': (0,-148)} # Start/end columns (slice)
    image_files = {'Two LEDs': 'bothpng.png', 'One LED': 'rightpng.png'}
    
    # Define manual windows for valley detection - Central part of capillaries
    windows = [(283, 380), (363, 420), (591, 657), (657, 732)] 
    baseline_win = 60

    profiles_data = {}
    print("Loading images and extracting profiles...")
    for name, filename in image_files.items():
        img_path = os.path.join(image_folder, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
             print(f"Warning: Could not load image: {img_path}. Skipping profile '{name}'.")
             continue
        
        row = profile_rows[name]
        cols = profile_cols[name]
        # Handle slicing, ensure start < end if using positive indices
        start_col, end_col = cols
        if end_col < 0: # Negative index means count from end
            profile = image[row, start_col:end_col]
        else: # Positive index means count from start
            profile = image[row, start_col:end_col+1] # +1 for inclusive end

        if profile.size == 0:
            print(f"Warning: Extracted profile for '{name}' is empty. Check row/column indices.")
            continue
            
        profiles_data[name] = profile
        print(f"Loaded profile '{name}' with length {profile.size}")

    if not profiles_data:
        print("Error: No valid profiles were loaded. Exiting.")
        return

    all_results = []
    # --- Process and Visualize Each Profile ---
    for name, profile in profiles_data.items():
        print(f"\nProcessing {name} profile...")
        
        # 1. Calculate Noise (excluding windows)
        sigma_noise, trend, detrended, mask = calculate_noise_sigma(profile, windows)
        print(f"  Noise sigma (excluding windows): {sigma_noise:.2f}")
        
        # 2. Visualize Noise Calculation
        noise_plot_filename = os.path.join(output_folder, f"noise_calc_{name.lower().replace(' ', '_')}.png")
        visualize_noise_calculation(profile, sigma_noise, trend, detrended, mask,
                                    noise_plot_filename,
                                    title=f"Noise Calculation - {name}")
        print(f"  Noise calculation plot saved to: {noise_plot_filename}")

        # 3. Visualize Valley Detection (uses the same sigma_noise)
        valley_plot_filename = os.path.join(output_folder, f"valley_detection_{name.lower().replace(' ', '_')}.png")
        depths, _, _, snr = visualize_valley_detection(
            profile, windows, baseline_win, 
            valley_plot_filename,
            title=f"Valley Detection - {name}")
        print(f"  Average valley depth: {depths.mean():.2f}" if depths.size > 0 else "  No valleys found.")
        print(f"  Calculated SNR: {snr:.2f}")
        print(f"  Valley detection plot saved to: {valley_plot_filename}")
        
        all_results.append({'name': name, 'snr': snr, 'avg_depth': depths.mean() if depths.size > 0 else 0.0, 'sigma_noise': sigma_noise})


    # --- Compare SNR and Visualize Results ---
    if len(profiles_data) > 1:
        print("\nGenerating SNR comparison...")
        comparison_plot_filename = os.path.join(output_folder, "snr_comparison.png")
        # Pass the original profiles and names
        comparison_results = visualize_snr_comparison(
            list(profiles_data.values()), 
            list(profiles_data.keys()), 
            windows, 
            baseline_win, 
            comparison_plot_filename
        )
        print(f"  SNR comparison plot saved to: {comparison_plot_filename}")
        
        # --- Print Summary using comparison results (already sorted) ---
        print("\nSummary:")
        if len(comparison_results) >= 2:
            # Assuming the first two results are the ones we want to compare
            snr_1 = comparison_results[0]['snr']
            name_1 = comparison_results[0]['name']
            snr_2 = comparison_results[1]['snr']
            name_2 = comparison_results[1]['name']
            print(f"{name_1} SNR: {snr_1:.2f}")
            print(f"{name_2} SNR: {snr_2:.2f}")
            if snr_2 != 0 and np.isfinite(snr_1) and np.isfinite(snr_2):
                improvement = snr_1 / snr_2
                print(f"Improvement factor ({name_1} vs {name_2}): {improvement:.2f}x")
            else:
                print(f"Improvement factor: N/A (Denominator SNR is zero or invalid)")
        elif len(comparison_results) == 1:
             print(f"{comparison_results[0]['name']} SNR: {comparison_results[0]['snr']:.2f}")

    elif all_results: # Only one profile was processed
        print("\nSummary:")
        print(f"{all_results[0]['name']} SNR: {all_results[0]['snr']:.2f}")
        
    print(f"\nAll plots saved to: {output_folder}")


if __name__ == "__main__":
    main() 