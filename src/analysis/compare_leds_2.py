"""
Filename: src/analysis/compare_leds_2.py
----------------------------------------

Compute SNR of one‑dimensional line‑profiles with two approaches:

1. cheap_snr() – difference‑of‑neighbours (fast, no parameters)
2. detailed_snr() – trend removal + automatic masking of anatomical features

By: Marcus Forst
"""
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import binary_dilation
from typing import Optional, Tuple, List  # Add typing imports

# Import paths from config
from src.config import PATHS, load_source_sans

source_sans = load_source_sans()

# ---------------------------------------------------------------------
# 1) CHEAP & EASY – DIFFERENCE‑OF‑NEIGHBOURS ---------------------------
# ---------------------------------------------------------------------
def cheap_snr(profile: np.ndarray) -> float:
    """
    SNR estimate based on the standard deviation of neighbour differences.
    Assumes noise is spatially white.
    """
    diff        = np.diff(profile)               # first‑difference
    sigma_noise = diff.std(ddof=1) / np.sqrt(2)  # white‑noise mapping
    mu_signal   = profile.mean()
    return mu_signal / sigma_noise

# ---------------------------------------------------------------------
# Helper function for noise calculation
# ---------------------------------------------------------------------
def calculate_noise_sigma(profile: np.ndarray,
                          trend_win: int = 101,
                          trend_poly: int = 3,
                          peak_prom: float = 3.0) -> tuple[float, np.ndarray]:
    """
    Calculates the noise standard deviation (sigma) after detrending and masking features.

    Returns
    -------
    tuple[float, np.ndarray]
        A tuple containing the noise standard deviation and the boolean mask used.
    """
    # remove low‑frequency illumination trend
    trend = savgol_filter(profile, trend_win, trend_poly, mode="interp")
    detrended = profile - trend

    # detect peaks AND valleys that are likely anatomical
    sigma_est = detrended.std(ddof=1)
    peaks, _ = find_peaks(detrended, prominence=peak_prom * sigma_est)
    valleys, _ = find_peaks(-detrended, prominence=peak_prom * sigma_est)
    feature_idx = np.concatenate((peaks, valleys))

    # build a boolean mask that is False where features occur
    mask = np.ones_like(profile, dtype=bool)
    if feature_idx.size > 0:
        mask[feature_idx] = False

    # grow the mask by ±2 pixels so steep edges are excluded
    mask = binary_dilation(mask, structure=np.array([1, 1, 1, 1, 1]), iterations=1) # Dilate by 2 pixels on each side

    # noise = σ of detrended data *only* in feature‑free zones
    sigma_noise = detrended[mask].std(ddof=1)
    return sigma_noise, mask


# ---------------------------------------------------------------------
# 2) DETAILED – TREND REMOVAL + FEATURE MASKING -----------------------
# ---------------------------------------------------------------------
def detailed_snr(profile: np.ndarray,
                 trend_win:int = 101,
                 trend_poly:int = 3,
                 peak_prom:float = 3.0) -> tuple[float, np.ndarray]:
    """
    Estimate SNR while avoiding 'real' anatomical features.

    Parameters
    ----------
    profile     : 1‑D numpy array
    trend_win   : Savitzky‑Golay window length (odd integer)
    trend_poly  : Polynomial order for Sav‑Gol fit
    peak_prom   : Minimum peak prominence (× sigma of detrended trace)
                  used to identify features to exclude from σ estimate.

    Returns
    -------
    tuple[float, np.ndarray]
        A tuple containing the estimated SNR and the boolean mask used for features.
    """
    sigma_noise, mask = calculate_noise_sigma(profile, trend_win, trend_poly, peak_prom)

    # signal = mean of *original* profile (or trend) across the whole line
    mu_signal = profile.mean()

    return mu_signal / sigma_noise, mask


# ---------------------------------------------------------------------
# 4) OPTIONAL – VISUALISE WHAT GOT MASKED -----------------------------
# ---------------------------------------------------------------------
def show_mask(profile, mask, title):

    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })


    x = np.arange(profile.size)
    plt.figure(figsize=(9,3))
    plt.plot(x, profile, lw=1)
    # Invert mask for plotting: True where features ARE
    plt.plot(x[~mask], profile[~mask], 'r.', ms=4, label="masked (features)")
    plt.title(title, fontproperties=source_sans)
    plt.xlabel("Position (pixels)", fontproperties=source_sans)
    plt.ylabel("Intensity (a.u.)", fontproperties=source_sans)
    plt.legend(prop=source_sans)
    plt.tight_layout()

# ---------------------------------------------------------------------
# 5) SNR USING VALLEY DEPTH AS THE SIGNAL -----------------------------
# ---------------------------------------------------------------------
def valley_depths_manual(profile: np.ndarray,
                          windows: list[tuple[int, int]],
                          baseline_win: int = 20) -> np.ndarray:
    """
    Compute depths of valleys whose pixel bounds are given explicitly.

    windows      – list of (start, stop) pixel indices for each valley
    baseline_win – half‑width (pixels) used to take baseline from both
                   sides of the valley.  Example: 20 ⇒ 40‑pixel baseline.
    Returns an array of valley depths (one per window).
    """
    depths = []
    for (lo, hi) in windows:
        seg          = profile[lo:hi]
        valley_rel   = np.argmin(seg)                # index *within* seg
        valley_abs   = lo + valley_rel               # absolute index
        valley_int   = profile[valley_abs]

        # baseline = mean of left + right flanks, each baseline_win wide
        left_slice   = profile[max(0, valley_abs - baseline_win): valley_abs]
        right_slice  = profile[valley_abs + 1: valley_abs + 1 + baseline_win]
        baseline_int = np.mean(np.concatenate([left_slice, right_slice]))

        depths.append(baseline_int - valley_int)     # positive number
    return np.asarray(depths)


def valley_depths_auto(profile: np.ndarray,
                       peak_prom: float = 3.0,
                       baseline_win: int = 20,
                       trend_win: int = 101,  # Added trend_win for consistency
                       trend_poly: int = 3    # Added trend_poly for consistency
                       ) -> np.ndarray:
    """
    Same idea, but find valleys automatically by prominence.
    Returns array of depths for *all* detected valleys.
    """
    # Reuse trend calculation consistent with detailed_snr/calculate_noise_sigma
    trend = savgol_filter(profile, trend_win, trend_poly, mode="interp")
    detrended = profile - trend
    sigma_est = detrended.std(ddof=1) # Estimate sigma based on the whole detrended profile initially
    valleys, _ = find_peaks(-detrended, prominence=peak_prom * sigma_est)

    depths = []
    for v in valleys:
        valley_int   = profile[v]
        left_slice   = profile[max(0, v - baseline_win): v]
        right_slice  = profile[v + 1: v + 1 + baseline_win]
        baseline_int = np.mean(np.concatenate([left_slice, right_slice]))
        depths.append(baseline_int - valley_int)
    return np.asarray(depths)


def snr_from_valleys(profile: np.ndarray,
                     depths: np.ndarray,
                     sigma_noise: Optional[float] = None,
                     trend_win: int = 101,
                     trend_poly: int = 3,
                     peak_prom: float = 3.0
                     ) -> float:
    """
    Combine valley depths with a noise estimate to give SNR.
    If sigma_noise is None we compute it using the 'calculate_noise_sigma' routine.
    """
    if sigma_noise is None:
        # Use the helper function to calculate sigma_noise
        sigma_noise, _ = calculate_noise_sigma(profile, trend_win, trend_poly, peak_prom)

    # Handle case where depths might be empty
    if depths.size == 0:
        return 0.0  # Or np.nan, or raise an error, depending on desired behavior

    return depths.mean() / sigma_noise


# ---------------------------------------------------------------------
# 6) VISUALIZE VALLEY DETECTION AND WINDOWING --------------------------
# ---------------------------------------------------------------------
def visualize_valley_detection(profile: np.ndarray, 
                              windows: list[tuple[int, int]] = None,
                              valleys: np.ndarray = None, 
                              baseline_win: int = 20,
                              title: str = "Valley Detection Visualization"):
    """
    Visualize the valley detection process, showing the profile, detected valleys,
    windows (for manual detection), and baselines used for depth calculation.
    
    Parameters
    ----------
    profile      : 1-D numpy array of the profile
    windows      : List of (start, stop) tuples for manual windows
    valleys      : Array of valley indices for automatic detection
    baseline_win : Half-width used for baseline calculation
    title        : Plot title
    """
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    x = np.arange(profile.size)
    plt.figure(figsize=(10, 6))
    
    # Plot the profile
    plt.plot(x, profile, 'k-', lw=1, label="Profile")
    
    # Colors for visualization
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Manual windows visualization
    if windows is not None:
        for i, (lo, hi) in enumerate(windows):
            color = colors[i % len(colors)]
            
            # Highlight the window
            plt.axvspan(lo, hi, alpha=0.2, color=color)
            
            # Find valley within this window
            seg = profile[lo:hi]
            valley_rel = np.argmin(seg)
            valley_abs = lo + valley_rel
            valley_int = profile[valley_abs]
            
            # Calculate baseline from flanks
            left_slice = profile[max(0, valley_abs - baseline_win): valley_abs]
            right_slice = profile[valley_abs + 1: valley_abs + 1 + baseline_win]
            
            # Plot the valley point
            plt.plot(valley_abs, valley_int, 'o', color=color, ms=5)
            
            # Plot baseline regions
            plt.plot(range(max(0, valley_abs - baseline_win), valley_abs), 
                    left_slice, '.', color=color, ms=2, alpha=0.7)
            plt.plot(range(valley_abs + 1, min(profile.size, valley_abs + 1 + baseline_win)), 
                    right_slice, '.', color=color, ms=2, alpha=0.7)
            
            # Calculate and plot baseline level
            baseline_int = np.mean(np.concatenate([left_slice, right_slice]))
            plt.hlines(baseline_int, max(0, valley_abs - baseline_win), 
                      min(profile.size - 1, valley_abs + baseline_win), 
                      color=color, linestyle='--', lw=0.7)
            
            # Visualize depth
            plt.vlines(valley_abs, valley_int, baseline_int, color=color, linestyle='-', lw=1)
            
            # Add annotation
            depth = baseline_int - valley_int
            plt.annotate(f"Depth: {depth:.1f}", 
                        xy=(valley_abs, (valley_int + baseline_int)/2),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=6, color=color)
    
    # Automatic valley detection visualization
    if valleys is not None:
        for i, v in enumerate(valleys):
            color = colors[i % len(colors)]
            valley_int = profile[v]
            
            # Calculate baseline
            left_slice = profile[max(0, v - baseline_win): v]
            right_slice = profile[v + 1: v + 1 + baseline_win]
            
            # Plot the valley point
            plt.plot(v, valley_int, 's', color=color, ms=5)
            
            # Plot baseline regions
            plt.plot(range(max(0, v - baseline_win), v), 
                    left_slice, '.', color=color, ms=2, alpha=0.7)
            plt.plot(range(v + 1, min(profile.size, v + 1 + baseline_win)), 
                    right_slice, '.', color=color, ms=2, alpha=0.7)
            
            # Calculate and plot baseline level
            baseline_int = np.mean(np.concatenate([left_slice, right_slice]))
            plt.hlines(baseline_int, max(0, v - baseline_win), 
                      min(profile.size - 1, v + baseline_win), 
                      color=color, linestyle='--', lw=0.7)
            
            # Visualize depth
            plt.vlines(v, valley_int, baseline_int, color=color, linestyle='-', lw=1)
            
            # Add annotation
            depth = baseline_int - valley_int
            plt.annotate(f"Depth: {depth:.1f}", 
                        xy=(v, (valley_int + baseline_int)/2),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=6, color=color)
    
    plt.title(title, fontproperties=source_sans)
    plt.xlabel("Position (pixels)", fontproperties=source_sans)
    plt.ylabel("Intensity (a.u.)", fontproperties=source_sans)
    
    if windows is not None:
        plt.legend(["Profile", "Valley", "Baseline points", "Baseline level"], 
                  prop=source_sans, loc='best')
    
    plt.tight_layout()


# ---------------------------------------------------------------------
# 7) VISUALIZE SNR CALCULATION METHODS ---------------------------------
# ---------------------------------------------------------------------
def visualize_snr_methods(profile: np.ndarray, title: str = "SNR Methods Visualization"):
    """
    Visualize how the cheap_snr and detailed_snr methods work, showing the
    intermediate calculations and principles behind each method.
    
    Parameters
    ----------
    profile : 1-D numpy array
    title   : Plot title
    """
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    x = np.arange(profile.size)
    fig = plt.figure(figsize=(10, 8))
    
    # ----- Cheap SNR Method ----- #
    # Calculate components for cheap SNR
    diff = np.diff(profile)
    sigma_noise_cheap = diff.std(ddof=1) / np.sqrt(2)
    mu_signal_cheap = profile.mean()
    snr_cheap = mu_signal_cheap / sigma_noise_cheap
    
    # ----- Detailed SNR Method ----- #
    # Calculate components for detailed SNR
    trend_win, trend_poly, peak_prom = 101, 3, 3.0  # Default parameters
    
    # Get detrending components
    trend = savgol_filter(profile, trend_win, trend_poly, mode="interp")
    detrended = profile - trend
    
    # Get feature detection components
    sigma_est = detrended.std(ddof=1)
    peaks, _ = find_peaks(detrended, prominence=peak_prom * sigma_est)
    valleys, _ = find_peaks(-detrended, prominence=peak_prom * sigma_est)
    feature_idx = np.concatenate((peaks, valleys))
    
    # Create mask
    mask = np.ones_like(profile, dtype=bool)
    if feature_idx.size > 0:
        mask[feature_idx] = False
    mask_dilated = binary_dilation(mask, structure=np.array([1, 1, 1, 1, 1]), iterations=1)
    
    # Calculate noise using the mask
    sigma_noise_detailed = detrended[mask_dilated].std(ddof=1)
    mu_signal_detailed = profile.mean()
    snr_detailed = mu_signal_detailed / sigma_noise_detailed
    
    # ----- Create the plots ----- #
    # Plot 1: Original profile
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(x, profile, 'k-', lw=1, label="Original Profile")
    ax1.axhline(mu_signal_cheap, color='r', ls='--', lw=0.8, label=f"Mean Signal: {mu_signal_cheap:.1f}")
    ax1.set_title(f"{title} - Original Profile", fontproperties=source_sans)
    ax1.set_ylabel("Intensity (a.u.)", fontproperties=source_sans)
    ax1.legend(prop=source_sans, loc='best')
    
    # Plot 2: Cheap SNR - Difference of neighbors
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(x[:-1], diff, 'b-', lw=0.7, label="First Difference")
    ax2.axhline(0, color='k', ls='-', lw=0.5)
    ax2.axhspan(-sigma_noise_cheap*2, sigma_noise_cheap*2, alpha=0.2, color='b', label=f"±2σ noise: {sigma_noise_cheap:.1f}")
    ax2.set_title(f"Cheap SNR Method: {snr_cheap:.1f} (mean / (std(diff) / √2))", fontproperties=source_sans)
    ax2.set_ylabel("Diff. Int.", fontproperties=source_sans)
    ax2.legend(prop=source_sans, loc='best')
    
    # Plot 3: Detailed SNR - Detrending and Feature Masking
    ax3 = plt.subplot(3, 1, 3)
    
    # Plot detrended data
    ax3.plot(x, detrended, 'g-', lw=0.7, label="Detrended Profile")
    ax3.axhline(0, color='k', ls='-', lw=0.5)
    
    # Mark peaks and valleys
    for peak in peaks:
        ax3.plot(peak, detrended[peak], 'r^', ms=4)
    for valley in valleys:
        ax3.plot(valley, detrended[valley], 'rv', ms=4)
    
    # Highlight masked regions
    masked_x = x[~mask_dilated]
    masked_y = detrended[~mask_dilated]
    ax3.plot(masked_x, masked_y, 'rx', ms=3, label="Masked Features")
    
    # Show noise estimation region
    non_masked_x = x[mask_dilated]
    non_masked_y = detrended[mask_dilated]
    ax3.plot(non_masked_x, non_masked_y, 'g.', ms=1, alpha=0.5, label="Used for σ")
    
    # Add noise levels
    ax3.axhspan(-sigma_noise_detailed*2, sigma_noise_detailed*2, alpha=0.1, color='g', 
               label=f"±2σ noise: {sigma_noise_detailed:.1f}")
    
    ax3.set_title(f"Detailed SNR Method: {snr_detailed:.1f} (mean / σ_masked)", fontproperties=source_sans)
    ax3.set_xlabel("Position (pixels)", fontproperties=source_sans)
    ax3.set_ylabel("Detrended Int.", fontproperties=source_sans)
    ax3.legend(prop=source_sans, loc='best')
    
    fig.tight_layout()

    # ---------------------------------------------------------------------
# PARAMETER‑FREE BASELINE (auto trend) --------------------------------
# ---------------------------------------------------------------------
def _auto_trend(profile: np.ndarray) -> np.ndarray:
    """
    Return a smooth trend with an automatically chosen window length
    (10 % of the line length, at least 51 px, forced to be odd).
    """
    n   = profile.size
    win = max(51, ((n // 10) | 1))      # make odd with bit‑or 1
    return savgol_filter(profile, win, 3, mode="interp")


def valley_depths_trend(profile: np.ndarray,
                        peak_prom: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Depths defined as:  baseline(trend) – valley_intensity

    Returns
    -------
    depths  : 1‑D array of valley depths (positive values)
    valleys : indices of the valley pixels
    """
    # trend      = _auto_trend(profile) 
    trend      = robust_trend(profile)  
    detrended  = profile - trend
    sigma_est  = detrended.std(ddof=1)
    valleys, _ = find_peaks(-detrended, prominence=peak_prom * sigma_est)

    depths = trend[valleys] - profile[valleys]   # vectorised, no loops
    return depths, valleys


def snr_from_valleys_trend(profile: np.ndarray,
                           peak_prom: float = 3.0) -> float:
    """
    SNR = mean(valley_depths_trend) / σ_noise
    Noise σ is computed exactly the same way as in detailed_snr().
    """
    depths, _ = valley_depths_trend(profile, peak_prom=peak_prom)

    # Re‑use the existing helper for σ_noise so the mask logic is identical
    sigma_noise, _ = calculate_noise_sigma(profile,
                                           peak_prom=peak_prom)

    return depths.mean() / sigma_noise


# ---------------------------------------------------------------------
# 8) VISUALIZE TREND-BASED VALLEY DETECTION ---------------------------
# ---------------------------------------------------------------------
def visualize_trend_method(profile: np.ndarray, title: str = "Trend-Based Method Visualization"):
    """
    Visualize how the trend-based valley detection method works.
    
    Parameters
    ----------
    profile : 1-D numpy array
    title   : Plot title
    """
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    x = np.arange(profile.size)
    fig = plt.figure(figsize=(10, 9))
    
    # ----- Calculate trend and detrended profile ----- #
    # trend = _auto_trend(profile)
    trend = robust_trend(profile)
    detrended = profile - trend
    
    # Auto window length calculation
    n = profile.size
    win = max(51, ((n // 10) | 1))
    
    # ----- Get valleys based on detrended profile ----- #
    peak_prom = 3.0
    sigma_est = detrended.std(ddof=1)
    valleys, _ = find_peaks(-detrended, prominence=peak_prom * sigma_est)
    
    # ----- Calculate depths directly from trend ----- #
    depths = trend[valleys] - profile[valleys]
    
    # ----- Calculate noise for SNR ----- #
    sigma_noise, mask = calculate_noise_sigma(profile, peak_prom=peak_prom)
    
    # ----- Calculate SNR ----- #
    snr = depths.mean() / sigma_noise
    
    # ----- Create the plots ----- #
    # Plot 1: Original profile with trend
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(x, profile, 'k-', lw=1, label="Original Profile")
    ax1.plot(x, trend, 'r-', lw=1, label=f"Auto Trend (window={win})")
    
    # Mark valleys
    ax1.plot(valleys, profile[valleys], 'bo', ms=4, label="Detected Valleys")
    
    # Draw lines from valleys to trend to show depth
    for i, v in enumerate(valleys):
        ax1.vlines(v, profile[v], trend[v], color='g', linestyle='-', lw=0.8)
        ax1.annotate(f"{depths[i]:.1f}", 
                   xy=(v, (profile[v] + trend[v])/2),
                   xytext=(5, 0), textcoords='offset points',
                   fontsize=6, color='g')
    
    ax1.set_title(f"{title} - Profile & Trend", fontproperties=source_sans)
    ax1.set_ylabel("Intensity (a.u.)", fontproperties=source_sans)
    ax1.legend(prop=source_sans, loc='best')
    
    # Plot 2: Detrended profile with valley detection
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(x, detrended, 'b-', lw=0.7, label="Detrended Profile")
    ax2.axhline(0, color='k', ls='-', lw=0.5)
    
    # Show prominence threshold
    threshold = -peak_prom * sigma_est
    ax2.axhline(threshold, color='r', ls='--', lw=0.5, 
              label=f"Valley Threshold ({peak_prom}σ)")
    
    # Mark valleys on detrended profile
    ax2.plot(valleys, detrended[valleys], 'ro', ms=4, label="Detected Valleys")
    
    # Show prominence for each valley
    for v in valleys:
        # Draw vertical line for prominence visualization
        ax2.vlines(v, detrended[v], 0, color='r', linestyle=':', lw=0.8)
    
    ax2.set_title(f"Valley Detection in Detrended Profile", fontproperties=source_sans)
    ax2.set_ylabel("Detrended Int.", fontproperties=source_sans)
    ax2.legend(prop=source_sans, loc='best')
    
    # Plot 3: Noise calculation (similar to detailed SNR)
    ax3 = plt.subplot(3, 1, 3)
    
    # Plot detrended data again for noise visualization
    ax3.plot(x, detrended, 'g-', lw=0.7, label="Detrended Profile")
    ax3.axhline(0, color='k', ls='-', lw=0.5)
    
    # Show masked areas used for noise calculation
    masked_x = x[~mask]
    masked_y = detrended[~mask]
    ax3.plot(masked_x, masked_y, 'rx', ms=3, label="Masked Features")
    
    # Show the points used for σ noise calculation
    non_masked_x = x[mask]
    non_masked_y = detrended[mask]
    ax3.plot(non_masked_x, non_masked_y, 'g.', ms=1, alpha=0.5, label="Used for σ")
    
    # Show noise band
    ax3.axhspan(-sigma_noise*2, sigma_noise*2, alpha=0.1, color='g', 
               label=f"±2σ noise: {sigma_noise:.1f}")
    
    ax3.set_title(f"Noise Estimation for SNR: {snr:.1f} (mean(depths) / σ)", 
                 fontproperties=source_sans)
    ax3.set_xlabel("Position (pixels)", fontproperties=source_sans)
    ax3.set_ylabel("Detrended Int.", fontproperties=source_sans)
    ax3.legend(prop=source_sans, loc='best')
    
    fig.tight_layout()
    
    # Return for possible further use
    return valleys, depths

def robust_trend(profile: np.ndarray,
                 win_frac: float = 0.08,      # 8 % of trace length
                 poly: int = 3,
                 n_iter: int = 3,
                 clip_sigma: float = 2.5) -> np.ndarray:
    """
    Iteratively fit a SavGol trend, discarding points that fall ≥clip_sigma
    BELOW the trend (i.e. valleys) at each iteration.
    """
    n   = profile.size
    win = max(51, ((int(n * win_frac)) | 1))        # odd length
    mask = np.ones(n, bool)

    trend = savgol_filter(profile, win, poly, mode="interp")
    for _ in range(n_iter):
        resid = profile - trend
        # mask out deep dips only (negative residuals)
        mask &= resid > -clip_sigma * resid[mask].std(ddof=1)
        trend = savgol_filter(profile[mask], win, poly, mode="interp")
        # restore full length by linear interpolation
        trend = np.interp(np.arange(n), np.where(mask)[0], trend)

    return trend



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
    profile_both = both_image[574, 74:-74]
    profile_right = right_image[634, 0:-148]

    # ---------------------------------------------------------------------
    # 3) RUN & PRINT -------------------------------------------------------
    # ---------------------------------------------------------------------
    cheap_both  = cheap_snr(profile_both)
    cheap_right  = cheap_snr(profile_right)
    # Get SNR and mask from detailed_snr
    det_both, mask_both = detailed_snr(profile_both)
    det_right, mask_right = detailed_snr(profile_right)

    print(f"Cheap method  : Two LEDs SNR = {cheap_both:5.2f}   | One LED SNR = {cheap_right:5.2f}")
    print(f"Detailed meth.: Two LEDs SNR = {det_both:5.2f}   | One LED SNR = {det_right:5.2f}")

    # Visualize how the SNR methods work
    visualize_snr_methods(profile_both, "Two LEDs Profile")
    plt.show()
    
    visualize_snr_methods(profile_right, "One LED Profile")
    plt.show()

    # Use the masks returned by detailed_snr
    show_mask(profile_both, mask_both, "Two LEDs – features highlighted")
    show_mask(profile_right, mask_right, "One LED  – features highlighted")
    plt.show()

    # ----- manual windows, e.g. [(350, 400), (550, 600), ...] -------------
    windows = [(283, 380), (363, 420), (591, 657), (657, 732)]
    depths_both = valley_depths_manual(profile_both, windows, baseline_win=60)
    depths_right = valley_depths_manual(profile_right, windows, baseline_win=60)
    snr_both_manu = snr_from_valleys(profile_both, depths_both)
    snr_right_manu = snr_from_valleys(profile_right, depths_right)
    
    # Visualize manual valley detection
    visualize_valley_detection(profile_both, windows=windows, baseline_win=60, title="Two LEDs - Manual Valley Detection")
    visualize_valley_detection(profile_right, windows=windows, baseline_win=60, title="One LED - Manual Valley Detection")
    plt.show()

    # ----- automatic detection --------------------------------------------
    depths_both_a = valley_depths_auto(profile_both)
    depths_right_a = valley_depths_auto(profile_right)
    
    # Get valleys for visualization (need to recompute them)
    trend_both = savgol_filter(profile_both, 101, 3, mode="interp")
    detrended_both = profile_both - trend_both
    valleys_both, _ = find_peaks(-detrended_both, prominence=3.0 * detrended_both.std(ddof=1))
    
    trend_right = savgol_filter(profile_right, 101, 3, mode="interp")
    detrended_right = profile_right - trend_right
    valleys_right, _ = find_peaks(-detrended_right, prominence=3.0 * detrended_right.std(ddof=1))
    
    # Visualize automatic valley detection
    visualize_valley_detection(profile_both, valleys=valleys_both, title="Two LEDs - Automatic Valley Detection")
    visualize_valley_detection(profile_right, valleys=valleys_right, title="One LED - Automatic Valley Detection")
    plt.show()
    
    # Pass parameters consistently if sigma_noise needs to be calculated inside
    snr_both_auto = snr_from_valleys(profile_both, depths_both_a)
    snr_right_auto = snr_from_valleys(profile_right, depths_right_a)

    print(f"Manual‑window SNR : Two LEDs = {snr_both_manu:5.2f} | One LED = {snr_right_manu:5.2f}")
    print(f"Auto‑detect  SNR : Two LEDs = {snr_both_auto:5.2f} | One LED = {snr_right_auto:5.2f}")

    # ----- trend-based detection -----------------------------------------
    snr_both_trend = snr_from_valleys_trend(profile_both)
    snr_right_trend = snr_from_valleys_trend(profile_right)
    
    # Visualize trend-based valley detection
    visualize_trend_method(profile_both, "Two LEDs - Trend-Based Valley Detection")
    plt.show()
    
    visualize_trend_method(profile_right, "One LED - Trend-Based Valley Detection")
    plt.show()

    print(f"Trend‑baseline SNR: Two LEDs = {snr_both_trend:5.2f} | "
        f"One LED = {snr_right_trend:5.2f}")


if __name__ == "__main__":
    main()
