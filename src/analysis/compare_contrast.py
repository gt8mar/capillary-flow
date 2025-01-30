import os
import numpy as np
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from matplotlib.font_manager import FontProperties

def analyze_image_contrast(image_path):
    """
    Analyze contrast metrics for a microscope image.
    
    Parameters:
    image_path (str): Path to the image file
    
    Returns:
    dict: Dictionary containing various contrast metrics
    """
    # Read and convert image to grayscale if needed
    img = io.imread(image_path)
    if len(img.shape) == 3:
        img = rgb2gray(img)
    
    # Convert to float to avoid overflow
    img = img.astype(float)

    # Autocontrast the image
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    
    # Calculate various contrast metrics
    metrics = {}
    
    # 1. Michelson contrast (global)
    i_max = float(np.max(img))
    i_min = float(np.min(img))
    metrics['michelson_contrast'] = (i_max - i_min) / (i_max + i_min)
    
    # 2. RMS contrast
    metrics['rms_contrast'] = np.std(img) / np.mean(img)
    
    # 3. Weber contrast (using mean background)
    # Assume background is represented by the bottom 10% of pixel intensities
    background = float(np.percentile(img, 10))
    foreground = float(np.mean(img[img > background]))
    metrics['weber_contrast'] = (foreground - background) / background
    
    # 4. Intensity range
    metrics['intensity_range'] = i_max - i_min
    
    # 5. Signal-to-noise ratio (SNR)
    metrics['snr'] = np.mean(img) / np.std(img)
    
    return metrics

def compare_images(image_paths):
    """
    Compare contrast metrics across multiple images.
    
    Parameters:
    image_paths (list): List of paths to image files
    
    Returns:
    list: List of tuples containing (image_name, metrics_dict)
    """
    results = []
    
    for path in image_paths:
        name = Path(path).stem
        metrics = analyze_image_contrast(path)
        results.append((name, metrics))
    
    return results

def plot_contrast_comparison(results):
    """
    Create a visualization comparing contrast metrics across images.
    
    Parameters:
    results (list): Output from compare_images function
    """
    metrics = list(results[0][1].keys())
    n_metrics = len(metrics)
    n_images = len(results)
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3*n_metrics))
    fig.suptitle('Contrast Metric Comparison')
    
    for idx, metric in enumerate(metrics):
        values = [result[1][metric] for result in results]
        axes[idx].bar([result[0] for result in results], values)
        axes[idx].set_title(f'{metric.replace("_", " ").title()}')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_contrast_comparison_together(image_paths):
    """
    Create a single bar graph where the different bars for each image are next to each other and
    color coded by metric. The x axis is the image name and the subdivisions on the x axis are the different metrics.
    """
    results = compare_images(image_paths)
    
    # Scale the values of the metrics to be between 0 and 1
    scaled_results = []
    for name, metrics in results:
        scaled_metrics = {}
        for metric in metrics:
            min_val = min(result[1][metric] for result in results)
            max_val = max(result[1][metric] for result in results)
            if max_val != min_val:
                scaled_metrics[metric] = (metrics[metric] - min_val) / (max_val - min_val)
            else:
                scaled_metrics[metric] = 0.5  # If all values are the same
        scaled_results.append((name, scaled_metrics))
    
    # Remove specified metrics
    final_results = []
    for name, metrics in scaled_results:
        filtered_metrics = {k: v for k, v in metrics.items() 
                          if k not in ['snr', 'michelson_contrast', 'intensity_range']}
        final_results.append((name, filtered_metrics))
    
    metrics = list(final_results[0][1].keys())
    n_metrics = len(metrics)
    n_images = len(final_results)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / n_metrics
    colors = plt.cm.tab20(np.linspace(0, 1, n_metrics))
    
    for idx, metric in enumerate(metrics):
        values = [result[1][metric] for result in final_results]
        x = np.arange(n_images) + idx * width
        ax.bar(x, values, width=width, label=metric.replace("_", " ").title(), color=colors[idx])
    
    ax.set_xticks(np.arange(n_images) + 0.4)
    ax.set_xticklabels([f"{result[0].split('_')[0]}_{result[0].split('_')[1]}" 
                        for result in final_results], rotation=45)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    ax.set_title('Contrast Metric Comparison')
    plt.tight_layout()
    return fig

def plot_contrast_boxplots_color(results):
    """
    Create box plots comparing green vs white for RMS and Weber contrast.
    """
    plt.close('all')
    source_sans = FontProperties(fname='C:\\Users\\ejerison\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 4, 'lines.linewidth': 0.5
    })

    # Initialize data structure for green and white metrics
    metrics_by_color = {
        'green': {'rms_contrast': [], 'weber_contrast': []},
        'white': {'rms_contrast': [], 'weber_contrast': []}
    }
    
    # Collect data
    for name, metrics in results:
        color = name.split('_')[1].lower()
        metrics_by_color[color]['rms_contrast'].append(metrics['rms_contrast'])
        metrics_by_color[color]['weber_contrast'].append(metrics['weber_contrast'])
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2))
    fig.suptitle('Contrast Metrics Comparison: Green vs White', fontproperties=source_sans, fontsize = 12)
    
    metrics = ['rms_contrast', 'weber_contrast']
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        box_data = [metrics_by_color['green'][metric], 
                   metrics_by_color['white'][metric]]
        
        bplot = ax.boxplot(box_data, 
                          labels=['Green', 'White'],
                          patch_artist=True)
        
        bplot['boxes'][0].set_facecolor('lightgreen')
        bplot['boxes'][1].set_facecolor('lightgray')
        
        # Add individual points with jitter
        for i, color in enumerate(['green', 'white']):
            x = np.random.normal(i + 1, 0.04, size=len(metrics_by_color[color][metric]))
            ax.plot(x, metrics_by_color[color][metric], 'o', color='black', alpha=0.5, markersize=4)
        
        # Set title and add grid
        ax.set_title(f'{metric.replace("_", " ").title()}', fontproperties=source_sans)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add mean values
        green_mean = np.mean(metrics_by_color['green'][metric])
        white_mean = np.mean(metrics_by_color['white'][metric])
        ax.text(0.98, 0.98, f'Green mean: {green_mean:.3f}\nWhite mean: {white_mean:.3f}', 
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontproperties=source_sans)
    
    plt.tight_layout()
    plt.savefig('C:\\Users\\ejerison\\capillary-flow\\results\\contrast_boxplots.png', dpi=400)
    plt.savefig('C:\\Users\\ejerison\\capillary-flow\\results\\contrast_boxplots.pdf', dpi=400)
    plt.close()
    return 0 # fig

if __name__ == '__main__':
    image_folder = 'I:\\Marcus\\data\\2021\\comparison_folder\\new'
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
    
    # Compare images
    results = compare_images(image_paths)
    
    # Plot separate graphs
    # fig1 = plot_contrast_comparison(results)
    # plt.show()
    
    # Plot combined graph
    # fig2 = plot_contrast_comparison_together(image_paths)
    # plt.show()

    # Create box plots
    fig = plot_contrast_boxplots_color(results)
    # plt.savefig('C:\\Users\\ejerison\\capillary-flow\\results\\contrast_boxplots.png', dpi=400)
    # plt.savefig('C:\\Users\\ejerison\\capillary-flow\\results\\contrast_boxplots.pdf', dpi=400)
    # plt.show()

    # # load new images:
    # image_folder2 = 'C:\\Users\\gt8mar\\Desktop\\data\\241213_led_sides\\fig'
    # image_paths2 = [os.path.join(image_folder2, f) for f in os.listdir(image_folder2)]

    # # Compare images
    # results2 = compare_images(image_paths2)

    # # Plot separate graphs
    # fig1 = plot_contrast_comparison(results2)
    # plt.show()

    # # Plot combined graph
    # fig2 = plot_contrast_comparison_together(image_paths2)
    # plt.show()

    # # Create box plots
    # fig = plot_contrast_boxplots_color(results2)
    # plt.show()