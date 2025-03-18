"""
Filename: src/analysis/apply_frog_ml_models.py

Script to apply trained machine learning models to estimate RBC counts in kymograph images.
This script loads the pre-trained models and applies them to all kymograph images in a specified folder,
then outputs the results to a CSV file.
"""

import os
import pandas as pd
import numpy as np
import cv2
import joblib
import tensorflow as tf
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def extract_features(kymograph):
    """
    Extract handcrafted features from a kymograph for the random forest model.
    
    Args:
        kymograph: A grayscale kymograph image
        
    Returns:
        feature_vector: A 1D numpy array of features
    """
    features = []
    
    # Basic statistical features
    features.append(np.mean(kymograph))  # Mean intensity
    features.append(np.std(kymograph))   # Standard deviation
    features.append(np.median(kymograph))  # Median intensity
    features.append(np.max(kymograph))   # Max intensity
    features.append(np.min(kymograph))   # Min intensity
    features.append(np.percentile(kymograph, 25))  # 25th percentile
    features.append(np.percentile(kymograph, 75))  # 75th percentile
    
    # Gradient-based features
    sobelx = cv2.Sobel(kymograph, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(kymograph, cv2.CV_64F, 0, 1, ksize=3)
    features.append(np.mean(np.abs(sobelx)))  # Mean horizontal gradient
    features.append(np.mean(np.abs(sobely)))  # Mean vertical gradient
    
    # Edge detection features
    edges = cv2.Canny(kymograph.astype(np.uint8), 100, 200)
    features.append(np.sum(edges > 0) / kymograph.size)  # Edge density
    
    # Texture features using GLCM
    if kymograph.shape[0] > 1 and kymograph.shape[1] > 1:  # Check if kymograph is not single pixel
        try:
            from skimage.feature import graycomatrix, graycoprops
            # Normalize and quantize for GLCM
            kymograph_norm = (kymograph / 16).astype(np.uint8)
            glcm = graycomatrix(kymograph_norm, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16, symmetric=True, normed=True)
            features.append(np.mean(graycoprops(glcm, 'contrast')))
            features.append(np.mean(graycoprops(glcm, 'homogeneity')))
            features.append(np.mean(graycoprops(glcm, 'energy')))
            features.append(np.mean(graycoprops(glcm, 'correlation')))
        except Exception as e:
            # If GLCM fails, add zeros
            features.extend([0, 0, 0, 0])
    else:
        features.extend([0, 0, 0, 0])
    
    # Histogram features
    hist = cv2.calcHist([kymograph.astype(np.uint8)], [0], None, [10], [0, 256])
    hist = hist.flatten() / kymograph.size
    features.extend(hist)
    
    # Shape features
    features.append(kymograph.shape[0])  # Height
    features.append(kymograph.shape[1])  # Width
    features.append(kymograph.shape[0] / kymograph.shape[1])  # Aspect ratio
    
    return np.array(features)


def preprocess_kymograph_for_cnn(kymograph, target_size=(100, 1200)):
    """
    Preprocess a kymograph for input to the CNN.
    
    Args:
        kymograph: A grayscale kymograph image
        target_size: Target size for resizing (height, width)
        
    Returns:
        processed_kymograph: Preprocessed kymograph ready for CNN input
    """
    # Resize to target dimensions
    if kymograph.shape[0] != target_size[0] or kymograph.shape[1] != target_size[1]:
        resized = cv2.resize(kymograph, (target_size[1], target_size[0]))
    else:
        resized = kymograph.copy()
    
    # Normalize to [0, 1]
    normalized = resized / 255.0
    
    # Add channel dimension (grayscale = 1 channel)
    with_channel = normalized.reshape(target_size[0], target_size[1], 1)
    
    return with_channel


def load_models(model_dir):
    """
    Load the trained Random Forest and CNN models.
    
    Args:
        model_dir: Directory containing the trained models
        
    Returns:
        rf_model: Trained Random Forest model
        cnn_model: Trained CNN model
    """
    # Load Random Forest model
    rf_model_path = os.path.join(model_dir, 'random_forest_model.joblib')
    if not os.path.exists(rf_model_path):
        raise FileNotFoundError(f"Random Forest model not found at {rf_model_path}")
    
    rf_model = joblib.load(rf_model_path)
    print(f"Loaded Random Forest model from {rf_model_path}")
    
    # Load CNN model
    cnn_model_path = os.path.join(model_dir, 'cnn_model.h5')
    if not os.path.exists(cnn_model_path):
        raise FileNotFoundError(f"CNN model not found at {cnn_model_path}")
    
    cnn_model = tf.keras.models.load_model(cnn_model_path)
    print(f"Loaded CNN model from {cnn_model_path}")
    
    return rf_model, cnn_model


def estimate_counts(kymograph_dir, model_dir, output_dir=None, target_size=(100, 1200), plot_hist=True):
    """
    Estimate RBC counts for all kymographs in a directory using trained models.
    
    Args:
        kymograph_dir: Directory containing kymograph images
        model_dir: Directory containing trained models
        output_dir: Directory to save the results (default: same as kymograph_dir)
        target_size: Target size for CNN images (height, width)
        plot_hist: Whether to plot a histogram of predictions
        
    Returns:
        results_df: DataFrame with the estimated counts
    """
    # Set default output directory
    if output_dir is None:
        output_dir = kymograph_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    rf_model, cnn_model = load_models(model_dir)
    
    # Lists to store results
    results = []
    kymograph_files = []
    rf_predictions = []
    cnn_predictions = []
    ensemble_predictions = []
    
    # Process each kymograph
    print(f"Processing kymographs in {kymograph_dir}...")
    kymograph_count = 0
    
    for kymograph_file in os.listdir(kymograph_dir):
        if not kymograph_file.endswith((".tiff", ".tif")):
            continue
        
        kymograph_count += 1
        if kymograph_count % 10 == 0:
            print(f"Processed {kymograph_count} kymographs...")
        
        # Load image
        kymograph_path = os.path.join(kymograph_dir, kymograph_file)
        kymograph = cv2.imread(kymograph_path, cv2.IMREAD_GRAYSCALE)
        
        if kymograph is None:
            print(f"Warning: Could not read {kymograph_file}")
            continue
        
        # Extract features for Random Forest
        feature_vector = extract_features(kymograph)
        feature_vector = feature_vector.reshape(1, -1)
        
        # Preprocess for CNN
        cnn_input = preprocess_kymograph_for_cnn(kymograph, target_size)
        cnn_input = np.expand_dims(cnn_input, axis=0)
        
        # Make predictions
        rf_prediction = rf_model.predict(feature_vector)[0]
        cnn_prediction = cnn_model.predict(cnn_input, verbose=0)[0][0]
        
        # Average prediction (ensemble)
        ensemble_prediction = (rf_prediction + cnn_prediction) / 2
        
        # Round to nearest integer for cell counts
        rf_prediction_rounded = round(rf_prediction)
        cnn_prediction_rounded = round(cnn_prediction)
        ensemble_prediction_rounded = round(ensemble_prediction)
        
        # Store results
        results.append({
            'Filename': kymograph_file,
            'RF_Prediction': rf_prediction,
            'RF_Prediction_Rounded': rf_prediction_rounded,
            'CNN_Prediction': cnn_prediction,
            'CNN_Prediction_Rounded': cnn_prediction_rounded,
            'Ensemble_Prediction': ensemble_prediction,
            'Ensemble_Prediction_Rounded': ensemble_prediction_rounded
        })
        
        # Store for histogram
        kymograph_files.append(kymograph_file)
        rf_predictions.append(rf_prediction)
        cnn_predictions.append(cnn_prediction)
        ensemble_predictions.append(ensemble_prediction)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    csv_path = os.path.join(output_dir, f'kymograph_count_predictions_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved predictions for {len(results_df)} kymographs to {csv_path}")
    
    # Create a simpler version with just filename and rounded ensemble prediction
    simple_results = results_df[['Filename', 'Ensemble_Prediction_Rounded']].copy()
    simple_results.rename(columns={'Ensemble_Prediction_Rounded': 'Estimated_Count'}, inplace=True)
    
    simple_csv_path = os.path.join(output_dir, f'kymograph_counts_simple_{timestamp}.csv')
    simple_results.to_csv(simple_csv_path, index=False)
    print(f"Saved simplified predictions to {simple_csv_path}")
    
    # Plot histogram of predictions if requested
    if plot_hist and len(results_df) > 0:
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.hist(ensemble_predictions, bins=20, alpha=0.7, label='Ensemble')
        plt.hist(rf_predictions, bins=20, alpha=0.5, label='Random Forest')
        plt.hist(cnn_predictions, bins=20, alpha=0.5, label='CNN')
        
        plt.title('Distribution of Predicted RBC Counts')
        plt.xlabel('Predicted Count')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(output_dir, f'prediction_histogram_{timestamp}.png')
        plt.savefig(plot_path, dpi=300)
        print(f"Saved prediction histogram to {plot_path}")
        
        # Plot correlation between RF and CNN predictions
        plt.figure(figsize=(8, 8))
        plt.scatter(rf_predictions, cnn_predictions, alpha=0.7)
        plt.plot([min(rf_predictions), max(rf_predictions)], 
                [min(rf_predictions), max(rf_predictions)], 'r--')
        plt.title('RF vs CNN Predictions')
        plt.xlabel('Random Forest Prediction')
        plt.ylabel('CNN Prediction')
        plt.grid(True, alpha=0.3)
        
        # Calculate correlation
        correlation = np.corrcoef(rf_predictions, cnn_predictions)[0, 1]
        plt.annotate(f'Correlation: {correlation:.2f}', 
                    xy=(0.05, 0.95), 
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Save correlation plot
        corr_plot_path = os.path.join(output_dir, f'model_correlation_{timestamp}.png')
        plt.savefig(corr_plot_path, dpi=300)
        print(f"Saved model correlation plot to {corr_plot_path}")
    
    return results_df


def main():
    """
    Main function to parse arguments and run the estimation.
    """
    parser = argparse.ArgumentParser(description='Apply trained ML models to estimate RBC counts in kymographs.')
    
    parser.add_argument('--kymograph_dir', type=str, default="D:\\frog\\results\\kymographs",
                        help='Directory containing kymograph images')
    
    parser.add_argument('--model_dir', type=str, default="D:\\frog\\results\\ml_models",
                        help='Directory containing trained models')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the results (default: same as kymograph_dir)')
    
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable histogram plotting')
    
    args = parser.parse_args()
    
    # Estimate counts
    results_df = estimate_counts(
        kymograph_dir=args.kymograph_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        plot_hist=not args.no_plot
    )
    
    # Print summary statistics
    print("\nSummary Statistics for Ensemble Predictions:")
    print(f"Mean: {results_df['Ensemble_Prediction'].mean():.2f}")
    print(f"Median: {results_df['Ensemble_Prediction'].median():.2f}")
    print(f"Min: {results_df['Ensemble_Prediction'].min():.2f}")
    print(f"Max: {results_df['Ensemble_Prediction'].max():.2f}")
    print(f"Standard Deviation: {results_df['Ensemble_Prediction'].std():.2f}")
    
    # Print sample predictions
    print("\nSample predictions (first 5 results):")
    if len(results_df) > 0:
        sample_size = min(5, len(results_df))
        print(results_df[['Filename', 'RF_Prediction_Rounded', 'CNN_Prediction_Rounded', 'Ensemble_Prediction_Rounded']].head(sample_size))


if __name__ == "__main__":
    main() 