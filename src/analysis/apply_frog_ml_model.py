"""
Filename: src/analysis/apply_frog_ml_model.py

This script loads pre-trained Random Forest and CNN models to estimate the number of red blood cells
in kymograph images without needing ground truth counts.
"""

import os
import numpy as np
import cv2
import joblib
import tensorflow as tf
import pandas as pd
from frog_ml_estimate_counts import (
    extract_features, 
    preprocess_kymograph_for_cnn,
    apply_models_to_new_data
)

def main():
    """
    Main function to apply pre-trained ML models to new kymograph data.
    """
    # Paths
    new_kymograph_dir = "H:\\240729\\Frog2\\Right\\kymographs"  # Directory with your new kymographs
    output_dir = "H:\\240729\\Frog2\\Right\\counts"  # Where to save the predictions
    model_dir = "H:\\frog\\results\\ml_models"  # Where your trained models are saved
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    rf_model_path = os.path.join(model_dir, "random_forest_model.joblib")
    cnn_model_path = os.path.join(model_dir, "cnn_model.h5")
    
    print(f"Loading Random Forest model from {rf_model_path}...")
    rf_model = joblib.load(rf_model_path)
    
    print(f"Loading CNN model from {cnn_model_path}...")
    cnn_model = tf.keras.models.load_model(cnn_model_path)
    
    print("Models loaded successfully!")
    
    # Target size for CNN
    target_size = (100, 1200)
    
    # Apply models to new data
    print(f"Analyzing kymographs in {new_kymograph_dir}...")
    predictions_df = apply_models_to_new_data(
        rf_model, 
        cnn_model, 
        new_kymograph_dir, 
        output_dir, 
        target_size
    )
    
    # Print sample predictions
    if len(predictions_df) > 0:
        print("\nSample predictions:")
        sample_size = min(5, len(predictions_df))
        print(predictions_df.head(sample_size).to_string())
        print(f"\nAll predictions saved to {os.path.join(output_dir, 'predictions.csv')}")
    else:
        print("No kymographs were processed. Check your kymograph directory.")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 