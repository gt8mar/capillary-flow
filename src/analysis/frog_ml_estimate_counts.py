"""
Filename: src/analysis/frog_ml_estimate_counts.py

File to estimate the number of rbcs in a kymograph using a machine learning model.
"""

import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import time


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


def create_cnn_model(input_shape):
    """
    Create a CNN model for kymograph count estimation.
    
    Args:
        input_shape: Tuple of (height, width, channels)
        
    Returns:
        model: A compiled Keras model
    """
    model = Sequential([
        # First convolution block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Second convolution block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Third convolution block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1)  # Single output neuron for regression
    ])
    
    # Compile model with mean squared error loss and Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    
    return model


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


def prepare_data(kymograph_dir, counts_df, target_size=(100, 1200), test_size=0.2, random_state=42):
    """
    Prepare data for both Random Forest and CNN models.
    
    Args:
        kymograph_dir: Directory containing kymograph images
        counts_df: DataFrame with count information
        target_size: Target size for CNN images (height, width)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        rf_data: Dictionary with Random Forest data
        cnn_data: Dictionary with CNN data
    """
    # Filter out unreliable data (where Adjustment_Type is "TODO")
    if 'Adjustment_Type' in counts_df.columns:
        reliable_counts_df = counts_df[counts_df['Adjustment_Type'] != "TODO"].copy()
        filtered_count = len(counts_df) - len(reliable_counts_df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} unreliable records with Adjustment_Type='TODO'")
        counts_df = reliable_counts_df
    
    # Lists to store data
    kymographs = []
    features = []
    counts = []
    file_names = []
    velocities = []
    
    # Process each kymograph
    for kymograph_file in os.listdir(kymograph_dir):
        if not kymograph_file.endswith(".tiff"):
            continue
        
        # Find corresponding count in dataframe
        file_basename = os.path.basename(kymograph_file)
        count_row = counts_df[counts_df['Image_Path'] == file_basename]
        
        if count_row.empty:
            continue
        
        count = count_row['Final_Counts'].values[0]
        
        # Get velocity if available
        velocity = None
        if 'Velocity (um/s)' in count_row.columns:
            velocity = count_row['Velocity (um/s)'].values[0]
        
        # Load and preprocess image
        kymograph_path = os.path.join(kymograph_dir, kymograph_file)
        kymograph = cv2.imread(kymograph_path, cv2.IMREAD_GRAYSCALE)
        
        if kymograph is None:
            print(f"Warning: Could not read {kymograph_file}")
            continue
        
        # Extract features for Random Forest
        feature_vector = extract_features(kymograph)
        
        # Preprocess for CNN
        cnn_input = preprocess_kymograph_for_cnn(kymograph, target_size)
        
        # Store data
        kymographs.append(cnn_input)
        features.append(feature_vector)
        counts.append(count)
        file_names.append(file_basename)
        if velocity is not None:
            velocities.append(velocity)
    
    # Convert to numpy arrays
    X_features = np.array(features)
    X_images = np.array(kymographs)
    y = np.array(counts)
    
    print(f"Loaded {len(y)} valid kymographs with reliable counts")
    
    # Split data into training and testing sets
    if len(y) > 5:  # Only split if we have enough data
        X_features_train, X_features_test, X_images_train, X_images_test, y_train, y_test, files_train, files_test = train_test_split(
            X_features, X_images, y, file_names, test_size=test_size, random_state=random_state
        )
        
        # Create data dictionaries
        rf_data = {
            'X_train': X_features_train,
            'X_test': X_features_test,
            'y_train': y_train,
            'y_test': y_test,
            'files_train': files_train,
            'files_test': files_test
        }
        
        cnn_data = {
            'X_train': X_images_train,
            'X_test': X_images_test,
            'y_train': y_train,
            'y_test': y_test,
            'files_train': files_train,
            'files_test': files_test
        }
        
        print(f"Data split into {len(y_train)} training samples and {len(y_test)} testing samples")
    else:
        # If not enough data, just use all for training
        rf_data = {
            'X_train': X_features,
            'X_test': X_features,
            'y_train': y,
            'y_test': y,
            'files_train': file_names,
            'files_test': file_names
        }
        
        cnn_data = {
            'X_train': X_images,
            'X_test': X_images,
            'y_train': y,
            'y_test': y,
            'files_train': file_names,
            'files_test': file_names
        }
        
        print("Warning: Not enough data for train/test split. Using all data for both training and testing.")
    
    return rf_data, cnn_data


def train_random_forest(rf_data, n_estimators=100, random_state=42):
    """
    Train a Random Forest model on kymograph features.
    
    Args:
        rf_data: Dictionary with Random Forest data
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility
        
    Returns:
        model: Trained Random Forest model
        metrics: Dictionary with performance metrics
    """
    print("Training Random Forest model...")
    start_time = time.time()
    
    # Create pipeline with feature scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=n_estimators, random_state=random_state))
    ])
    
    # Train model
    pipeline.fit(rf_data['X_train'], rf_data['y_train'])
    
    # Make predictions
    y_pred = pipeline.predict(rf_data['X_test'])
    
    # Calculate metrics
    mae = mean_absolute_error(rf_data['y_test'], y_pred)
    rmse = np.sqrt(mean_squared_error(rf_data['y_test'], y_pred))
    r2 = r2_score(rf_data['y_test'], y_pred)
    
    training_time = time.time() - start_time
    
    # Store metrics
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'training_time': training_time,
        'predictions': y_pred,
        'actual': rf_data['y_test']
    }
    
    print(f"Random Forest - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, Training time: {training_time:.2f}s")
    
    return pipeline, metrics


def train_cnn(cnn_data, input_shape, epochs=50, batch_size=8, patience=10, model_path=None):
    """
    Train a CNN model on kymograph images.
    
    Args:
        cnn_data: Dictionary with CNN data
        input_shape: Input shape of the images (height, width, channels)
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        patience: Early stopping patience
        model_path: Path to save the best model
        
    Returns:
        model: Trained CNN model
        metrics: Dictionary with performance metrics
    """
    print("Training CNN model...")
    start_time = time.time()
    
    # Create model
    model = create_cnn_model(input_shape)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    ]
    
    if model_path:
        callbacks.append(ModelCheckpoint(model_path, save_best_only=True))
    
    # Train model
    history = model.fit(
        cnn_data['X_train'], cnn_data['y_train'],
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Make predictions
    y_pred = model.predict(cnn_data['X_test']).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(cnn_data['y_test'], y_pred)
    rmse = np.sqrt(mean_squared_error(cnn_data['y_test'], y_pred))
    r2 = r2_score(cnn_data['y_test'], y_pred)
    
    training_time = time.time() - start_time
    
    # Store metrics
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'training_time': training_time,
        'predictions': y_pred,
        'actual': cnn_data['y_test'],
        'history': history.history
    }
    
    print(f"CNN - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, Training time: {training_time:.2f}s")
    
    return model, metrics


def compare_models(rf_metrics, cnn_metrics, output_dir):
    """
    Compare Random Forest and CNN models and generate comparison plots.
    
    Args:
        rf_metrics: Dictionary with Random Forest metrics
        cnn_metrics: Dictionary with CNN metrics
        output_dir: Directory to save comparison plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison table
    comparison_data = {
        'Model': ['Random Forest', 'CNN'],
        'MAE': [rf_metrics['mae'], cnn_metrics['mae']],
        'RMSE': [rf_metrics['rmse'], cnn_metrics['rmse']],
        'R²': [rf_metrics['r2'], cnn_metrics['r2']],
        'Training Time (s)': [rf_metrics['training_time'], cnn_metrics['training_time']]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Create actual vs predicted plot
    plt.figure(figsize=(12, 6))
    
    # Random Forest
    plt.subplot(1, 2, 1)
    plt.scatter(rf_metrics['actual'], rf_metrics['predictions'], alpha=0.7)
    plt.plot([min(rf_metrics['actual']), max(rf_metrics['actual'])], 
             [min(rf_metrics['actual']), max(rf_metrics['actual'])], 'r--')
    plt.title('Random Forest: Actual vs Predicted')
    plt.xlabel('Actual Counts')
    plt.ylabel('Predicted Counts')
    plt.grid(True, alpha=0.3)
    
    # CNN
    plt.subplot(1, 2, 2)
    plt.scatter(cnn_metrics['actual'], cnn_metrics['predictions'], alpha=0.7)
    plt.plot([min(cnn_metrics['actual']), max(cnn_metrics['actual'])], 
             [min(cnn_metrics['actual']), max(cnn_metrics['actual'])], 'r--')
    plt.title('CNN: Actual vs Predicted')
    plt.xlabel('Actual Counts')
    plt.ylabel('Predicted Counts')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300)
    
    # If CNN has history, plot learning curves
    if 'history' in cnn_metrics:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(cnn_metrics['history']['loss'], label='Train Loss')
        plt.plot(cnn_metrics['history']['val_loss'], label='Validation Loss')
        plt.title('CNN Learning Curves - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(cnn_metrics['history']['mean_absolute_error'], label='Train MAE')
        plt.plot(cnn_metrics['history']['val_mean_absolute_error'], label='Validation MAE')
        plt.title('CNN Learning Curves - MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cnn_learning_curves.png'), dpi=300)
    
    # Print comparison summary
    print("\nModel Comparison Summary:")
    print(comparison_df.to_string(index=False))
    print("\nResults saved to:", output_dir)


def apply_models_to_new_data(rf_model, cnn_model, kymograph_dir, output_dir, target_size=(100, 1200)):
    """
    Apply trained models to new kymograph images.
    
    Args:
        rf_model: Trained Random Forest model
        cnn_model: Trained CNN model
        kymograph_dir: Directory containing new kymograph images
        output_dir: Directory to save results
        target_size: Target size for CNN images (height, width)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for kymograph_file in os.listdir(kymograph_dir):
        if not kymograph_file.endswith(".tiff"):
            continue
        
        # Load and preprocess image
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
        cnn_prediction = cnn_model.predict(cnn_input)[0][0]
        
        # Average prediction (ensemble)
        ensemble_prediction = (rf_prediction + cnn_prediction) / 2
        
        # Store results
        results.append({
            'File': kymograph_file,
            'RF_Prediction': rf_prediction,
            'CNN_Prediction': cnn_prediction,
            'Ensemble_Prediction': ensemble_prediction
        })
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    print(f"Applied models to {len(results)} new images. Results saved to {os.path.join(output_dir, 'predictions.csv')}")
    
    return results_df


def main():
    """
    Main function to estimate the number of rbcs in a kymograph using a machine learning model.
    
    Kymograph directory:
    - D:\\frog\\results\\kymographs

    Training kymograph data directory:
    - D:\\frog\\calibration\\kymographs

    Training counts data file:
    - D:\\frog\\counted_kymos_CalFrog4_final_final.csv

        Counts row:
        - "Final_Counts"

        Kymograph identifier row:
        - "Image_Path" (just the basename of the kymograph file)

        Kymograph velocity row:
        - "Velocity (um/s)"

        Counting status row:
        - "Adjustment_Type" if this is "TODO" then the counts are not reliable, drop these rows.


    Output directory:
    - D:\\frog\\results

    """
    kymograph_dir = "D:\\frog\\results\\kymographs"
    output_dir = "D:\\frog\\results"
    training_kymograph_dir = "D:\\frog\\calibration\\kymographs"
    training_counts_file = "D:\\frog\\counted_kymos_CalFrog4_final_final.csv"
    model_output_dir = os.path.join(output_dir, "ml_models")
    comparison_output_dir = os.path.join(output_dir, "model_comparison")
    
    # Create output directories
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(comparison_output_dir, exist_ok=True)
    
    # Load training data
    print(f"Loading training data from {training_counts_file}...")
    counts_df = pd.read_csv(training_counts_file)
    print(f"Loaded {len(counts_df)} records from CSV")
    
    # Check if Adjustment_Type column exists
    if 'Adjustment_Type' not in counts_df.columns:
        print("Warning: 'Adjustment_Type' column not found in the CSV. Cannot filter unreliable data.")
    
    # Define target size for CNN (height, width)
    # Using a fixed height close to the median with the standard width of 1200
    target_size = (100, 1200)
    
    # Prepare data
    print("\nPreparing data...")
    rf_data, cnn_data = prepare_data(training_kymograph_dir, counts_df, target_size)
    
    # Check if we have enough data
    min_samples = 5
    if len(rf_data['y_train']) < min_samples:
        print(f"Error: Not enough training data. Need at least {min_samples} samples, but only have {len(rf_data['y_train'])}.")
        return
    
    # Train Random Forest model
    rf_model, rf_metrics = train_random_forest(rf_data)
    
    # Save Random Forest model
    rf_model_path = os.path.join(model_output_dir, 'random_forest_model.joblib')
    joblib.dump(rf_model, rf_model_path)
    print(f"Random Forest model saved to {rf_model_path}")
    
    # Train CNN model
    input_shape = (target_size[0], target_size[1], 1)  # Height, width, channels
    cnn_model_path = os.path.join(model_output_dir, 'cnn_model.h5')
    cnn_model, cnn_metrics = train_cnn(cnn_data, input_shape, model_path=cnn_model_path)
    print(f"CNN model saved to {cnn_model_path}")
    
    # Compare models
    compare_models(rf_metrics, cnn_metrics, comparison_output_dir)
    
    # Apply models to new data
    print("\nApplying models to new data...")
    predictions_df = apply_models_to_new_data(rf_model, cnn_model, kymograph_dir, output_dir, target_size)
    
    # Print some sample predictions
    print("\nSample predictions:")
    if len(predictions_df) > 0:
        sample_size = min(5, len(predictions_df))
        print(predictions_df.head(sample_size).to_string())
    
    print("\nDone!")


def get_kymograph_sizes(kymograph_dir):
    """
    Get the sizes of the kymographs in the directory. for testing only, now ignore. 
    """
    kymograph_dir = "D:\\frog\\results\\kymographs"

    kymograph_sizes = []
    kymograph_rows = []
    kymograph_cols = []

    for kymograph_file in os.listdir(kymograph_dir):
        if not kymograph_file.endswith(".tiff"):
            continue

        kymograph = cv2.imread(os.path.join(kymograph_dir, kymograph_file), cv2.IMREAD_GRAYSCALE)
        kymograph_sizes.append(kymograph.shape)
        kymograph_rows.append(kymograph.shape[0])
        kymograph_cols.append(kymograph.shape[1])
    
    # plot the histogram of kymograph rows
    plt.hist(kymograph_rows, bins=100)
    plt.title("Histogram of kymograph rows")
    plt.xlabel("Rows")
    plt.ylabel("Frequency")
    # plt.show()

    # print information about the distribution of kymograph rows
    print('Information about the distribution of kymograph rows:')
    print(f"Mean: {np.mean(kymograph_rows)}")
    print(f"Median: {np.median(kymograph_rows)}")
    print(f"Standard deviation: {np.std(kymograph_rows)}")
    print(f"Minimum: {np.min(kymograph_rows)}")
    print(f"Maximum: {np.max(kymograph_rows)}")
    print(f'kurtosis: {pd.Series(kymograph_rows).kurt()}')
    print(f'skewness: {pd.Series(kymograph_rows).skew()}')
    print(f'mode: {pd.Series(kymograph_rows).mode()}') 
    

    # plot the histogram of kymograph cols
    plt.hist(kymograph_cols, bins=100)
    plt.title("Histogram of kymograph cols")
    plt.xlabel("Cols")
    plt.ylabel("Frequency")
    # plt.show()

    # print unique values of kymograph cols
    print('Information about the distribution of kymograph cols:')
    print(f"Unique values of kymograph cols: {np.unique(kymograph_cols)}")

if __name__ == "__main__":
    main()
