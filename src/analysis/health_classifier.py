"""Module for classifying health conditions based on capillary flow metrics.

This module implements various classification techniques to predict diabetes,
hypertension, and heart disease from capillary flow measurements and evaluates
feature importance.
"""

import platform
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from typing import Dict, Tuple, List
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.utils.class_weight import compute_class_weight
from matplotlib.font_manager import FontProperties
from imblearn.over_sampling import SMOTE
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# Import paths from config instead of defining computer paths locally
from src.config import PATHS, load_source_sans
cap_flow_path = PATHS['cap_flow']
source_sans = load_source_sans()

def prepare_data() -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Load and prepare data for classification, focusing on velocity measurements.
    
    Returns:
        Tuple containing:
            - DataFrame with all features
            - Dictionary with target variables and their X, y arrays
    """
    # Load data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Debug print
    print("\nUnique diabetes values in dataset:")
    print(df['Diabetes'].unique())
    print("\nValue counts for Diabetes:")
    print(df['Diabetes'].value_counts())
    
    # After loading df
    print("\nRaw data condition counts:")
    print("\nDiabetes values:")
    print(df['Diabetes'].value_counts(dropna=False))
    print("\nHypertension values:")
    print(df['Hypertension'].value_counts(dropna=False))
    # print("\nHeartDisease values:")
    # print(df['HeartDisease'].value_counts(dropna=False))
    
    # Create features for each participant
    participant_data = []
    for participant in df['Participant'].unique():
        participant_df = df[df['Participant'] == participant]
        
        # Basic velocity statistics for each pressure
        pressure_stats = participant_df.pivot_table(
            index='Participant',
            columns='Pressure',
            values='Video_Median_Velocity',
            aggfunc=['mean']  # Simplified to just mean velocity
        ).fillna(0)  # Fill NaN with 0 for missing pressures
        
        # Flatten multi-index columns
        pressure_stats.columns = [f'velocity_at_{pressure}psi' 
                                for (_, pressure) in pressure_stats.columns]
        
        # Calculate velocity response characteristics
        velocity_values = participant_df.groupby('Pressure')['Video_Median_Velocity'].mean().values
        
        # Calculate up/down velocity differences
        up_velocities = participant_df[participant_df['UpDown'] == 'U']['Video_Median_Velocity']
        down_velocities = participant_df[participant_df['UpDown'] == 'D']['Video_Median_Velocity']
        
        # Basic statistics
        stats = {
            'Participant': participant,
            
            # Key velocity features
            'baseline_velocity': velocity_values[0] if len(velocity_values) > 0 else 0,
            'max_velocity': np.max(velocity_values) if len(velocity_values) > 0 else 0,
            'velocity_range': (np.max(velocity_values) - np.min(velocity_values)) if len(velocity_values) > 0 else 0,
            'mean_velocity': np.mean(velocity_values) if len(velocity_values) > 0 else 0,
            'velocity_std': np.std(velocity_values) if len(velocity_values) > 0 else 0,
            
            # Up/Down differences
            'up_down_diff': np.mean(up_velocities) - np.mean(down_velocities) if len(up_velocities) > 0 and len(down_velocities) > 0 else 0,
            
            # Basic demographic info (as control variables)
            'Age': participant_df['Age'].iloc[0],
            # 'SYS_BP': participant_df['SYS_BP'].iloc[0] if 'SYS_BP' in participant_df else None,
            # 'DIA_BP': participant_df['DIA_BP'].iloc[0] if 'DIA_BP' in participant_df else None,
            
            # Target variables
            'Diabetes': str(participant_df['Diabetes'].iloc[0]).upper() == 'TRUE',
            'Hypertension': participant_df['Hypertension'].iloc[0] == True,
            # 'HeartDisease': participant_df['HeartDisease'].iloc[0] == True
        }
        
        # Add pressure-specific velocity features
        stats.update(pressure_stats.iloc[0].to_dict())
        
        participant_data.append(stats)
    
    processed_df = pd.DataFrame(participant_data)
    
    # Handle missing values
    numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
    
    # Print feature names for verification
    feature_cols = [col for col in processed_df.columns 
                   if col not in ['Participant', 'Diabetes', 'Hypertension']] # 'HeartDisease'
    print("\nFeatures being used:")
    for col in feature_cols:
        print(f"- {col}")
    
    # Prepare X and y for each condition
    target_dict = {}
    for condition in ['Diabetes', 'Hypertension']: # 'HeartDisease'
        X = processed_df[feature_cols].values
        y = processed_df[condition].values
        target_dict[condition] = (X, y)
    
    # Print data shape and feature info
    print(f"\nTotal samples: {len(processed_df)}")
    print("\nFeature value ranges:")
    for col in processed_df.columns:
        if col not in ['Participant', 'Diabetes', 'Hypertension']: # 'HeartDisease'
            print(f"{col}:")
            print(f"  Range: {processed_df[col].min():.2f} to {processed_df[col].max():.2f}")
            print(f"  Mean: {processed_df[col].mean():.2f}")
            print(f"  Null values: {processed_df[col].isnull().sum()}")

    # Print correlation with target variables
    for condition in ['Diabetes', 'Hypertension']: # 'HeartDisease'
        print(f"\nTop 5 correlations with {condition}:")
        correlations = processed_df.drop(['Participant'], axis=1).corr()[condition]
        print(correlations.sort_values(ascending=False)[:5])
    
    return processed_df, target_dict

def prepare_data_log(df) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Load and prepare data for classification, focusing on velocity measurements.
    
    Returns:
        Tuple containing:
            - DataFrame with all features
            - Dictionary with target variables and their X, y arrays
    """    
    # Debug print
    print("\nUnique diabetes values in dataset:")
    print(df['Diabetes'].unique())
    print("\nValue counts for Diabetes:")
    print(df['Diabetes'].value_counts())
    
    # After loading df
    print("\nRaw data condition counts:")
    print("\nDiabetes values:")
    print(df['Diabetes'].value_counts(dropna=False))
    print("\nHypertension values:")
    print(df['Hypertension'].value_counts(dropna=False))
    # print("\nHeartDisease values:")
    # print(df['HeartDisease'].value_counts(dropna=False))
    
    # Create features for each participant
    participant_data = []
    for participant in df['Participant'].unique():
        participant_df = df[df['Participant'] == participant]
        
        # Basic velocity statistics for each pressure
        pressure_stats = participant_df.pivot_table(
            index='Participant',
            columns='Pressure',
            values='Log_Video_Median_Velocity',
            aggfunc=['mean']  # Simplified to just mean velocity
        ).fillna(0)  # Fill NaN with 0 for missing pressures
        
        # Flatten multi-index columns
        pressure_stats.columns = [f'log_velocity_at_{pressure}psi' 
                                for (_, pressure) in pressure_stats.columns]
        
        # Calculate velocity response characteristics
        velocity_values = participant_df.groupby('Pressure')['Log_Video_Median_Velocity'].mean().values
        
        # Calculate up/down velocity differences
        up_velocities = participant_df[participant_df['UpDown'] == 'U']['Video_Median_Velocity']
        down_velocities = participant_df[participant_df['UpDown'] == 'D']['Video_Median_Velocity']
        
        # Basic statistics
        stats = {
            'Participant': participant,
            
            # # Key velocity features
            # 'baseline_velocity': velocity_values[0] if len(velocity_values) > 0 else 0,
            # 'max_velocity': np.max(velocity_values) if len(velocity_values) > 0 else 0,       # this value might be somewhat helpful but makes it slightly worse
            # 'velocity_range': (np.max(velocity_values) - np.min(velocity_values)) if len(velocity_values) > 0 else 0, # this variable makes the auc way worse
            # 'mean_velocity': np.mean(velocity_values) if len(velocity_values) > 0 else 0,
            # 'velocity_std': np.std(velocity_values) if len(velocity_values) > 0 else 0,       # this variable makes the auc way worse
            
            # Up/Down differences
            'up_down_diff': np.mean(up_velocities) - np.mean(down_velocities) if len(up_velocities) > 0 and len(down_velocities) > 0 else 0,
            
            # Basic demographic info (as control variables)
            # 'Age': participant_df['Age'].iloc[0],
            # 'SYS_BP': participant_df['SYS_BP'].iloc[0] if 'SYS_BP' in participant_df else None,
            # 'DIA_BP': participant_df['DIA_BP'].iloc[0] if 'DIA_BP' in participant_df else None,
            
            # Target variables
            'Diabetes': str(participant_df['Diabetes'].iloc[0]).upper() == 'TRUE',
            'Hypertension': participant_df['Hypertension'].iloc[0] == True
            # 'HeartDisease': participant_df['HeartDisease'].iloc[0] == True
        }
        
        # Add pressure-specific velocity features
        stats.update(pressure_stats.iloc[0].to_dict())
        
        participant_data.append(stats)
    
    processed_df = pd.DataFrame(participant_data)
    
    # Handle missing values
    numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
    
    # Print feature names for verification
    feature_cols = [col for col in processed_df.columns 
                   if col not in ['Participant', 'Diabetes', 'Hypertension']] # 'HeartDisease'
    print("\nFeatures being used:")
    for col in feature_cols:
        print(f"- {col}")
    
    # Prepare X and y for each condition
    target_dict = {}
    for condition in ['Diabetes', 'Hypertension']: # 'HeartDisease'
        X = processed_df[feature_cols].values
        y = processed_df[condition].values
        target_dict[condition] = (X, y)
    
    # Print data shape and feature info
    print(f"\nTotal samples: {len(processed_df)}")
    print("\nFeature value ranges:")
    for col in processed_df.columns:
        if col not in ['Participant', 'Diabetes', 'Hypertension']: # 'HeartDisease'
            print(f"{col}:")
            print(f"  Range: {processed_df[col].min():.2f} to {processed_df[col].max():.2f}")
            print(f"  Mean: {processed_df[col].mean():.2f}")
            print(f"  Null values: {processed_df[col].isnull().sum()}")

    # Print correlation with target variables
    for condition in ['Diabetes', 'Hypertension']: # 'HeartDisease'
        print(f"\nTop 5 correlations with {condition}:")
        correlations = processed_df.drop(['Participant'], axis=1).corr()[condition]
        print(correlations.sort_values(ascending=False)[:5])
    
    return processed_df, target_dict


def plot_auc_curves(results: Dict, condition: str, output_dir: str):
    """Plot ROC curves with AUC for all classifiers.
    
    Args:
        results: Dictionary containing classifier results
        condition: Name of health condition being classified
        output_dir: Directory to save the plot
    """
    plt.close()
    # Set up style and font
    sns.set_style("whitegrid")
    
    # Try to use Source Sans font if available, otherwise use default
    try:
        source_sans = FontProperties(fname=os.path.join(PATHS['downloads'], 'Source_Sans_3\\static\\SourceSans3-Regular.ttf'))
    except:
        print("Source Sans font not found, using default font")
        source_sans = None
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

    plt.figure(figsize=(2.4, 2))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot ROC curve for each classifier
    for (name, res), color in zip(results.items(), colors):
        try:
            # Get predictions for test set
            if hasattr(res['classifier'], 'predict_proba'):
                y_pred = res['classifier'].predict_proba(res['X_test'])[:, 1]
            else:
                y_pred = res['classifier'].predict(res['X_test'])
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(res['y_test'], y_pred)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{name} (AUC = {roc_auc:.2f})')
        except Exception as e:
            print(f"Warning: Could not plot ROC curve for {name}: {str(e)}")
            continue
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontproperties=source_sans if source_sans else None)
    plt.ylabel('True Positive Rate', fontproperties=source_sans if source_sans else None)
    plt.title(f'ROC Curves for {condition} Classification', fontproperties=source_sans if source_sans else None)
    plt.legend(loc="lower right", prop=source_sans if source_sans else None)
    
    # Save plot directly to output_dir
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=400, bbox_inches='tight')
    except Exception as e:
        print(f"Warning: Could not save ROC curve plot: {str(e)}")
    plt.close()
    return 0

def plot_3D_features(processed_df, results, feature_cols, condition, output_dir):
    """
    Create 3D visualization of the top 3 important features with predictions
    for both training and test data to evaluate model performance.
    
    Args:
        processed_df: DataFrame containing participant data
        results: Dictionary containing classifier results
        feature_cols: List of feature names
        condition: Name of health condition being classified
        output_dir: Directory to save the visualization

    Returns:
        0 if run correctly
    """
    plt.close()
    # Set up style and font
    sns.set_style("whitegrid")
    
    # Try to use Source Sans font if available, otherwise use default
    try:
        source_sans = FontProperties(fname=os.path.join(PATHS['downloads'], 'Source_Sans_3\\static\\SourceSans3-Regular.ttf'))
    except:
        print("Source Sans font not found, using default font")
        source_sans = None  
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    # For this visualization we need both test and training data
    # Get the Random Forest classifier and its results
    rf_results = results['Random Forest']
    classifier = rf_results['classifier']
    X_test = rf_results['X_test']
    y_test = rf_results['y_test']
    
    # We need to extract the training data that was used to train the model
    # First get the original feature matrix and target vector
    original_X = processed_df[[col for col in processed_df.columns 
                   if col not in ['Participant', 'Diabetes', 'Hypertension', 'is_healthy']]].values
    original_y = processed_df[condition if condition != 'Healthy_vs_Affected' else 'is_healthy'].values
    
    # Scale the features as done in evaluate_classifiers
    scaler = StandardScaler()
    original_X_scaled = scaler.fit_transform(original_X)
    
    # Split using the same random state as in evaluate_classifiers
    X_train_full, X_test_full, y_train, y_test_full = train_test_split(
        original_X_scaled, original_y, test_size=0.2, random_state=42, stratify=original_y
    )
    
    # Apply SMOTE to training data as done in evaluate_classifiers
    X_train_resampled, y_train_resampled = apply_smote(X_train_full, y_train)
    
    # Get feature importances from trained Random Forest
    importances = classifier.feature_importances_
    
    # Get selected features from feature importance
    if len(feature_cols) == 1:
        selected_features = feature_cols
        X_train = X_train_resampled  # No feature selection needed
    else:
        # Get selected features (the ones with importance > mean)
        feature_importance = pd.Series(
            classifier.feature_importances_,
            index=rf_results['feature_importance'].index
        )
        selected_features = feature_importance.index.tolist()
        
        # Get indices of selected features in original feature space
        selected_indices = [feature_cols.index(feature) for feature in selected_features 
                          if feature in feature_cols]
        
        # Extract selected features from training data
        X_train = X_train_resampled[:, selected_indices] if selected_indices else X_train_resampled
    
    # Get top 3 features for visualization
    top_features = rf_results['feature_importance'].index[:3].tolist()
    print(f"Top 3 features for visualization: {top_features}")
    
    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=(20, 10))
    
    # ==================== Plot Test Data ====================
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Predict on test data
    if hasattr(classifier, 'predict_proba'):
        y_test_prob = classifier.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= 0.5).astype(int)
    else:
        y_test_pred = classifier.predict(X_test)
    
    # Create confusion matrix categories for test data
    test_categories = np.zeros(len(y_test), dtype=int)
    test_categories[np.logical_and(y_test == 0, y_test_pred == 0)] = 0  # True Negative
    test_categories[np.logical_and(y_test == 0, y_test_pred == 1)] = 1  # False Positive
    test_categories[np.logical_and(y_test == 1, y_test_pred == 0)] = 2  # False Negative
    test_categories[np.logical_and(y_test == 1, y_test_pred == 1)] = 3  # True Positive
    
    # Define colors and labels for confusion matrix categories
    colors = ['#2ca02c', '#ff7f0e', '#d62728', '#1f77b4']  # TN, FP, FN, TP
    color_labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    
    # Plot each category for test data
    for i, label in enumerate(color_labels):
        mask = test_categories == i
        if np.any(mask):
            ax1.scatter(
                X_test[mask, 0], 
                X_test[mask, 1], 
                X_test[mask, 2] if X_test.shape[1] > 2 else np.zeros(np.sum(mask)),
                c=[colors[i]],
                label=label,
                s=70,
                alpha=0.7,
                edgecolors='w',
                linewidth=0.5
            )
    
    # Add labels and title for test data plot
    if len(top_features) >= 3:
        ax1.set_xlabel(top_features[0], fontsize=10)
        ax1.set_ylabel(top_features[1], fontsize=10)
        ax1.set_zlabel(top_features[2], fontsize=10)
    else:
        ax1.set_xlabel(top_features[0] if len(top_features) > 0 else "Feature 1", fontsize=10)
        ax1.set_ylabel(top_features[1] if len(top_features) > 1 else "Feature 2", fontsize=10)
        ax1.set_zlabel("Feature 3", fontsize=10)
        
    ax1.set_title(f'Test Data - {condition} Classification', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Calculate metrics for test data
    tn_test = np.sum(test_categories == 0)
    fp_test = np.sum(test_categories == 1)
    fn_test = np.sum(test_categories == 2)
    tp_test = np.sum(test_categories == 3)
    
    test_accuracy = (tp_test + tn_test) / (tp_test + tn_test + fp_test + fn_test) if (tp_test + tn_test + fp_test + fn_test) > 0 else 0
    test_precision = tp_test / (tp_test + fp_test) if (tp_test + fp_test) > 0 else 0
    test_recall = tp_test / (tp_test + fn_test) if (tp_test + fn_test) > 0 else 0
    test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
    
    # Add test metrics as text on the plot
    ax1.text2D(0.05, 0.05, 
               f"Test Data Performance:\n"
               f"Accuracy: {test_accuracy:.3f}\n"
               f"Precision: {test_precision:.3f}\n"
               f"Recall: {test_recall:.3f}\n"
               f"F1 Score: {test_f1:.3f}\n"
               f"Samples: {len(y_test)}",
               fontsize=10, transform=ax1.transAxes,
               bbox=dict(facecolor='white', alpha=0.8))
    
    # ==================== Plot Training Data ====================
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Predict on training data (to check for overfitting)
    if hasattr(classifier, 'predict_proba'):
        y_train_prob = classifier.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_prob >= 0.5).astype(int)
    else:
        y_train_pred = classifier.predict(X_train)
    
    # Create confusion matrix categories for training data
    train_categories = np.zeros(len(y_train_resampled), dtype=int)
    train_categories[np.logical_and(y_train_resampled == 0, y_train_pred == 0)] = 0  # True Negative
    train_categories[np.logical_and(y_train_resampled == 0, y_train_pred == 1)] = 1  # False Positive
    train_categories[np.logical_and(y_train_resampled == 1, y_train_pred == 0)] = 2  # False Negative
    train_categories[np.logical_and(y_train_resampled == 1, y_train_pred == 1)] = 3  # True Positive
    
    # Plot each category for training data
    for i, label in enumerate(color_labels):
        mask = train_categories == i
        if np.any(mask):
            ax2.scatter(
                X_train[mask, 0], 
                X_train[mask, 1], 
                X_train[mask, 2] if X_train.shape[1] > 2 else np.zeros(np.sum(mask)),
                c=[colors[i]],
                label=label,
                s=70,
                alpha=0.7,
                edgecolors='w',
                linewidth=0.5
            )
    
    # Add labels and title for training data plot
    if len(top_features) >= 3:
        ax2.set_xlabel(top_features[0], fontsize=10)
        ax2.set_ylabel(top_features[1], fontsize=10)
        ax2.set_zlabel(top_features[2], fontsize=10)
    else:
        ax2.set_xlabel(top_features[0] if len(top_features) > 0 else "Feature 1", fontsize=10)
        ax2.set_ylabel(top_features[1] if len(top_features) > 1 else "Feature 2", fontsize=10)
        ax2.set_zlabel("Feature 3", fontsize=10)
        
    ax2.set_title(f'Training Data (with SMOTE) - {condition} Classification', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Calculate metrics for training data
    tn_train = np.sum(train_categories == 0)
    fp_train = np.sum(train_categories == 1)
    fn_train = np.sum(train_categories == 2)
    tp_train = np.sum(train_categories == 3)
    
    train_accuracy = (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train) if (tp_train + tn_train + fp_train + fn_train) > 0 else 0
    train_precision = tp_train / (tp_train + fp_train) if (tp_train + fp_train) > 0 else 0
    train_recall = tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else 0
    train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
    
    # Add training metrics as text on the plot
    ax2.text2D(0.05, 0.05, 
               f"Training Data Performance:\n"
               f"Accuracy: {train_accuracy:.3f}\n"
               f"Precision: {train_precision:.3f}\n"
               f"Recall: {train_recall:.3f}\n"
               f"F1 Score: {train_f1:.3f}\n"
               f"Samples: {len(y_train_resampled)} (after SMOTE)",
               fontsize=10, transform=ax2.transAxes,
               bbox=dict(facecolor='white', alpha=0.8))
    
    # Add an overall title comparing train vs test performance
    plt.suptitle(f"Model Performance Comparison - {condition}\n" +
                f"Train Accuracy: {train_accuracy:.3f}  vs  Test Accuracy: {test_accuracy:.3f}",
                fontsize=16)
    
    # Save plot with descriptive filename
    plot_filename = f'3d_feature_plot_train_test_comparison_{condition.lower().replace(" ", "_")}.png'
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Create a summary of overfitting analysis
    overfitting_report_path = os.path.join(output_dir, f'overfitting_analysis_{condition.lower().replace(" ", "_")}.txt')
    with open(overfitting_report_path, 'w') as f:
        f.write(f"Overfitting Analysis for {condition} Classification\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Random Forest Performance Comparison\n")
        f.write("-" * 40 + "\n\n")
        
        f.write(f"{'Metric':<15} {'Training':<15} {'Test':<15} {'Difference':<15}\n")
        f.write(f"{'Accuracy':<15} {train_accuracy:.3f}{'':<15} {test_accuracy:.3f}{'':<15} {train_accuracy-test_accuracy:.3f}\n")
        f.write(f"{'Precision':<15} {train_precision:.3f}{'':<15} {test_precision:.3f}{'':<15} {train_precision-test_precision:.3f}\n")
        f.write(f"{'Recall':<15} {train_recall:.3f}{'':<15} {test_recall:.3f}{'':<15} {train_recall-test_recall:.3f}\n")
        f.write(f"{'F1 Score':<15} {train_f1:.3f}{'':<15} {test_f1:.3f}{'':<15} {train_f1-test_f1:.3f}\n\n")
        
        # Add interpretation
        f.write("Interpretation:\n")
        if train_accuracy - test_accuracy > 0.2:
            f.write("- STRONG EVIDENCE OF OVERFITTING: The model performs significantly better on training data.\n")
        elif train_accuracy - test_accuracy > 0.1:
            f.write("- MODERATE OVERFITTING: There is a notable gap between training and test performance.\n")
        elif train_accuracy - test_accuracy > 0.05:
            f.write("- SLIGHT OVERFITTING: There is a small gap between training and test performance.\n")
        else:
            f.write("- GOOD GENERALIZATION: The model performs similarly on training and test data.\n")
            
        if test_accuracy < 0.6:
            f.write("- LOW TEST ACCURACY: The model struggles to generalize to unseen data.\n")
        
        # Add recommendations
        f.write("\nRecommendations:\n")
        if train_accuracy - test_accuracy > 0.1:
            f.write("- Consider regularization techniques to reduce overfitting\n")
            f.write("- Simplify the model (reduce max_depth, min_samples_leaf, etc.)\n")
            f.write("- Collect more diverse training data if possible\n")
        if test_accuracy < 0.6:
            f.write("- The feature set may not be predictive enough for this classification task\n")
            f.write("- Try feature engineering to create more informative features\n")
            f.write("- Consider different algorithms or ensemble methods\n")
        
    return 0

def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to balance classes in the training data.
    
    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple containing:
            - Resampled feature matrix
            - Resampled target vector
    """
    print("\nClass distribution before SMOTE:")
    print(Counter(y))
    
    # Initialize SMOTE
    smote = SMOTE(random_state=random_state)
    
    try:
        # Apply SMOTE to generate synthetic samples
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print("Class distribution after SMOTE:")
        print(Counter(y_resampled))
        
        return X_resampled, y_resampled
    
    except ValueError as e:
        print(f"Warning: SMOTE failed - {str(e)}")
        print("Falling back to original data")
        return X, y

def evaluate_classifiers(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
    """Evaluate different classification models with SMOTE balancing and feature selection.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
    
    Returns:
        Dictionary containing model performances and feature importance
    """
    # Check if we have enough classes for classification
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"Warning: Only one class found ({unique_classes[0]}). Skipping classification.")
        return None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data BEFORE applying SMOTE (to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE to training data only
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # Skip feature selection if there's only one feature
    if len(feature_names) == 1:
        print(f"Only one feature ({feature_names[0]}), skipping feature selection")
        selected_features = feature_names
        selected_indices = [0]  # Only one feature at index 0
        X_train_selected = X_train_resampled
        X_test_selected = X_test
    else:
        # Feature selection using Random Forest on resampled data
        selector = RandomForestClassifier(n_estimators=100, random_state=42)
        selector.fit(X_train_resampled, y_train_resampled)
        
        # Get feature importance scores
        importance_scores = pd.Series(selector.feature_importances_, index=feature_names)
        selected_features = importance_scores[importance_scores > importance_scores.mean()].index
        print("\nSelected features:", selected_features.tolist())
        
        # Get indices of selected features
        selected_indices = [feature_names.index(feature) for feature in selected_features]
        
        # Filter X to only include selected features
        X_train_selected = X_train_resampled[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
    
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42)
    }
    
    results = {}
    for name, clf in classifiers.items():
        try:
            # Train on resampled data
            clf.fit(X_train_selected, y_train_resampled)
            
            # Cross-validation on original data to get unbiased performance estimate
            cv_scores = cross_val_score(clf, X_scaled[:, selected_indices] if len(selected_indices) > 0 else X_scaled, y, cv=5)
            
            # Predict on test set
            y_pred = clf.predict(X_test_selected)
            
            # Store results
            results[name] = {
                'classifier': clf,
                'cv_scores': cv_scores,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'feature_importance': pd.Series(
                    clf.feature_importances_,
                    index=selected_features
                ).sort_values(ascending=False) if hasattr(clf, 'feature_importances_') else None,
                'X_test': X_test_selected,
                'y_test': y_test
            }
        except Exception as e:
            print(f"Warning: {name} classifier failed: {str(e)}")
            continue
    
    return results

def plot_results(results: Dict, condition: str, output_dir: str):
    """Plot classification results and feature importance."""
    if results is None:
        print(f"No results to plot for {condition}")
        return
    
    try:
        # Plot CV scores comparison
        plt.figure(figsize=(10, 6))
        # Convert to DataFrame with explicit index for newer pandas version
        cv_scores_df = pd.DataFrame.from_dict(
            {name: pd.Series(res['cv_scores']) for name, res in results.items()}
        )
        
        # Use newer pandas groupby syntax
        sns.boxplot(data=cv_scores_df.melt(), x='variable', y='value')
        plt.title(f'Cross-validation Scores - {condition}')
        plt.xticks(rotation=45)
        plt.xlabel('Classifier')
        plt.ylabel('Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cv_scores.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot CV scores: {str(e)}")
    
    # Plot feature importance for each applicable classifier
    for name, res in results.items():
        if res['feature_importance'] is not None:
            try:
                plt.figure(figsize=(12, 6))
                importance_df = res['feature_importance'].reset_index()
                importance_df.columns = ['Feature', 'Importance']
                
                # Use newer pandas plotting syntax
                sns.barplot(data=importance_df, x='Importance', y='Feature')
                plt.title(f'Feature Importance - {name} - {condition}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'feature_importance_{name}.png'))
                plt.close()
            except Exception as e:
                print(f"Warning: Could not plot feature importance for {name}: {str(e)}")

def classify_healthy_vs_affected(df: pd.DataFrame, output_dir: str = None) -> None:
    """Classify between Set01 (Healthy) and other sets (Affected).
    
    Args:
        df: DataFrame containing participant data
        output_dir: Directory to save results (default: None, will create a directory)
    """
    # Create binary labels (Set01 = Healthy, others = Affected)
    healthy_mask = df['SET'].str.startswith('set01')
    
    participant_data = []
    for participant in df['Participant'].unique():
        participant_df = df[df['Participant'] == participant]
        
        # Basic velocity statistics for each pressure
        pressure_stats = participant_df.pivot_table(
            index='Participant',
            columns='Pressure',
            values='Video_Median_Velocity',
            aggfunc=['mean']
        ).fillna(0)
        
        # Flatten multi-index columns
        pressure_stats.columns = [f'velocity_at_{pressure}psi' 
                                for (_, pressure) in pressure_stats.columns]
        
        # Calculate velocity response characteristics
        velocity_values = participant_df.groupby('Pressure')['Video_Median_Velocity'].mean().values
        
        # Calculate up/down velocity differences
        up_velocities = participant_df[participant_df['UpDown'] == 'U']['Video_Median_Velocity']
        down_velocities = participant_df[participant_df['UpDown'] == 'D']['Video_Median_Velocity']
        
        # Basic statistics
        stats = {
            'Participant': participant,
            'is_healthy': participant_df['SET'].iloc[0].startswith('set01'),
            
            # Key velocity features
            'baseline_velocity': velocity_values[0] if len(velocity_values) > 0 else 0,
            'max_velocity': np.max(velocity_values) if len(velocity_values) > 0 else 0,
            'velocity_range': (np.max(velocity_values) - np.min(velocity_values)) if len(velocity_values) > 0 else 0,
            'mean_velocity': np.mean(velocity_values) if len(velocity_values) > 0 else 0,
            'velocity_std': np.std(velocity_values) if len(velocity_values) > 0 else 0,
            
            # Up/Down differences
            'up_down_diff': np.mean(up_velocities) - np.mean(down_velocities) if len(up_velocities) > 0 and len(down_velocities) > 0 else 0,
            
            # Basic demographic info (as control variables)
            'Age': participant_df['Age'].iloc[0],
            # 'SYS_BP': participant_df['SYS_BP'].iloc[0] if 'SYS_BP' in participant_df else None,
            # 'DIA_BP': participant_df['DIA_BP'].iloc[0] if 'DIA_BP' in participant_df else None,
        }
        
        # Add pressure-specific velocity features
        stats.update(pressure_stats.iloc[0].to_dict())
        
        participant_data.append(stats)
    
    processed_df = pd.DataFrame(participant_data)
    
    # Handle missing values
    numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
    
    # Print class distribution
    print("\nClass distribution:")
    print(processed_df['is_healthy'].value_counts())
    
    # Prepare features and target
    feature_cols = [col for col in processed_df.columns 
                   if col not in ['Participant', 'is_healthy', 'Diabetes', 'Hypertension']]  # 'HeartDisease'
    
    X = processed_df[feature_cols].values
    y = processed_df['is_healthy'].values
    
    print("\nFeatures being used:")
    for col in feature_cols:
        print(f"- {col}")
    
    # Print correlations with healthy status
    print("\nTop 5 correlations with healthy status:")
    correlations = processed_df[feature_cols + ['is_healthy']].corr()['is_healthy']
    print(correlations.sort_values(ascending=False)[:5])
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'Classifier', 'Healthy')
        os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate classifiers
    results = evaluate_classifiers(X, y, feature_cols)
    
    if results is not None:
        # Plot results directly in the Healthy directory
        plot_results(results, 'Healthy_vs_Affected', output_dir)
        plot_auc_curves(results, 'Healthy_vs_Affected', output_dir)
        
        # Save classification reports
        report_path = os.path.join(output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            for name, res in results.items():
                f.write(f"\n{name} Classification Report:\n")
                f.write(res['classification_report'])
                f.write("\nCross-validation scores:\n")
                f.write(f"Mean: {res['cv_scores'].mean():.3f} (+/- {res['cv_scores'].std() * 2:.3f})\n")
    return 0

def classify_healthy_vs_affected_log(df: pd.DataFrame, output_dir: str = None) -> None:
    """Classify between Set01 (Healthy) and other sets (Affected).
    
    Args:
        df: DataFrame containing participant data
        output_dir: Directory to save results (default: None, will create a directory)
    """
    # Create binary labels (Set01 = Healthy, others = Affected)
    healthy_mask = df['SET'].str.startswith('set01')
    
    participant_data = []
    for participant in df['Participant'].unique():
        participant_df = df[df['Participant'] == participant]
        
        # Basic velocity statistics for each pressure
        pressure_stats = participant_df.pivot_table(
            index='Participant',
            columns='Pressure',
            values='Log_Video_Median_Velocity',
            aggfunc=['mean']
        ).fillna(0)
        
        # Flatten multi-index columns
        pressure_stats.columns = [f'log_velocity_at_{pressure}psi' 
                                for (_, pressure) in pressure_stats.columns]
        
        # Calculate velocity response characteristics
        velocity_values = participant_df.groupby('Pressure')['Log_Video_Median_Velocity'].mean().values
        
        # Calculate up/down velocity differences
        up_velocities = participant_df[participant_df['UpDown'] == 'U']['Video_Median_Velocity']
        down_velocities = participant_df[participant_df['UpDown'] == 'D']['Video_Median_Velocity']
        
        # Basic statistics
        stats = {
            'Participant': participant,
            'is_healthy': participant_df['SET'].iloc[0].startswith('set01'),
            
            # Key velocity features
            # 'baseline_velocity': velocity_values[0] if len(velocity_values) > 0 else 0,
            # 'max_velocity': np.max(velocity_values) if len(velocity_values) > 0 else 0,       # this value might be somewhat helpful but makes it slightly worse
            # 'velocity_range': (np.max(velocity_values) - np.min(velocity_values)) if len(velocity_values) > 0 else 0, # this variable makes the auc way worse
            # 'mean_velocity': np.mean(velocity_values) if len(velocity_values) > 0 else 0,
            # 'velocity_std': np.std(velocity_values) if len(velocity_values) > 0 else 0,       # this variable makes the auc way worse
            
            # Up/Down differences
            'up_down_diff': np.abs(np.mean(up_velocities) - np.mean(down_velocities)) if len(up_velocities) > 0 and len(down_velocities) > 0 else 0,
            
            # Basic demographic info (as control variables)
            'Age': participant_df['Age'].iloc[0],
            # 'SYS_BP': participant_df['SYS_BP'].iloc[0] if 'SYS_BP' in participant_df else None,
            # 'DIA_BP': participant_df['DIA_BP'].iloc[0] if 'DIA_BP' in participant_df else None,
        }
        
        # Add pressure-specific velocity features
        stats.update(pressure_stats.iloc[0].to_dict())
        
        participant_data.append(stats)
    
    processed_df = pd.DataFrame(participant_data)
    
    # Handle missing values
    numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
    
    # Print class distribution
    print("\nClass distribution:")
    print(processed_df['is_healthy'].value_counts())
    
    # Prepare features and target
    feature_cols = [col for col in processed_df.columns 
                   if col not in ['Participant', 'is_healthy', 'Diabetes', 'Hypertension']]  # 'HeartDisease'
    
    X = processed_df[feature_cols].values
    y = processed_df['is_healthy'].values
    
    print("\nFeatures being used:")
    for col in feature_cols:
        print(f"- {col}")
    
    # Print correlations with healthy status
    print("\nTop 5 correlations with healthy status:")
    correlations = processed_df[feature_cols + ['is_healthy']].corr()['is_healthy']
    print(correlations.sort_values(ascending=False)[:5])
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(cap_flow_path, 'results', 'Classifier', 'Healthy')
        os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate classifiers
    results = evaluate_classifiers(X, y, feature_cols)
    
    if results is not None:
        # Plot results directly in the Healthy directory
        plot_results(results, 'Healthy_vs_Affected', output_dir)
        plot_auc_curves(results, 'Healthy_vs_Affected', output_dir)
        plot_3D_features(processed_df, results, feature_cols,'Healthy_vs_Affected', output_dir)
        
        # Save classification reports with descriptive filename
        report_path = os.path.join(output_dir, 'log_velocity_based_healthy_vs_affected_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("Classification Report for Healthy vs. Affected using Log Velocity Features\n")
            f.write("=" * 70 + "\n\n")
            
            # Add overall summary section
            f.write("SUMMARY OF CLASSIFIER PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Classifier':<20} {'CV Score (mean)':<15} {'AUC':<10}\n")
            
            # Calculate AUC for each classifier for the summary table
            for name, res in results.items():
                # Calculate AUC
                if hasattr(res['classifier'], 'predict_proba'):
                    y_pred = res['classifier'].predict_proba(res['X_test'])[:, 1]
                else:
                    y_pred = res['classifier'].predict(res['X_test'])
                
                fpr, tpr, _ = roc_curve(res['y_test'], y_pred)
                roc_auc = auc(fpr, tpr)
                
                # Write summary line
                f.write(f"{name:<20} {res['cv_scores'].mean():.3f} ± {res['cv_scores'].std():.3f} {roc_auc:.3f}\n")
            
            f.write("\n\n")
            
            # Detailed results for each classifier
            for name, res in results.items():
                f.write(f"\n{'=' * 20} {name} {'=' * 20}\n\n")
                
                # Add AUC value
                if hasattr(res['classifier'], 'predict_proba'):
                    y_pred_prob = res['classifier'].predict_proba(res['X_test'])[:, 1]
                else:
                    y_pred_prob = res['classifier'].predict(res['X_test'])
                
                fpr, tpr, _ = roc_curve(res['y_test'], y_pred_prob)
                roc_auc = auc(fpr, tpr)
                f.write(f"AUC: {roc_auc:.3f}\n\n")
                
                # Add confusion matrix with labels
                f.write("Confusion Matrix:\n")
                cm = res['confusion_matrix']
                f.write(f"{'':>10}Predicted Negative  Predicted Positive\n")
                f.write(f"Actual Negative{cm[0][0]:>10}{cm[0][1]:>20}\n")
                f.write(f"Actual Positive{cm[1][0]:>10}{cm[1][1]:>20}\n\n")
                
                # Add standard classification report
                f.write("Classification Report:\n")
                f.write(res['classification_report'])
                
                # Add cross-validation scores
                f.write("\nCross-validation scores:\n")
                f.write(f"Mean: {res['cv_scores'].mean():.3f} (±{res['cv_scores'].std():.3f})\n")
                f.write(f"Individual scores: {', '.join([f'{score:.3f}' for score in res['cv_scores']])}\n\n")

def prepare_demographic_data() -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Load and prepare data for classification using only demographic features.
    
    Returns:
        Tuple containing:
            - DataFrame with demographic features
            - Dictionary with target variables and their X, y arrays
    """
    # Load data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Create features for each participant
    participant_data = []
    for participant in df['Participant'].unique():
        participant_df = df[df['Participant'] == participant]
        
        # Basic statistics
        stats = {
            'Participant': participant,
            
            # Demographic features only
            'Age': participant_df['Age'].iloc[0],
            'SYS_BP': participant_df['SYS_BP'].iloc[0] if 'SYS_BP' in participant_df.columns else np.nan,
            'DIA_BP': participant_df['DIA_BP'].iloc[0] if 'DIA_BP' in participant_df.columns else np.nan,
            
            # Target variables
            'Diabetes': str(participant_df['Diabetes'].iloc[0]).upper() == 'TRUE',
            'Hypertension': participant_df['Hypertension'].iloc[0] == True,
            # 'HeartDisease': participant_df['HeartDisease'].iloc[0] == True,
            'is_healthy': participant_df['SET'].iloc[0].startswith('set01') if 'SET' in participant_df.columns else False
        }
        
        participant_data.append(stats)
    
    processed_df = pd.DataFrame(participant_data)
    
    # Handle missing values
    numeric_cols = ['Age', 'SYS_BP', 'DIA_BP']
    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
    
    # Print feature names for verification
    feature_cols = ['Age', 'SYS_BP', 'DIA_BP']
    print("\nDemographic features being used:")
    for col in feature_cols:
        print(f"- {col}")
    
    # Prepare X and y for each condition
    target_dict = {}
    for condition in ['Diabetes', 'Hypertension', 'is_healthy']: # 'HeartDisease'
        X = processed_df[feature_cols].values
        y = processed_df[condition].values
        target_dict[condition] = (X, y)
    
    # Print data shape and feature info
    print(f"\nTotal samples: {len(processed_df)}")
    print("\nFeature value ranges:")
    for col in feature_cols:
        print(f"{col}:")
        print(f"  Range: {processed_df[col].min():.2f} to {processed_df[col].max():.2f}")
        print(f"  Mean: {processed_df[col].mean():.2f}")
        print(f"  Null values: {processed_df[col].isnull().sum()}")

    # Print correlation with target variables
    for condition in ['Diabetes', 'Hypertension', 'is_healthy']:  # 'HeartDisease'
        print(f"\nCorrelations with {condition}:")
        correlations = processed_df[feature_cols + [condition]].corr()[condition]
        print(correlations)
    
    return processed_df, target_dict

def prepare_individual_demographic_data(feature: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load and prepare data for classification using a single demographic feature.
    
    Args:
        feature: The demographic feature to use ('Age', 'SYS_BP', or 'DIA_BP')
        
    Returns:
        Dictionary with target variables and their X, y arrays
    """
    # Load data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Create features for each participant
    participant_data = []
    for participant in df['Participant'].unique():
        participant_df = df[df['Participant'] == participant]
        
        # Check if feature exists in the dataframe
        if feature not in participant_df.columns:
            print(f"Warning: Feature '{feature}' not found in dataframe. Available columns: {participant_df.columns.tolist()}")
            continue
        
        # Basic statistics
        stats = {
            'Participant': participant,
            
            # Single demographic feature
            feature: participant_df[feature].iloc[0] if not pd.isna(participant_df[feature].iloc[0]) else np.nan,
            
            # Target variables
            'Diabetes': str(participant_df['Diabetes'].iloc[0]).upper() == 'TRUE',
            'Hypertension': participant_df['Hypertension'].iloc[0] == True,
            #'HeartDisease': participant_df['HeartDisease'].iloc[0] == True,
            'is_healthy': participant_df['SET'].iloc[0].startswith('set01') if 'SET' in participant_df.columns else False
        }
        
        participant_data.append(stats)
    
    processed_df = pd.DataFrame(participant_data)
    
    # Check if we have any data
    if processed_df.empty:
        print(f"Error: No data available for feature '{feature}'")
        return {}
    
    # Handle missing values
    processed_df[feature] = processed_df[feature].fillna(processed_df[feature].mean())
    
    print(f"\nUsing single demographic feature: {feature}")
    print(f"Shape of processed dataframe: {processed_df.shape}")
    
    # Prepare X and y for each condition
    target_dict = {}
    for condition in ['Diabetes', 'Hypertension', 'is_healthy']: # 'HeartDisease',
        X = processed_df[[feature]].values
        y = processed_df[condition].values
        
        # Print class distribution
        print(f"\nClass distribution for {condition}:")
        print(pd.Series(y).value_counts())
        
        target_dict[condition] = (X, y)
    
    # Print data shape and feature info
    print(f"\nTotal samples: {len(processed_df)}")
    print(f"\nFeature value range for {feature}:")
    print(f"  Range: {processed_df[feature].min():.2f} to {processed_df[feature].max():.2f}")
    print(f"  Mean: {processed_df[feature].mean():.2f}")
    print(f"  Null values: {processed_df[feature].isnull().sum()}")

    # Print correlation with target variables
    for condition in ['Diabetes', 'Hypertension', 'is_healthy']: #'HeartDisease',
        print(f"\nCorrelation of {feature} with {condition}:")
        correlation = processed_df[[feature, condition]].corr().iloc[0, 1]
        print(f"  {correlation:.4f}")
    
    return target_dict

def analyze_demographic_features():
    """Analyze how well demographic features predict health conditions.

    Saves results to:
        - os.path.join(cap_flow_path, 'results', 'Classifier', 'Demographics')
    """
    print("\nStarting demographic feature analysis...")
    
    # Create output directory
    output_dir = os.path.join(cap_flow_path, 'results', 'Classifier', 'Demographics')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Analyze all demographic features together
    processed_df, target_dict = prepare_demographic_data()
    
    # Print class distribution for each condition
    print("\nClass distribution for conditions:")
    for condition in ['Diabetes', 'Hypertension', 'is_healthy']:  # 'HeartDisease'
        class_counts = processed_df[condition].value_counts()
        print(f"\n{condition}:")
        print(class_counts)
    
    # Create directory for all demographics
    all_demographics_dir = os.path.join(output_dir, 'All_Demographics')
    os.makedirs(all_demographics_dir, exist_ok=True)
    
    # Analyze each condition with all demographic features
    feature_names = ['Age', 'SYS_BP', 'DIA_BP']
    for condition, (X, y) in target_dict.items():
        print(f"\nAnalyzing {condition} with all demographic features...")
        
        # Evaluate classifiers
        results = evaluate_classifiers(X, y, feature_names)
        
        if results is not None:
            # Plot results including ROC curves
            plot_results(results, condition, all_demographics_dir)
            plot_auc_curves(results, condition, all_demographics_dir)
            
            # Save classification reports with descriptive filenames
            condition_name = 'healthy_vs_affected' if condition == 'is_healthy' else condition.lower()
            report_path = os.path.join(all_demographics_dir, f'all_demographics_{condition_name}_classification_report.txt')
            with open(report_path, 'w') as f:
                f.write(f"Classification Report for {condition} using All Demographic Features (Age, SYS_BP, DIA_BP)\n")
                f.write("=" * 80 + "\n\n")
                
                # Add overall summary section
                f.write("SUMMARY OF CLASSIFIER PERFORMANCE\n")
                f.write("-" * 30 + "\n")
                f.write(f"{'Classifier':<20} {'CV Score (mean)':<15} {'AUC':<10}\n")
                
                # Calculate AUC for each classifier for the summary table
                for name, res in results.items():
                    # Calculate AUC
                    if hasattr(res['classifier'], 'predict_proba'):
                        y_pred = res['classifier'].predict_proba(res['X_test'])[:, 1]
                    else:
                        y_pred = res['classifier'].predict(res['X_test'])
                    
                    fpr, tpr, _ = roc_curve(res['y_test'], y_pred)
                    roc_auc = auc(fpr, tpr)
                    
                    # Write summary line
                    f.write(f"{name:<20} {res['cv_scores'].mean():.3f} ± {res['cv_scores'].std():.3f} {roc_auc:.3f}\n")
                
                f.write("\n\n")
                
                # Detailed results for each classifier
                for name, res in results.items():
                    f.write(f"\n{'=' * 20} {name} {'=' * 20}\n\n")
                    
                    # Add AUC value
                    if hasattr(res['classifier'], 'predict_proba'):
                        y_pred_prob = res['classifier'].predict_proba(res['X_test'])[:, 1]
                    else:
                        y_pred_prob = res['classifier'].predict(res['X_test'])
                    
                    fpr, tpr, _ = roc_curve(res['y_test'], y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    f.write(f"AUC: {roc_auc:.3f}\n\n")
                    
                    # Add confusion matrix with labels
                    f.write("Confusion Matrix:\n")
                    cm = res['confusion_matrix']
                    f.write(f"{'':>10}Predicted Negative  Predicted Positive\n")
                    f.write(f"Actual Negative{cm[0][0]:>10}{cm[0][1]:>20}\n")
                    f.write(f"Actual Positive{cm[1][0]:>10}{cm[1][1]:>20}\n\n")
                    
                    # Add standard classification report
                    f.write("Classification Report:\n")
                    f.write(res['classification_report'])
                    
                    # Add cross-validation scores
                    f.write("\nCross-validation scores:\n")
                    f.write(f"Mean: {res['cv_scores'].mean():.3f} (±{res['cv_scores'].std():.3f})\n")
                    f.write(f"Individual scores: {', '.join([f'{score:.3f}' for score in res['cv_scores']])}\n\n")
    
    # 2. Analyze each demographic feature individually
    for feature in ['Age', 'SYS_BP', 'DIA_BP']:
        print(f"\nAnalyzing individual feature: {feature}")
        
        # Create feature-specific directory
        feature_dir = os.path.join(output_dir, feature)
        os.makedirs(feature_dir, exist_ok=True)
        
        # Prepare data for this feature
        target_dict = prepare_individual_demographic_data(feature)
        
        # Analyze each condition with this feature
        for condition, (X, y) in target_dict.items():
            print(f"\nAnalyzing {condition} with {feature}...")
            
            # Evaluate classifiers
            results = evaluate_classifiers(X, y, [feature])
            
            if results is not None:
                # Plot results including ROC curves
                plot_results(results, condition, feature_dir)
                plot_auc_curves(results, condition, feature_dir)
                
                # Save classification reports with descriptive filenames
                condition_name = 'healthy_vs_affected' if condition == 'is_healthy' else condition.lower()
                report_path = os.path.join(feature_dir, f'{feature.lower()}_{condition_name}_classification_report.txt')
                with open(report_path, 'w') as f:
                    f.write(f"Classification Report for {condition} using only {feature}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    # Add overall summary section
                    f.write("SUMMARY OF CLASSIFIER PERFORMANCE\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"{'Classifier':<20} {'CV Score (mean)':<15} {'AUC':<10}\n")
                    
                    # Calculate AUC for each classifier for the summary table
                    for name, res in results.items():
                        # Calculate AUC
                        if hasattr(res['classifier'], 'predict_proba'):
                            y_pred = res['classifier'].predict_proba(res['X_test'])[:, 1]
                        else:
                            y_pred = res['classifier'].predict(res['X_test'])
                        
                        fpr, tpr, _ = roc_curve(res['y_test'], y_pred)
                        roc_auc = auc(fpr, tpr)
                        
                        # Write summary line
                        f.write(f"{name:<20} {res['cv_scores'].mean():.3f} ± {res['cv_scores'].std():.3f} {roc_auc:.3f}\n")
                    
                    f.write("\n\n")
                    
                    # Detailed results for each classifier
                    for name, res in results.items():
                        f.write(f"\n{'=' * 20} {name} {'=' * 20}\n\n")
                        
                        # Add AUC value
                        if hasattr(res['classifier'], 'predict_proba'):
                            y_pred_prob = res['classifier'].predict_proba(res['X_test'])[:, 1]
                        else:
                            y_pred_prob = res['classifier'].predict(res['X_test'])
                        
                        fpr, tpr, _ = roc_curve(res['y_test'], y_pred_prob)
                        roc_auc = auc(fpr, tpr)
                        f.write(f"AUC: {roc_auc:.3f}\n\n")
                        
                        # Add confusion matrix with labels
                        f.write("Confusion Matrix:\n")
                        cm = res['confusion_matrix']
                        f.write(f"{'':>10}Predicted Negative  Predicted Positive\n")
                        f.write(f"Actual Negative{cm[0][0]:>10}{cm[0][1]:>20}\n")
                        f.write(f"Actual Positive{cm[1][0]:>10}{cm[1][1]:>20}\n\n")
                        
                        # Add standard classification report
                        f.write("Classification Report:\n")
                        f.write(res['classification_report'])
                        
                        # Add cross-validation scores
                        f.write("\nCross-validation scores:\n")
                        f.write(f"Mean: {res['cv_scores'].mean():.3f} (±{res['cv_scores'].std():.3f})\n")
                        f.write(f"Individual scores: {', '.join([f'{score:.3f}' for score in res['cv_scores']])}\n\n")
    
    print("\nDemographic feature analysis complete.")

def main():
    """Main function to run the classification analysis."""
    print("\nStarting health condition classification analysis...")
    
    # Create main output directory
    output_dir = os.path.join(cap_flow_path, 'results', 'Classifier')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Velocity umbrella folder for health condition classifications
    velocity_dir = os.path.join(output_dir, 'Velocity')
    os.makedirs(velocity_dir, exist_ok=True)
    
    # Load raw data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)

    # set all values for Log_Video_Median_Velocity that are lower than 0 to 0
    df['Log_Video_Median_Velocity'] = df['Log_Video_Median_Velocity'].apply(lambda x: 0 if x < 0 else x)

    # print the first row of the dataframe in full with all columns and the header 
    print(df.iloc[0])
    
    # Run demographic feature analysis
    print("\nRunning demographic feature analysis...")
    analyze_demographic_features()
    
    # Run healthy vs affected classification
    print("\nRunning Healthy vs Affected classification...")
    # Create Healthy directory inside Velocity folder
    healthy_dir = os.path.join(velocity_dir, 'Healthy')
    os.makedirs(healthy_dir, exist_ok=True)
    
    # Update classify_healthy_vs_affected_log to use the new directory
    healthy_results = classify_healthy_vs_affected_log(df, output_dir=healthy_dir)
    
    # Prepare data for condition-specific classification
    processed_df, target_dict = prepare_data_log(df)
    
    # Print class distribution for each condition
    print("\nClass distribution for specific conditions:")
    for condition in ['Diabetes', 'Hypertension']: # 'HeartDisease'
        class_counts = processed_df[condition].value_counts()
        print(f"\n{condition}:")
        print(class_counts)
    
    # Analyze each condition (excluding HeartDisease)
    for condition, (X, y) in target_dict.items():
        # Skip HeartDisease condition
        if condition == 'HeartDisease':
            print(f"\nSkipping {condition} analysis")
            continue
            
        print(f"\nAnalyzing {condition}...")
        
        feature_names = [col for col in processed_df.columns 
                        if col not in ['Participant', 'Diabetes', 'Hypertension']] # , 'HeartDisease'
        
        # Create condition-specific directory inside Velocity folder
        condition_dir = os.path.join(velocity_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)
        
        # Evaluate classifiers
        results = evaluate_classifiers(X, y, feature_names)
        
        if results is not None:
            # Plot results directly in the condition directory
            plot_results(results, condition, condition_dir)
            plot_auc_curves(results, condition, condition_dir)
            
            # Save classification reports with descriptive filenames
            report_path = os.path.join(condition_dir, f'velocity_based_{condition}_classification_report.txt')
            with open(report_path, 'w') as f:
                f.write(f"Classification Report for {condition} using Velocity Features\n")
                f.write("=" * 50 + "\n\n")
                
                # Add overall summary section
                f.write("SUMMARY OF CLASSIFIER PERFORMANCE\n")
                f.write("-" * 30 + "\n")
                f.write(f"{'Classifier':<20} {'CV Score (mean)':<15} {'AUC':<10}\n")
                
                # Calculate AUC for each classifier for the summary table
                for name, res in results.items():
                    # Calculate AUC
                    if hasattr(res['classifier'], 'predict_proba'):
                        y_pred = res['classifier'].predict_proba(res['X_test'])[:, 1]
                    else:
                        y_pred = res['classifier'].predict(res['X_test'])
                    
                    fpr, tpr, _ = roc_curve(res['y_test'], y_pred)
                    roc_auc = auc(fpr, tpr)
                    
                    # Write summary line
                    f.write(f"{name:<20} {res['cv_scores'].mean():.3f} ± {res['cv_scores'].std():.3f} {roc_auc:.3f}\n")
                
                f.write("\n\n")
                
                # Detailed results for each classifier
                for name, res in results.items():
                    f.write(f"\n{'=' * 20} {name} {'=' * 20}\n\n")
                    
                    # Add AUC value
                    if hasattr(res['classifier'], 'predict_proba'):
                        y_pred_prob = res['classifier'].predict_proba(res['X_test'])[:, 1]
                    else:
                        y_pred_prob = res['classifier'].predict(res['X_test'])
                    
                    fpr, tpr, _ = roc_curve(res['y_test'], y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    f.write(f"AUC: {roc_auc:.3f}\n\n")
                    
                    # Add confusion matrix with labels
                    f.write("Confusion Matrix:\n")
                    cm = res['confusion_matrix']
                    f.write(f"{'':>10}Predicted Negative  Predicted Positive\n")
                    f.write(f"Actual Negative{cm[0][0]:>10}{cm[0][1]:>20}\n")
                    f.write(f"Actual Positive{cm[1][0]:>10}{cm[1][1]:>20}\n\n")
                    
                    # Add standard classification report
                    f.write("Classification Report:\n")
                    f.write(res['classification_report'])
                    
                    # Add cross-validation scores
                    f.write("\nCross-validation scores:\n")
                    f.write(f"Mean: {res['cv_scores'].mean():.3f} (±{res['cv_scores'].std():.3f})\n")
                    f.write(f"Individual scores: {', '.join([f'{score:.3f}' for score in res['cv_scores']])}\n\n")
    
    # Generate 3D visualization for Healthy vs Affected classification
    # Get the log velocity-based healthy classification data
    healthy_df = pd.DataFrame([])
    
    # Need to run again to get the processed dataframe for healthy vs affected
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Extract healthy classification data
    participant_data = []
    for participant in df['Participant'].unique():
        participant_df = df[df['Participant'] == participant]
        
        # Basic velocity statistics for each pressure
        pressure_stats = participant_df.pivot_table(
            index='Participant',
            columns='Pressure',
            values='Log_Video_Median_Velocity',
            aggfunc=['mean']
        ).fillna(0)
        
        # Flatten multi-index columns
        pressure_stats.columns = [f'log_velocity_at_{pressure}psi' 
                                for (_, pressure) in pressure_stats.columns]
        
        # Calculate up/down velocity differences
        up_velocities = participant_df[participant_df['UpDown'] == 'U']['Video_Median_Velocity']
        down_velocities = participant_df[participant_df['UpDown'] == 'D']['Video_Median_Velocity']
        
        # Basic statistics
        stats = {
            'Participant': participant,
            'is_healthy': participant_df['SET'].iloc[0].startswith('set01'),
            'up_down_diff': np.mean(up_velocities) - np.mean(down_velocities) if len(up_velocities) > 0 and len(down_velocities) > 0 else 0,
        }
        
        # Add pressure-specific velocity features
        stats.update(pressure_stats.iloc[0].to_dict())
        
        participant_data.append(stats)
    
      
    return processed_df, target_dict

if __name__ == "__main__":
    processed_df, target_dict = main() 