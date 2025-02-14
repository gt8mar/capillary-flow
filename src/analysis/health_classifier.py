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

# Get the hostname and set paths like in fit_comparison.py
hostname = platform.node()
computer_paths = {
    "LAPTOP-I5KTBOR3": {
        'cap_flow': 'C:\\Users\\gt8ma\\capillary-flow',
    },
    "Quake-Blood": {
        'cap_flow': "C:\\Users\\gt8mar\\capillary-flow",
    },
}
default_paths = {'cap_flow': "/hpc/projects/capillary-flow"}
paths = computer_paths.get(hostname, default_paths)
cap_flow_path = paths['cap_flow']

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
    print("\nHeartDisease values:")
    print(df['HeartDisease'].value_counts(dropna=False))
    
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
            'SYS_BP': participant_df['SYS_BP'].iloc[0] if 'SYS_BP' in participant_df else None,
            'DIA_BP': participant_df['DIA_BP'].iloc[0] if 'DIA_BP' in participant_df else None,
            
            # Target variables
            'Diabetes': str(participant_df['Diabetes'].iloc[0]).upper() == 'TRUE',
            'Hypertension': participant_df['Hypertension'].iloc[0] == True,
            'HeartDisease': participant_df['HeartDisease'].iloc[0] == True
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
                   if col not in ['Participant', 'Diabetes', 'Hypertension', 'HeartDisease']]
    print("\nFeatures being used:")
    for col in feature_cols:
        print(f"- {col}")
    
    # Prepare X and y for each condition
    target_dict = {}
    for condition in ['Diabetes', 'Hypertension', 'HeartDisease']:
        X = processed_df[feature_cols].values
        y = processed_df[condition].values
        target_dict[condition] = (X, y)
    
    # Print data shape and feature info
    print(f"\nTotal samples: {len(processed_df)}")
    print("\nFeature value ranges:")
    for col in processed_df.columns:
        if col not in ['Participant', 'Diabetes', 'Hypertension', 'HeartDisease']:
            print(f"{col}:")
            print(f"  Range: {processed_df[col].min():.2f} to {processed_df[col].max():.2f}")
            print(f"  Mean: {processed_df[col].mean():.2f}")
            print(f"  Null values: {processed_df[col].isnull().sum()}")

    # Print correlation with target variables
    for condition in ['Diabetes', 'Hypertension', 'HeartDisease']:
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
    source_sans = FontProperties(fname='C:\\Users\\gt8ma\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
    
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
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontproperties=source_sans)
    plt.ylabel('True Positive Rate', fontproperties=source_sans)
    plt.title(f'ROC Curves for {condition} Classification', fontproperties=source_sans)
    plt.legend(loc="lower right", prop=source_sans)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, condition, 'roc_curves.png'), dpi=400, bbox_inches='tight')
    plt.close()

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
            cv_scores = cross_val_score(clf, X_scaled[:, selected_indices], y, cv=5)
            
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
        
    # Create condition-specific directory
    condition_dir = os.path.join(output_dir, condition)
    os.makedirs(condition_dir, exist_ok=True)
    
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
    plt.savefig(os.path.join(condition_dir, 'cv_scores.png'))
    plt.close()
    
    # Plot feature importance for each applicable classifier
    for name, res in results.items():
        if res['feature_importance'] is not None:
            plt.figure(figsize=(12, 6))
            importance_df = res['feature_importance'].head(10).reset_index()
            importance_df.columns = ['Feature', 'Importance']
            
            # Use newer pandas plotting syntax
            sns.barplot(data=importance_df, x='Importance', y='Feature')
            plt.title(f'Top 10 Feature Importance - {name} - {condition}')
            plt.tight_layout()
            plt.savefig(os.path.join(condition_dir, f'feature_importance_{name}.png'))
            plt.close()

def classify_healthy_vs_affected(df: pd.DataFrame) -> None:
    """Classify between Set01 (Healthy) and other sets (Affected).
    
    Args:
        df: DataFrame containing participant data
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
            'SYS_BP': participant_df['SYS_BP'].iloc[0] if 'SYS_BP' in participant_df else None,
            'DIA_BP': participant_df['DIA_BP'].iloc[0] if 'DIA_BP' in participant_df else None,
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
                   if col not in ['Participant', 'is_healthy', 'Diabetes', 'Hypertension', 'HeartDisease']]
    
    X = processed_df[feature_cols].values
    y = processed_df['is_healthy'].values
    
    print("\nFeatures being used:")
    for col in feature_cols:
        print(f"- {col}")
    
    # Print correlations with healthy status
    print("\nTop 5 correlations with healthy status:")
    correlations = processed_df[feature_cols + ['is_healthy']].corr()['is_healthy']
    print(correlations.sort_values(ascending=False)[:5])
    
    # Create output directory
    output_dir = os.path.join(cap_flow_path, 'results', 'Classifier', 'Healthy')
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate classifiers
    results = evaluate_classifiers(X, y, feature_cols)
    
    if results is not None:
        # Plot results
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

def main():
    """Main function to run the classification analysis."""
    print("\nStarting health condition classification analysis...")
    
    # Create output directory
    output_dir = os.path.join(cap_flow_path, 'results', 'Classifier')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # print the first row of the dataframe in full with all columns and the header 
    print(df.iloc[0])
    
    # Run healthy vs affected classification
    print("\nRunning Healthy vs Affected classification...")
    classify_healthy_vs_affected(df)
    
    # Prepare data for condition-specific classification
    processed_df, target_dict = prepare_data()
    
    # Print class distribution for each condition
    print("\nClass distribution for specific conditions:")
    for condition in ['Diabetes', 'Hypertension', 'HeartDisease']:
        class_counts = processed_df[condition].value_counts()
        print(f"\n{condition}:")
        print(class_counts)
    
    # Analyze each condition
    for condition, (X, y) in target_dict.items():
        print(f"\nAnalyzing {condition}...")
        
        feature_names = [col for col in processed_df.columns 
                        if col not in ['Participant', 'Diabetes', 'Hypertension', 'HeartDisease']]
        
        # Evaluate classifiers
        results = evaluate_classifiers(X, y, feature_names)
        
        if results is not None:
            # Plot results including ROC curves
            plot_results(results, condition, output_dir)
            plot_auc_curves(results, condition, output_dir)
            
            # Save classification reports
            report_path = os.path.join(output_dir, condition, 'classification_report.txt')
            with open(report_path, 'w') as f:
                for name, res in results.items():
                    f.write(f"\n{name} Classification Report:\n")
                    f.write(res['classification_report'])
                    f.write("\nCross-validation scores:\n")
                    f.write(f"Mean: {res['cv_scores'].mean():.3f} (+/- {res['cv_scores'].std() * 2:.3f})\n")
    
    return processed_df, target_dict

if __name__ == "__main__":
    processed_df, target_dict = main() 