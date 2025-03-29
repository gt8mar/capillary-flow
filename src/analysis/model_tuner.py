"""
Filename: src/analysis/model_tuner.py
--------------------------------------

This script is used to tune the hyperparameters of a Random Forest classifier.
It uses cross-validation to find the best hyperparameters and then uses repeated
train/test splits to measure the variance in AUC.

By: Marcus Forst

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)
from typing import Tuple, List
import warnings

from src.config import PATHS

###############################################################################
#                         USER-CONFIGURABLE PATHS                             #
###############################################################################
DATA_CSV_PATH = os.path.join(PATHS['cap_flow'], "summary_df_nhp_video_stats.csv")  # <-- Update this!
OUTPUT_TEXTFILE = "random_forest_results.txt"

###############################################################################
#                          DATA LOADING & PREPARATION                         #
###############################################################################
def load_data(
    target_condition: str = "Diabetes",
    use_demographics: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads the capillary flow dataset and prepares features for classification.
    
    Args:
        target_condition: Which column ('Diabetes' or 'Hypertension') to use as the classification target.
        use_demographics: Whether to include Age, Systolic BP, and Diastolic BP in the feature set.
    
    Returns:
        X: Numpy array of shape [n_samples, n_features].
        y: Binary target array of shape [n_samples].
        feature_names: List of feature column names used in X.
    """
    # Read raw CSV
    df = pd.read_csv(DATA_CSV_PATH)

    # Collect unique participants
    participants = df["Participant"].unique()
    
    processed_rows = []

    for p in participants:
        sub = df[df["Participant"] == p].copy()

        # Group by pressure to calculate pressure-specific velocities
        pressure_groups = sub.groupby('Pressure')
        
        # Create a DataFrame to hold pressure-specific velocity stats
        pressure_cols = []
        pressure_data = []
        
        for pressure, pressure_df in pressure_groups:
            # Skip if pressure is NaN
            if pd.isna(pressure):
                continue
                
            prefix = f"p{int(pressure)}_"
            pressure_cols.extend([
                f"{prefix}mean_vel",
                f"{prefix}median_vel",
                f"{prefix}min_vel",
                f"{prefix}max_vel",
                f"{prefix}std_vel",
                # f"{prefix}q25_vel",
                # f"{prefix}q75_vel",
                # f"{prefix}skew_vel",
                # f"{prefix}kurt_vel",
                # f"{prefix}range_vel"
            ])
            
            velocities = pressure_df["Video_Median_Velocity"]
            pressure_data.extend([
                velocities.mean(),
                velocities.median(),
                velocities.min(),
                velocities.max(),
                velocities.std(),
                # velocities.quantile(0.25),
                # velocities.quantile(0.75),
                # velocities.skew(),
                # velocities.kurtosis(),
                # velocities.max() - velocities.min()
            ])
        
        # Up/down difference calculation
        up_vel = sub[sub["UpDown"] == "U"]["Video_Median_Velocity"].mean() if not sub[sub["UpDown"] == "U"].empty else 0
        down_vel = sub[sub["UpDown"] == "D"]["Video_Median_Velocity"].mean() if not sub[sub["UpDown"] == "D"].empty else 0
        up_down_diff = up_vel - down_vel

        # Basic demographic columns (if exist) 
        age = sub["Age"].iloc[0] if "Age" in sub.columns else np.nan
        sbp = sub["SYS_BP"].iloc[0] if "SYS_BP" in sub.columns else np.nan
        dbp = sub["DIA_BP"].iloc[0] if "DIA_BP" in sub.columns else np.nan

        # Condition label
        if target_condition == "Diabetes":
            label = True if str(sub["Diabetes"].iloc[0]).upper() == "TRUE" else False
        elif target_condition == "Hypertension":
            label = True if str(sub["Hypertension"].iloc[0]).upper() == "TRUE" else False
        else:
            raise ValueError("target_condition must be 'Diabetes' or 'Hypertension'")

        # Basic features
        row_dict = {
            "Participant": p,
            "mean_vel": sub["Video_Median_Velocity"].mean(),
            "max_vel": sub["Video_Median_Velocity"].max(),
            "min_vel": sub["Video_Median_Velocity"].min(),
            "up_down_diff": up_down_diff,
            target_condition: label,
        }
        
        # Add pressure-specific features
        for col, val in zip(pressure_cols, pressure_data):
            row_dict[col] = val

        # Optionally add demographics
        if use_demographics:
            row_dict["Age"] = age
            row_dict["SYS_BP"] = sbp
            row_dict["DIA_BP"] = dbp

        processed_rows.append(row_dict)

    # Convert to DataFrame
    processed_df = pd.DataFrame(processed_rows)

    # Fill missing numeric data with means
    for col in processed_df.columns:
        if processed_df[col].dtype in [np.float64, np.int64]:
            processed_df[col] = processed_df[col].fillna(processed_df[col].mean())

    # Prepare the feature matrix and target vector
    feature_names = [
        c for c in processed_df.columns
        if c not in ["Participant", target_condition]
    ]
    X = processed_df[feature_names].values
    y = processed_df[target_condition].values

    return X, y, feature_names

def load_data_log(
    target_condition: str = "Diabetes",
    use_demographics: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads the capillary flow dataset and applies log transformation to velocity features.
    
    Args:
        target_condition: Which column ('Diabetes' or 'Hypertension') to use as the classification target.
        use_demographics: Whether to include Age, Systolic BP, and Diastolic BP in the feature set.
    
    Returns:
        X: Numpy array of shape [n_samples, n_features].
        y: Binary target array of shape [n_samples].
        feature_names: List of feature column names used in X.
    """
    # First load data normally
    X, y, feature_names = load_data(
        target_condition=target_condition,
        use_demographics=use_demographics
    )
    
    # Convert to DataFrame for easier column identification
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Apply log transformation to velocity columns (avoid demographics)
    velocity_cols = [col for col in X_df.columns if any(
        term in col.lower() for term in ['vel', 'velocity', '_mean', '_max', '_min', '_std']
    )]
    
    # Apply log(x + 1) transformation to handle zeros
    for col in velocity_cols:
        X_df[col] = np.log1p(X_df[col])
    
    return X_df.values, y, feature_names

###############################################################################
#                      RANDOM FOREST TRAINING & TUNING                        #
###############################################################################
def tune_classifier(
    X: np.ndarray,
    y: np.ndarray,
    classifier_type: str = "rf",
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42
):
    """
    Perform hyperparameter tuning for the specified classifier type.
    
    Args:
        X: Feature matrix of shape [n_samples, n_features].
        y: Binary target vector of shape [n_samples].
        classifier_type: Type of classifier - 'rf' (Random Forest), 'xgb' (XGBoost), 
                         'svm' (Support Vector Machine), or 'lr' (Logistic Regression)
        n_splits: Number of cross-validation folds used in the grid search.
        n_repeats: Number of repeated random train/test splits to measure AUC variability.
        random_state: Controls reproducibility.
    
    Returns:
        best_model: The best-fitted classifier from the grid search.
        all_train_aucs: List of AUCs on the training sets across repeated splits.
        all_test_aucs: List of AUCs on the testing sets across repeated splits.
        best_params: Dict of best hyperparameters found by the grid search.
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define param_grid and model based on classifier_type
    if classifier_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=random_state)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"]
        }
    elif classifier_type == "xgb":
        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=random_state)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
    elif classifier_type == "svm":
        from sklearn.svm import SVC
        model = SVC(probability=True, random_state=random_state)
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto", 0.1, 1]
        }
    elif classifier_type == "lr":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=random_state, max_iter=10000)
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"]
        }
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
    # Proceed with GridSearchCV
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        model, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0
    )
    grid_search.fit(X_scaled, y)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # 2) Evaluate with repeated train/test splits to see variance in AUC
    all_train_aucs = []
    all_test_aucs = []

    # We will do repeated random splits
    for repeat_i in range(n_repeats):
        rs = np.random.randint(10000)  # pick a random seed each time
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=rs, stratify=y
        )

        # Fit the best model from the grid search
        model = RandomForestClassifier(**best_params, random_state=rs)
        model.fit(X_train, y_train)

        # Predict proba for AUC
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]

        train_auc = roc_auc_score(y_train, y_train_prob)
        test_auc = roc_auc_score(y_test, y_test_prob)

        all_train_aucs.append(train_auc)
        all_test_aucs.append(test_auc)

    # Finally, refit the best model on the entire dataset (scaled):
    best_model.fit(X_scaled, y)

    return best_model, all_train_aucs, all_test_aucs, best_params

###############################################################################
#                         PLOTTING TRAIN/TEST AUCs                            #
###############################################################################
def plot_train_test_aucs(all_train_aucs, all_test_aucs, output_png="train_test_aucs.png"):
    """
    Plot the train/test AUC across repeated splits and save to PNG.
    """
    plt.figure()
    plt.plot(range(1, len(all_train_aucs) + 1), all_train_aucs, marker='o', label='Train AUC')
    plt.plot(range(1, len(all_test_aucs) + 1), all_test_aucs, marker='o', label='Test AUC')
    plt.title("Train/Test AUC across repeated splits")
    plt.xlabel("Split index")
    plt.ylabel("AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

###############################################################################
#                   2D PCA VISUALIZATION OF CONFUSION CATEGORIES              #
###############################################################################
def plot_2d_pca_confusion(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    output_png: str = "pca_confusion.png"
):
    """
    Fit a PCA(2D) to X, then plot points color-coded by their confusion category:
        0 -> True Negative
        1 -> False Positive
        2 -> False Negative
        3 -> True Positive
    """
    # Scale the features the same way we did before (assume standard scaling is needed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Predict
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Build confusion categories
    confusion = np.zeros(len(y), dtype=int)
    confusion[(y == 0) & (y_pred == 0)] = 0  # TN
    confusion[(y == 0) & (y_pred == 1)] = 1  # FP
    confusion[(y == 1) & (y_pred == 0)] = 2  # FN
    confusion[(y == 1) & (y_pred == 1)] = 3  # TP

    # PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    # Plot
    plt.figure()
    cm_labels = ["TN", "FP", "FN", "TP"]
    cm_colors = ["green", "orange", "red", "blue"]

    for cat_idx in [0, 1, 2, 3]:
        mask = (confusion == cat_idx)
        if np.sum(mask) == 0:
            continue
        plt.scatter(
            X_2d[mask, 0], 
            X_2d[mask, 1],
            label=cm_labels[cat_idx],
            s=40,
            edgecolors='k'
        )

    plt.title("2D PCA of data colored by confusion matrix categories")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

###############################################################################
#           3D VISUALIZATION WITH TOP 3 IMPORTANT FEATURES                    #
###############################################################################
def plot_3d_top_features(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_png: str = "top3_features_3d.png"
):
    """
    Identify the top-3 most important RF features, then plot them in 3D,
    color-coding each sample by confusion category. 
    """
    if not hasattr(model, "feature_importances_"):
        warnings.warn("Provided model does not have feature_importances_. Skipping 3D plot.")
        return

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]  # descending
    top3_idx = sorted_idx[:3]

    # If fewer than 3 features are available, just pad with repeats
    if len(feature_names) < 3:
        top3_idx = np.array([0, 1, 2])[:len(feature_names)]

    top3_features = [feature_names[i] for i in top3_idx]
    print("Top 3 features selected:", top3_features)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Predict
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Build confusion categories
    confusion = np.zeros(len(y), dtype=int)
    confusion[(y == 0) & (y_pred == 0)] = 0  # TN
    confusion[(y == 0) & (y_pred == 1)] = 1  # FP
    confusion[(y == 1) & (y_pred == 0)] = 2  # FN
    confusion[(y == 1) & (y_pred == 1)] = 3  # TP

    X_3d = X_scaled[:, top3_idx]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cm_labels = ["TN", "FP", "FN", "TP"]
    cm_colors = ["green", "orange", "red", "blue"]

    for cat_idx in [0, 1, 2, 3]:
        mask = (confusion == cat_idx)
        if np.sum(mask) == 0:
            continue
        ax.scatter(
            X_3d[mask, 0],
            X_3d[mask, 1],
            X_3d[mask, 2],
            label=cm_labels[cat_idx],
            s=40,
            edgecolors='k'
        )

    ax.set_xlabel(top3_features[0])
    ax.set_ylabel(top3_features[1] if len(top3_features) > 1 else "NA")
    ax.set_zlabel(top3_features[2] if len(top3_features) > 2 else "NA")
    ax.set_title("3D Plot of Top-3 RF Features (Confusion Categories)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

###############################################################################
#                    FEATURE IMPORTANCE PLOT                                  #
###############################################################################
def plot_feature_importance(
    model,
    feature_names: List[str],
    output_png: str = "feature_importance.png",
    top_n: int = 10
):
    """
    Plot the feature importances from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        output_png: Path to save the plot
        top_n: Number of top features to display
    """
    if not hasattr(model, "feature_importances_"):
        warnings.warn("Model does not have feature_importances_ attribute. Skipping plot.")
        return
        
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top N features
    plt.figure(figsize=(10, 6))
    n_features = min(top_n, len(feature_names))
    plt.title(f"Top {n_features} Feature Importances")
    plt.barh(range(n_features), importances[indices[:n_features]], align="center")
    plt.yticks(range(n_features), [feature_names[i] for i in indices[:n_features]])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

###############################################################################
#                                   MAIN                                      #
###############################################################################
def main(
    target_condition: str = "Diabetes", 
    use_demographics: bool = True,
    classifier_type: str = "rf",
    apply_class_balancing: bool = True,
    use_log_transform: bool = False,
    n_splits: int = 5,
    n_repeats: int = 5
):
    """
    Main pipeline to:
      1) Load data with or without demographics, with optional log transform.
      2) Apply class balancing if needed.
      3) Tune the selected classifier type for best hyperparameters.
      4) Perform repeated splits to measure train/test AUC variance.
      5) Run a full cross-validation evaluation.
      6) Print results to a text file.
      7) Generate multiple visualizations.
    """
    print(f"Starting analysis for {target_condition} using {classifier_type} classifier.")
    
    # 1) Load data with optional log transform
    if use_log_transform:
        X, y, feature_names = load_data_log(
            target_condition=target_condition,
            use_demographics=use_demographics
        )
        print("Applied log transformation to velocity features.")
    else:
        X, y, feature_names = load_data(
            target_condition=target_condition,
            use_demographics=use_demographics
        )
    
    # Check for class imbalance
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"Class distribution: {class_dist}")
    
    # 2) Apply class balancing if requested
    if apply_class_balancing:
        X, y = apply_smote(X, y, random_state=42)
    
    # 3) Tune classifier 
    best_model, all_train_aucs, all_test_aucs, best_params = tune_classifier(
        X, y,
        classifier_type=classifier_type,
        n_splits=n_splits,
        n_repeats=n_repeats
    )
    
    # 4) Run full cross-validation evaluation
    cv_metrics = cross_validate_model(best_model, X, y, n_splits=n_splits)
    
    # 5) Log results to text file
    output_filename = f"{classifier_type}_{target_condition}_results.txt"
    with open(output_filename, "w") as f:
        f.write(f"===== {classifier_type.upper()} Fine Tuning Results =====\n\n")
        f.write(f"Target Condition: {target_condition}\n")
        f.write(f"Use Demographics: {use_demographics}\n")
        f.write(f"Log Transform: {use_log_transform}\n")
        f.write(f"Class Balancing: {apply_class_balancing}\n")
        f.write(f"Best Hyperparameters from Grid Search:\n{best_params}\n\n")

        # Summaries of repeated splits
        for i, (tr_auc, te_auc) in enumerate(zip(all_train_aucs, all_test_aucs), start=1):
            f.write(f"Split #{i}:  Train AUC={tr_auc:.4f},  Test AUC={te_auc:.4f}\n")

        f.write("\n--- Summary of Repeated Splits ---\n")
        f.write(f"Mean Train AUC = {np.mean(all_train_aucs):.4f} ± {np.std(all_train_aucs):.4f}\n")
        f.write(f"Mean Test AUC  = {np.mean(all_test_aucs):.4f} ± {np.std(all_test_aucs):.4f}\n\n")
        
        f.write("\n--- Cross-Validation Results ---\n")
        f.write(f"Mean Accuracy: {cv_metrics['accuracy']:.4f}\n")
        f.write(f"Mean Precision: {cv_metrics['precision']:.4f}\n")
        f.write(f"Mean Recall: {cv_metrics['recall']:.4f}\n")
        f.write(f"Mean F1 Score: {cv_metrics['f1']:.4f}\n")
        f.write(f"Mean AUC: {cv_metrics['auc']:.4f} ± {cv_metrics['std_auc']:.4f}\n\n")
        
        # Add average confusion matrix
        avg_cm = np.mean(cv_metrics['confusion_matrices'], axis=0)
        f.write("Average Confusion Matrix (from cross-validation):\n")
        f.write(str(avg_cm) + "\n\n")
        
        # Final in-sample metrics
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        final_prob = best_model.predict_proba(X_scaled)[:, 1]
        final_pred = (final_prob >= 0.5).astype(int)
        cm = confusion_matrix(y, final_pred)
        report = classification_report(y, final_pred)

        f.write("Confusion Matrix (entire dataset):\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report (entire dataset):\n")
        f.write(report + "\n")

    # 6) Generate plots
    output_prefix = f"{classifier_type}_{target_condition}"
    
    # AUC curves plot
    plot_train_test_aucs(
        all_train_aucs, all_test_aucs,
        output_png=f"{output_prefix}_train_test_aucs.png"
    )
    
    # 2D PCA confusion plot
    plot_2d_pca_confusion(
        best_model, X, y,
        output_png=f"{output_prefix}_pca_confusion.png"
    )
    
    # 3D top features plot
    plot_3d_top_features(
        best_model, X, y, feature_names,
        output_png=f"{output_prefix}_top3_features_3d.png"
    )
    
    # Feature importance plot
    plot_feature_importance(
        best_model, feature_names,
        output_png=f"{output_prefix}_feature_importance.png"
    )

    print("All done. Results have been written to:")
    print(f"  -> {output_filename}")
    print(f"Plots generated with prefix: {output_prefix}_*.png")

def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance.
    
    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed for reproducibility
        
    Returns:
        X_resampled, y_resampled: Balanced dataset after SMOTE
    """
    try:
        from imblearn.over_sampling import SMOTE
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Original class distribution: {dict(zip(unique, counts))}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Check new class distribution
        unique, counts = np.unique(y_resampled, return_counts=True)
        print(f"Balanced class distribution: {dict(zip(unique, counts))}")
        
        return X_resampled, y_resampled
    except ImportError:
        print("Warning: imblearn not installed. Skipping SMOTE. Install with 'pip install imbalanced-learn'")
        return X, y

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
):
    """
    Perform stratified k-fold cross-validation and return detailed metrics.
    
    Args:
        model: The model to evaluate
        X: Feature matrix
        y: Target vector
        n_splits: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of metrics across folds
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, confusion_matrix
    )
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize stratified k-fold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Metrics to track
    metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 
        'f1': [], 'auc': [], 'confusion_matrices': []
    }
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred))
        metrics['auc'].append(roc_auc_score(y_test, y_prob))
        metrics['confusion_matrices'].append(confusion_matrix(y_test, y_pred))
        
    # Calculate average metrics
    avg_metrics = {
        'accuracy': np.mean(metrics['accuracy']),
        'precision': np.mean(metrics['precision']),
        'recall': np.mean(metrics['recall']),
        'f1': np.mean(metrics['f1']),
        'auc': np.mean(metrics['auc']),
        'std_auc': np.std(metrics['auc']),
        'confusion_matrices': metrics['confusion_matrices']
    }
    
    return avg_metrics

if __name__ == "__main__":
    # Example usage with different configuration options:
    main(
        target_condition="Diabetes",
        use_demographics=True,
        classifier_type="rf",  # Options: "rf", "xgb", "svm", "lr"
        apply_class_balancing=True,
        use_log_transform=False,
        n_splits=5,
        n_repeats=5
    )
