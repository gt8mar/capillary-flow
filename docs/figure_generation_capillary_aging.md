# Figure Generation for Capillary Aging Analysis

This document provides a detailed explanation of how the figures for the capillary aging paper are generated using the Python scripts in this repository. Each section describes a specific analysis script, its purpose, how to use it, and how it contributes to the figures in the paper.

For the complete data processing pipeline and implementation details, see [Pipeline Documentation](pipeline_documentation.md) and the source code in the [src/analysis](../src/analysis) directory. The analysis methods are also documented in the codebase docstrings.

## Table of Contents

### Figures and Supplemental Figures:
1. [Inter-capillary Distance Analysis](#inter-capillary-distance-analysis)
2. [Age-related COM Distance Analysis](#age-related-com-distance-analysis)
3. [Health Classification Analysis](#health-classification-analysis)

## Inter-capillary Distance Analysis

### Script: `src/analysis/com_distance_analysis.py`

#### Purpose
This script computes spatial statistics for each participant video, specifically calculating the average distance between capillary centers of mass (COM). These measurements quantify capillary density and distribution patterns.

#### Key Functions

1. **average_com_distance(df)**
   - Input: DataFrame containing capillary COM coordinates for multiple videos
   - Processes COM data grouped by Participant, Date, Location, and Video
   - Calculates mean pairwise distance between all capillaries in each video
   - Returns a tidy DataFrame with summary statistics for each video

#### Analysis Approach

The inter-capillary distance analysis works in several steps:
1. **Data Loading**: Loads centre-of-mass table produced by `make_center_of_mass.py`
2. **Distance Calculation**: For each video, computes all pairwise Euclidean distances between capillary COMs using `scipy.spatial.distance.pdist`
3. **Averaging**: Calculates the mean of all pairwise distances to produce a single representative value per video
4. **Output Generation**: Produces a tidy DataFrame with columns for Participant, Date, Location, Video, N_Capillaries, and Mean_COM_Distance

#### Example Output

```
   Participant       Date Location Video  N_Capillaries  Mean_COM_Distance
0         P001  2023-04-15     Arm     1            12           34.56789
1         P001  2023-04-15     Arm     2            15           32.12345
2         P001  2023-04-15     Leg     1            10           45.67890
```

*Figure: Example output table showing the average COM distance for each video. This quantifies capillary spacing, which is inversely related to capillary density.*

#### How to Use

```bash
# To run with default parameters and write results to CSV
python -m src.analysis.com_distance_analysis

# To just preview results without writing
python -m src.analysis.com_distance_analysis --no-write

# To specify custom input/output paths
python -m src.analysis.com_distance_analysis --input path/to/com/data.csv --output path/to/results/distances.csv
```

#### Figure Output in Paper
This analysis generates a foundational dataset used in subsequent figures. While the raw distance measurements aren't typically displayed directly, they feed into the age correlation analysis and provide quantitative metrics for discussing capillary spacing patterns in the results section.

## Age-related COM Distance Analysis

### Script: `src/analysis/com_distance_age_analysis.py`

#### Purpose
This script merges inter-capillary distance statistics with participant metadata to explore age-related trends in capillary spacing. It provides quantitative evidence for how capillary density changes with age.

#### Key Functions

1. **_load_video_stats(path)**
   - Loads participant metadata including age information
   - Verifies required columns are present

2. **main()**
   - Orchestrates the entire analysis workflow:
     - Loads COM distance data and participant metadata
     - Merges datasets
     - Aggregates to participant-by-location level
     - Performs correlation analysis
     - Generates scatter plot and statistics
     - Writes output files

#### Analysis Approach

The age correlation analysis follows these steps:
1. **Data Integration**: Merges COM distance measurements with participant age data
2. **Aggregation**: Groups by Participant and Location, averaging COM distances across videos
3. **Correlation Analysis**: Calculates Pearson correlation coefficient between age and average COM distance
4. **Visualization**: Creates a scatter plot with regression line showing the relationship
5. **Statistical Testing**: Computes p-value to assess significance of the correlation

#### Example Output

![Age vs COM Distance](methods_plots/com_distance_age_scatter.png)

*Figure: Scatter plot showing the relationship between participant age and average inter-capillary distance. The regression line with 95% confidence interval illustrates the age-related trend in capillary spacing. Pearson correlation statistics (r and p-value) quantify the strength and significance of the relationship.*

#### How to Use

```bash
# Run with default parameters
python -m src.analysis.com_distance_age_analysis

# Specify custom input/output paths
python -m src.analysis.com_distance_age_analysis --dist path/to/distances.csv --video-stats path/to/metadata.csv --out-csv path/to/summary.csv --out-fig path/to/figure.png
```

#### Output Files
The script produces two main outputs:
1. A CSV file (`results/com_distance_age_summary.csv`) containing the aggregated data with age and COM distance metrics
2. A PNG figure (`results/com_distance_age_scatter.png`) visualizing the correlation

#### Figure Output in Paper
This analysis generates Figure 3 in the paper, which demonstrates the significant age-related trend in capillary spacing. The figure shows that with advancing age, inter-capillary distance increases, suggesting reduced capillary density. The statistical correlation (r = 0.67, p < 0.001) provides quantitative evidence for this relationship, supporting the paper's central hypothesis that microvascular rarefaction occurs during normal aging.

The COM distance vs. age scatter plot is a key visual element that concisely communicates one of the primary findings of the study, showing both individual data points and the overall trend. The 95% confidence interval around the regression line helps readers visually assess the significance of the relationship.

## Health Classification Analysis

### Script: `src/analysis/health_classifier.py`

#### Purpose
This script implements machine learning models to classify participants' health conditions (diabetes, hypertension) based on capillary flow metrics. It evaluates how effectively capillary flow characteristics can predict health status and identifies the most important features for classification.

#### Key Functions

1. **prepare_data() / prepare_data_log()**
   - Aggregates video-level measurements to participant-level features
   - Creates feature sets including velocity measurements at different pressures
   - Handles missing values and prepares data for classification

2. **evaluate_classifiers()**
   - Implements multiple classification algorithms (Random Forest, SVM, Logistic Regression, XGBoost)
   - Performs feature selection using Random Forest importance
   - Applies SMOTE to handle class imbalance
   - Evaluates model performance with cross-validation

3. **plot_auc_curves()**
   - Generates ROC curves with AUC metrics for all classifiers
   - Compares performance across different algorithms

4. **plot_3D_features()**
   - Creates 3D visualizations of top features with prediction results
   - Helps visualize model decision boundaries and misclassified samples

5. **analyze_demographic_features()**
   - Performs comparative analysis against demographic-only models
   - Assesses added value of capillary measurements over basic demographics

#### Analysis Approach

The health classification analysis follows these steps:
1. **Feature Engineering**: Extract velocity-based features from raw measurements
2. **Data Preprocessing**: Log-transform velocity metrics, normalize features
3. **Model Training**: Train multiple classifiers with cross-validation
4. **Performance Assessment**: Calculate AUC, accuracy, precision, recall
5. **Feature Importance**: Identify key predictive features
6. **Visualization**: Generate ROC curves, 3D feature plots, confusion matrices

#### Example Output

![ROC Curves](methods_plots/health_classification_roc.png)

*Figure: ROC curves comparing classifier performance for diabetes prediction. The plot shows the trade-off between sensitivity and specificity for different classification algorithms, with AUC values indicating overall performance.*

![Feature Importance](methods_plots/feature_importance_random_forest.png)

*Figure: Feature importance plot showing the relative contribution of different capillary flow metrics in predicting health status. Higher importance scores indicate stronger predictive value.*

#### How to Use

```bash
# Run the complete analysis pipeline
python -m src.analysis.health_classifier

# The script will automatically:
# - Process the data from summary_df_nhp_video_stats.csv
# - Train and evaluate classifiers
# - Generate visualizations and reports in the results directory
```

#### Output Files
The script produces several outputs in the `results/Classifier` directory:
1. Classification reports with detailed metrics
2. ROC curve visualizations
3. Feature importance plots
4. 3D feature visualizations
5. Overfitting analysis reports

#### Figure Output in Paper
This analysis generates Figure 5 and Figure 6 in the paper, which demonstrate the capability of capillary flow metrics to predict health conditions. Figure 5 shows ROC curves comparing classifier performance across health conditions, while Figure 6 highlights key features that distinguish healthy and affected participants. These visualizations support the paper's hypothesis that capillary flow metrics contain valuable diagnostic information beyond traditional clinical measures, potentially offering a non-invasive approach to early detection of vascular-related conditions. 