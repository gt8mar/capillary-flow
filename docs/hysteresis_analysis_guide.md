# Hysteresis Analysis Statistical Testing Guide

This guide explains how to calculate and visualize statistical significance in hysteresis analysis using the updated `hysterisis.py` module.

## Overview

The hysteresis analysis now includes:
1. **P-value calculation** for group differences in velocity hysteresis
2. **Visual annotations** with asterisks on plots for significant differences
3. **CSV export** of all statistical test results

## New Functions

### `calculate_hysteresis_pvalues()`

Calculates p-values for differences in hysteresis between groups.

**Parameters:**
- `processed_df` (pd.DataFrame): DataFrame containing participant data with `up_down_diff` values
- `use_absolute` (bool, default=False): Whether to use absolute values of hysteresis

**Returns:**
- pd.DataFrame with columns:
  - `grouping_factor`: The variable used for grouping (e.g., 'Age Group', 'Diabetes')
  - `group1`, `group2`: The groups being compared
  - `test_type`: Statistical test used ('Mann-Whitney U' or 'Kruskal-Wallis')
  - `statistic`: Test statistic value
  - `p_value`: P-value from the test
  - `n1`, `n2`: Sample sizes
  - `significant`: 'Yes' if p < 0.05, 'No' otherwise

**Statistical Tests Used:**
- **Mann-Whitney U test**: For comparing two groups (non-parametric)
- **Kruskal-Wallis test**: For comparing multiple groups (e.g., age groups)
  - Followed by pairwise Mann-Whitney U tests for post-hoc analysis

**Example:**
```python
from src.analysis.hysterisis import calculate_hysteresis_pvalues

# Calculate p-values
pvalues_df = calculate_hysteresis_pvalues(processed_df, use_absolute=False)

# Filter significant results
significant = pvalues_df[pvalues_df['p_value'] < 0.05]
print(significant)
```

### Updated `plot_up_down_diff_boxplots()`

Creates boxplots with significance annotations for hysteresis grouped by different factors.

**New Features:**
- Automatically calculates p-values for each grouping factor
- Adds asterisk annotations for significant comparisons:
  - `*` for p < 0.05
  - `**` for p < 0.01
  - `***` for p < 0.001
  - `ns` for non-significant (p ≥ 0.05)
- For multi-group comparisons (age), shows up to 3 most significant pairs
- Displays brackets connecting compared groups

**Example:**
```python
from src.analysis.hysterisis import plot_up_down_diff_boxplots

# Create plots with significance annotations
plot_up_down_diff_boxplots(
    processed_df, 
    use_absolute=True,
    output_dir='results/Hysteresis',
    use_log_velocity=False
)
```

## Running the Full Analysis

The main analysis pipeline now automatically:
1. Calculates velocity hysteresis for all participants
2. Computes p-values for group differences
3. Saves results to CSV
4. Generates annotated plots
5. Prints a significance summary

**Run the full analysis:**
```bash
python src/analysis/hysterisis.py
```

**Output files:**
- `hysteresis_pvalues.csv`: All statistical test results
- `hysteresis_by_age.png`: Plot by age groups
- `hysteresis_by_health_status.png`: Plot by health status
- `hysteresis_by_diabetes.png`: Plot by diabetes status
- `hysteresis_by_hypertension.png`: Plot by hypertension status
- `abs_hysteresis_*.png`: Corresponding plots for absolute hysteresis

## Interpreting Results

### P-value Significance Levels
- **p < 0.001**: Highly significant (***) - Very strong evidence of difference
- **p < 0.01**: Very significant (**) - Strong evidence of difference
- **p < 0.05**: Significant (*) - Evidence of difference
- **p ≥ 0.05**: Not significant (ns) - Insufficient evidence of difference

### Sample Output

```
STATISTICAL SIGNIFICANCE SUMMARY
======================================================================

Found 3 significant differences (p < 0.05):

Regular Hysteresis - Age Group: <30 vs 70+
  p = 0.0440 * (Mann-Whitney U)
  Sample sizes: n1=14, n2=13

Absolute Hysteresis - Age Group: <30 vs 30-49
  p = 0.0461 * (Mann-Whitney U)
  Sample sizes: n1=14, n2=7

Absolute Hysteresis - Age Group: <30 vs 50-59
  p = 0.0182 * (Mann-Whitney U)
  Sample sizes: n1=14, n2=13
```

## CSV Format

The `hysteresis_pvalues.csv` file contains the following columns:

| Column | Description |
|--------|-------------|
| grouping_factor | Variable used for grouping |
| group1 | First group in comparison |
| group2 | Second group in comparison |
| test_type | Statistical test used |
| statistic | Test statistic value |
| p_value | P-value from statistical test |
| n1 | Sample size of group 1 |
| n2 | Sample size of group 2 |
| significant | Yes/No for p < 0.05 |
| analysis_type | Regular or Absolute Hysteresis |

## Grouping Factors Analyzed

The analysis tests differences across the following grouping factors:

1. **Age Group**: <30, 30-49, 50-59, 60-69, 70+
2. **Health Status**: Healthy (set01) vs Affected (other sets)
3. **Diabetes**: No Diabetes vs Diabetes
4. **Hypertension**: No Hypertension vs Hypertension

## Technical Notes

### Why Mann-Whitney U Test?
- Non-parametric test (doesn't assume normal distribution)
- Robust for small sample sizes
- Appropriate for continuous outcome variable (hysteresis)
- Commonly used in biomedical research

### Why Kruskal-Wallis Test?
- Non-parametric equivalent of one-way ANOVA
- Tests if multiple groups come from same distribution
- Used for age groups (5 categories)
- If significant, followed by pairwise Mann-Whitney U tests

### Multiple Comparison Correction
The current implementation does **not** apply correction for multiple comparisons (e.g., Bonferroni). Consider applying correction if interpreting multiple tests simultaneously:

```python
from statsmodels.stats.multitest import multipletests

# Apply Bonferroni correction
pvalues = results_df['p_value'].values
rejected, corrected_p, _, _ = multipletests(pvalues, alpha=0.05, method='bonferroni')
results_df['p_value_corrected'] = corrected_p
results_df['significant_corrected'] = rejected
```

## Customization

### Adding New Grouping Factors

To add a new grouping factor, modify the `grouping_factors` list in `calculate_hysteresis_pvalues()`:

```python
grouping_factors = [
    # ... existing factors ...
    {
        'name': 'BMI Category',
        'column': 'BMI',
        'is_categorical': False,
        'bins': [0, 18.5, 25, 30, 100],
        'labels': ['Underweight', 'Normal', 'Overweight', 'Obese']
    }
]
```

### Adjusting Significance Threshold

To use a different significance level (e.g., p < 0.01):

```python
# In the code, replace 0.05 with your threshold
significant_tests = all_pvalues[all_pvalues['p_value'] < 0.01]
```

## Best Practices

1. **Check sample sizes**: Small groups (n < 5) may not have sufficient power
2. **Consider effect size**: Statistical significance ≠ clinical significance
3. **Report all tests**: Include both significant and non-significant results
4. **Visual inspection**: Always examine boxplots alongside p-values
5. **Multiple testing**: Consider correction if performing many tests

## Troubleshooting

### No significant differences found
- Check if sample sizes are adequate (n ≥ 5 per group recommended)
- Verify data quality and outliers
- Consider using different grouping schemes
- May indicate true lack of difference

### Too many significant comparisons
- Consider multiple comparison correction
- Check for confounding variables
- Verify data processing steps

## References

- Mann-Whitney U Test: Wilcoxon rank-sum test for independent samples
- Kruskal-Wallis Test: Non-parametric alternative to one-way ANOVA
- Statistical Annotation Guidelines: Follow APA or field-specific standards

## Questions?

For questions or issues with the hysteresis analysis, please:
1. Check the CSV output for detailed test results
2. Examine the console output for warnings
3. Verify your data meets the assumptions for non-parametric tests

---

**Last Updated**: November 2024  
**Module**: `src/analysis/hysterisis.py`  
**Author**: Marcus Forst

