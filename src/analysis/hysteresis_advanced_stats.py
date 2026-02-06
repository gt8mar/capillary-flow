"""
Advanced statistical tests for hysteresis analysis.

This module provides additional statistical tests beyond mean/median comparisons:
- Variance tests (Levene's, Bartlett's, F-test)
- Distribution shape tests (Kolmogorov-Smirnov, Anderson-Darling)
- Effect size measures (Cohen's d, Cliff's delta)

Author: Marcus Forst
Date: November 2024
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Dict, Tuple


def calculate_variance_tests(group1: np.ndarray, group2: np.ndarray, 
                             group1_name: str = "Group 1", 
                             group2_name: str = "Group 2") -> Dict:
    """Calculate tests for differences in variance between two groups.
    
    Args:
        group1: Array of values for first group
        group2: Array of values for second group
        group1_name: Name of first group (for display)
        group2_name: Name of second group (for display)
    
    Returns:
        Dictionary with variance test results
    """
    results = {
        'group1_name': group1_name,
        'group2_name': group2_name,
        'n1': len(group1),
        'n2': len(group2),
        'var1': np.var(group1, ddof=1),
        'var2': np.var(group2, ddof=1),
        'sd1': np.std(group1, ddof=1),
        'sd2': np.std(group2, ddof=1),
    }
    
    # Levene's test (robust to non-normality)
    stat_levene, p_levene = stats.levene(group1, group2)
    results['levene_statistic'] = stat_levene
    results['levene_pvalue'] = p_levene
    results['levene_significant'] = 'Yes' if p_levene < 0.05 else 'No'
    
    # Bartlett's test (sensitive to non-normality, assumes normality)
    stat_bartlett, p_bartlett = stats.bartlett(group1, group2)
    results['bartlett_statistic'] = stat_bartlett
    results['bartlett_pvalue'] = p_bartlett
    results['bartlett_significant'] = 'Yes' if p_bartlett < 0.05 else 'No'
    
    # F-test (ratio of variances)
    f_stat = results['var1'] / results['var2'] if results['var2'] > 0 else np.nan
    # Degrees of freedom
    df1 = len(group1) - 1
    df2 = len(group2) - 1
    # Two-tailed p-value
    if not np.isnan(f_stat):
        p_f = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
    else:
        p_f = np.nan
    results['f_statistic'] = f_stat
    results['f_pvalue'] = p_f
    results['f_significant'] = 'Yes' if p_f < 0.05 else 'No'
    
    # Variance ratio
    results['variance_ratio'] = f_stat
    results['variance_ratio_interpretation'] = (
        f"Group 1 variance is {f_stat:.2f}x Group 2 variance" if not np.isnan(f_stat) else "N/A"
    )
    
    return results


def calculate_distribution_shape_tests(group1: np.ndarray, group2: np.ndarray,
                                       group1_name: str = "Group 1",
                                       group2_name: str = "Group 2") -> Dict:
    """Calculate tests for differences in distribution shape between two groups.
    
    Args:
        group1: Array of values for first group
        group2: Array of values for second group
        group1_name: Name of first group (for display)
        group2_name: Name of second group (for display)
    
    Returns:
        Dictionary with distribution shape test results
    """
    results = {
        'group1_name': group1_name,
        'group2_name': group2_name,
        'n1': len(group1),
        'n2': len(group2),
    }
    
    # Kolmogorov-Smirnov test (tests if distributions are different)
    stat_ks, p_ks = stats.ks_2samp(group1, group2)
    results['ks_statistic'] = stat_ks
    results['ks_pvalue'] = p_ks
    results['ks_significant'] = 'Yes' if p_ks < 0.05 else 'No'
    
    # Skewness for each group
    skew1 = stats.skew(group1)
    skew2 = stats.skew(group2)
    results['skew1'] = skew1
    results['skew2'] = skew2
    results['skew_difference'] = abs(skew1 - skew2)
    
    # Kurtosis for each group
    kurt1 = stats.kurtosis(group1)
    kurt2 = stats.kurtosis(group2)
    results['kurtosis1'] = kurt1
    results['kurtosis2'] = kurt2
    results['kurtosis_difference'] = abs(kurt1 - kurt2)
    
    # Shapiro-Wilk test for normality (each group)
    if len(group1) >= 3:
        stat_sw1, p_sw1 = stats.shapiro(group1)
        results['shapiro_stat1'] = stat_sw1
        results['shapiro_pvalue1'] = p_sw1
        results['group1_normal'] = 'Yes' if p_sw1 > 0.05 else 'No'
    else:
        results['shapiro_stat1'] = np.nan
        results['shapiro_pvalue1'] = np.nan
        results['group1_normal'] = 'N/A (too few samples)'
    
    if len(group2) >= 3:
        stat_sw2, p_sw2 = stats.shapiro(group2)
        results['shapiro_stat2'] = stat_sw2
        results['shapiro_pvalue2'] = p_sw2
        results['group2_normal'] = 'Yes' if p_sw2 > 0.05 else 'No'
    else:
        results['shapiro_stat2'] = np.nan
        results['shapiro_pvalue2'] = np.nan
        results['group2_normal'] = 'N/A (too few samples)'
    
    return results


def calculate_effect_sizes(group1: np.ndarray, group2: np.ndarray,
                          group1_name: str = "Group 1",
                          group2_name: str = "Group 2") -> Dict:
    """Calculate effect size measures for group differences.
    
    Args:
        group1: Array of values for first group
        group2: Array of values for second group
        group1_name: Name of first group (for display)
        group2_name: Name of second group (for display)
    
    Returns:
        Dictionary with effect size measures
    """
    results = {
        'group1_name': group1_name,
        'group2_name': group2_name,
        'n1': len(group1),
        'n2': len(group2),
    }
    
    # Cohen's d (standardized mean difference)
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    n1 = len(group1)
    n2 = len(group2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan
    
    results['cohens_d'] = cohens_d
    results['mean_difference'] = mean1 - mean2
    results['pooled_std'] = pooled_std
    
    # Interpret Cohen's d
    if not np.isnan(cohens_d):
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "Negligible"
        elif abs_d < 0.5:
            interpretation = "Small"
        elif abs_d < 0.8:
            interpretation = "Medium"
        else:
            interpretation = "Large"
        results['cohens_d_interpretation'] = interpretation
    else:
        results['cohens_d_interpretation'] = "N/A"
    
    # Cliff's Delta (non-parametric effect size)
    # Counts how often values in group1 are larger than values in group2
    n_greater = sum(1 for x in group1 for y in group2 if x > y)
    n_less = sum(1 for x in group1 for y in group2 if x < y)
    cliffs_delta = (n_greater - n_less) / (n1 * n2)
    
    results['cliffs_delta'] = cliffs_delta
    
    # Interpret Cliff's Delta
    abs_delta = abs(cliffs_delta)
    if abs_delta < 0.147:
        interpretation = "Negligible"
    elif abs_delta < 0.33:
        interpretation = "Small"
    elif abs_delta < 0.474:
        interpretation = "Medium"
    else:
        interpretation = "Large"
    results['cliffs_delta_interpretation'] = interpretation
    
    return results


def comprehensive_comparison(group1: np.ndarray, group2: np.ndarray,
                            group1_name: str = "Group 1",
                            group2_name: str = "Group 2") -> pd.DataFrame:
    """Perform comprehensive statistical comparison between two groups.
    
    Includes:
    - Location tests (mean/median)
    - Variance tests
    - Distribution shape tests
    - Effect sizes
    
    Args:
        group1: Array of values for first group
        group2: Array of values for second group
        group1_name: Name of first group
        group2_name: Name of second group
    
    Returns:
        DataFrame with all test results
    """
    results = []
    
    # Location tests
    stat_mw, p_mw = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    stat_t, p_t = stats.ttest_ind(group1, group2)
    
    results.append({
        'test_category': 'Location',
        'test_name': 'Mann-Whitney U',
        'statistic': stat_mw,
        'p_value': p_mw,
        'significant': 'Yes' if p_mw < 0.05 else 'No',
        'interpretation': 'Tests if medians differ'
    })
    
    results.append({
        'test_category': 'Location',
        'test_name': 'Independent t-test',
        'statistic': stat_t,
        'p_value': p_t,
        'significant': 'Yes' if p_t < 0.05 else 'No',
        'interpretation': 'Tests if means differ'
    })
    
    # Variance tests
    var_results = calculate_variance_tests(group1, group2, group1_name, group2_name)
    
    results.append({
        'test_category': 'Variance',
        'test_name': "Levene's test",
        'statistic': var_results['levene_statistic'],
        'p_value': var_results['levene_pvalue'],
        'significant': var_results['levene_significant'],
        'interpretation': 'Tests if variances differ (robust)'
    })
    
    results.append({
        'test_category': 'Variance',
        'test_name': "Bartlett's test",
        'statistic': var_results['bartlett_statistic'],
        'p_value': var_results['bartlett_pvalue'],
        'significant': var_results['bartlett_significant'],
        'interpretation': 'Tests if variances differ (assumes normality)'
    })
    
    results.append({
        'test_category': 'Variance',
        'test_name': 'F-test',
        'statistic': var_results['f_statistic'],
        'p_value': var_results['f_pvalue'],
        'significant': var_results['f_significant'],
        'interpretation': f"Variance ratio: {var_results['variance_ratio']:.2f}"
    })
    
    # Distribution shape tests
    shape_results = calculate_distribution_shape_tests(group1, group2, group1_name, group2_name)
    
    results.append({
        'test_category': 'Distribution',
        'test_name': 'Kolmogorov-Smirnov',
        'statistic': shape_results['ks_statistic'],
        'p_value': shape_results['ks_pvalue'],
        'significant': shape_results['ks_significant'],
        'interpretation': 'Tests if distributions differ overall'
    })
    
    # Effect sizes
    effect_results = calculate_effect_sizes(group1, group2, group1_name, group2_name)
    
    results.append({
        'test_category': 'Effect Size',
        'test_name': "Cohen's d",
        'statistic': effect_results['cohens_d'],
        'p_value': np.nan,
        'significant': 'N/A',
        'interpretation': f"{effect_results['cohens_d_interpretation']} effect"
    })
    
    results.append({
        'test_category': 'Effect Size',
        'test_name': "Cliff's Delta",
        'statistic': effect_results['cliffs_delta'],
        'p_value': np.nan,
        'significant': 'N/A',
        'interpretation': f"{effect_results['cliffs_delta_interpretation']} effect"
    })
    
    return pd.DataFrame(results)


def analyze_under30_comprehensive(processed_df: pd.DataFrame, use_absolute: bool = False) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    """Comprehensive analysis of under-30 vs over-30 comparison.
    
    Args:
        processed_df: DataFrame with hysteresis data
        use_absolute: Whether to use absolute values
    
    Returns:
        Tuple of (summary_df, variance_dict, shape_dict, effect_dict)
    """
    # Prepare data
    plot_df = processed_df.copy()
    if use_absolute:
        plot_df['up_down_diff'] = plot_df['up_down_diff'].abs()
    
    plot_df = plot_df.dropna(subset=['up_down_diff', 'Age'])
    
    # Split groups
    under_30 = plot_df[plot_df['Age'] < 30]['up_down_diff'].values
    over_30 = plot_df[plot_df['Age'] >= 30]['up_down_diff'].values
    
    # Comprehensive comparison
    summary_df = comprehensive_comparison(under_30, over_30, "<30", "30+")
    
    # Detailed results
    variance_results = calculate_variance_tests(under_30, over_30, "<30", "30+")
    shape_results = calculate_distribution_shape_tests(under_30, over_30, "<30", "30+")
    effect_results = calculate_effect_sizes(under_30, over_30, "<30", "30+")
    
    return summary_df, variance_results, shape_results, effect_results

