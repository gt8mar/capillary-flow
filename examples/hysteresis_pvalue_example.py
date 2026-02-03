"""
Example script demonstrating hysteresis p-value calculation and visualization.

This script shows how to:
1. Calculate velocity hysteresis for participants
2. Compute p-values for group differences
3. Generate annotated plots with significance markers
4. Export results to CSV

Usage:
    python examples/hysteresis_pvalue_example.py
"""

import os
import sys
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.analysis.hysteresis import (
    calculate_velocity_hysteresis,
    calculate_hysteresis_pvalues,
    plot_up_down_diff_boxplots
)
from src.config import PATHS

def main():
    """Run example analysis."""
    print("\n" + "="*70)
    print("HYSTERESIS P-VALUE ANALYSIS EXAMPLE")
    print("="*70 + "\n")
    
    # Set up paths
    cap_flow_path = PATHS['cap_flow']
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    output_dir = os.path.join(cap_flow_path, 'examples', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from: {data_filepath}")
    df = pd.read_csv(data_filepath)
    print(f"Loaded {len(df)} rows, {len(df['Participant'].unique())} participants\n")
    
    # Step 1: Calculate hysteresis
    print("Step 1: Calculating velocity hysteresis...")
    print("-" * 70)
    processed_df = calculate_velocity_hysteresis(df, use_log_velocity=False)
    print(f"✓ Calculated hysteresis for {len(processed_df)} participants\n")
    
    # Step 2: Calculate p-values
    print("Step 2: Calculating p-values for group differences...")
    print("-" * 70)
    
    # Regular hysteresis
    pvalues_regular = calculate_hysteresis_pvalues(processed_df, use_absolute=False)
    pvalues_regular['analysis_type'] = 'Regular'
    
    # Absolute hysteresis
    pvalues_absolute = calculate_hysteresis_pvalues(processed_df, use_absolute=True)
    pvalues_absolute['analysis_type'] = 'Absolute'
    
    # Combine
    all_pvalues = pd.concat([pvalues_regular, pvalues_absolute], ignore_index=True)
    
    print(f"✓ Calculated {len(all_pvalues)} statistical tests\n")
    
    # Step 3: Display significant results
    print("Step 3: Significant Results (p < 0.05)")
    print("-" * 70)
    significant = all_pvalues[all_pvalues['p_value'] < 0.05].sort_values('p_value')
    
    if len(significant) > 0:
        for idx, row in significant.iterrows():
            sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*"
            print(f"\n{row['analysis_type']} Hysteresis - {row['grouping_factor']}:")
            print(f"  {row['group1']} vs {row['group2']}")
            print(f"  p-value: {row['p_value']:.4f} {sig_marker}")
            print(f"  Sample sizes: n1={int(row['n1'])}, n2={int(row['n2']) if not pd.isna(row['n2']) else 'N/A'}")
            print(f"  Test: {row['test_type']}")
    else:
        print("\nNo significant differences found (p < 0.05)")
    
    print("\n")
    
    # Step 4: Save results
    print("Step 4: Saving results...")
    print("-" * 70)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'example_hysteresis_pvalues.csv')
    all_pvalues.to_csv(csv_path, index=False)
    print(f"✓ Saved p-values to: {csv_path}")
    
    # Save summary of significant results
    summary_path = os.path.join(output_dir, 'example_significant_results.txt')
    with open(summary_path, 'w') as f:
        f.write("HYSTERESIS ANALYSIS - SIGNIFICANT RESULTS\n")
        f.write("="*70 + "\n\n")
        
        if len(significant) > 0:
            f.write(f"Found {len(significant)} significant differences (p < 0.05):\n\n")
            for idx, row in significant.iterrows():
                sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*"
                f.write(f"{row['analysis_type']} Hysteresis - {row['grouping_factor']}:\n")
                f.write(f"  {row['group1']} vs {row['group2']}\n")
                f.write(f"  p-value: {row['p_value']:.4f} {sig_marker}\n")
                f.write(f"  Sample sizes: n1={int(row['n1'])}, n2={int(row['n2']) if not pd.isna(row['n2']) else 'N/A'}\n")
                f.write(f"  Test: {row['test_type']}\n\n")
        else:
            f.write("No significant differences found at p < 0.05 level.\n")
    
    print(f"✓ Saved summary to: {summary_path}")
    
    # Step 5: Generate plots
    print("\nStep 5: Generating annotated plots...")
    print("-" * 70)
    
    # Regular hysteresis plots
    plot_up_down_diff_boxplots(
        processed_df, 
        use_absolute=False, 
        output_dir=output_dir,
        use_log_velocity=False
    )
    print("✓ Generated regular hysteresis plots")
    
    # Absolute hysteresis plots
    plot_up_down_diff_boxplots(
        processed_df, 
        use_absolute=True, 
        output_dir=output_dir,
        use_log_velocity=False
    )
    print("✓ Generated absolute hysteresis plots")
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  • example_hysteresis_pvalues.csv - All statistical test results")
    print("  • example_significant_results.txt - Summary of significant findings")
    print("  • hysteresis_by_*.png - Boxplots with significance annotations")
    print("  • abs_hysteresis_by_*.png - Absolute value plots")
    print("\nFor more information, see: docs/hysteresis_analysis_guide.md")
    print()


if __name__ == "__main__":
    main()

