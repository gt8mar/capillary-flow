#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.tools.find_earliest_date_dir import find_earliest_date_dir
from matplotlib.font_manager import FontProperties
from scipy.stats import pearsonr, spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess


def clean_translation_data(df):
    # Convert TranslationX and TranslationY to numeric, replacing errors with NaN
    df['TranslationX'] = pd.to_numeric(df['TranslationX'], errors='coerce')
    df['TranslationY'] = pd.to_numeric(df['TranslationY'], errors='coerce')
    
    # Optionally, drop rows where either column is NaN
    df = df.dropna(subset=['TranslationX', 'TranslationY'])
    
    return df

def plot_video_counts_by_pressure(agg_df, pressure_column='Pressure', video_column='Video'):
    """
    Plots the number of unique videos at each pressure level.

    Parameters:
    - agg_df (pd.DataFrame): The aggregated DataFrame.
    - pressure_column (str): The name of the column representing pressure levels.
    - video_column (str): The name of the column representing video identifiers.
    """
    # sns.set_style("whitegrid")
    source_sans = FontProperties(fname='C:\\Users\\gt8mar\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

    # Count the number of unique videos at each pressure level
    pressure_counts = agg_df.groupby(pressure_column)[video_column].nunique()

    # Plot the results
    plt.figure(figsize=(4.0, 3.0))
    pressure_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Number of Videos at Each Pressure', fontproperties=source_sans)
    plt.xlabel('Pressure (psi)', fontproperties=source_sans)
    plt.ylabel('Number of Videos', fontproperties=source_sans)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    plt.savefig('C:\\Users\\gt8mar\\capillary-flow\\results\\video_counts_by_pressure.png', dpi=400)
    plt.close()
    return 0

def plot_video_transl_by_pressure(agg_df, pressure_column='Pressure', video_column='Video', show_outliers=False):
    """
    Plots the standard deviation of translation at each pressure level.

    Args:
    - agg_df (pd.DataFrame): The aggregated DataFrame.
    - pressure_column (str): The name of the column representing pressure levels.
    - video_column (str): The name of the column representing video identifiers.

    Returns:
    - None
    """

    source_sans = FontProperties(fname='C:\\Users\\gt8mar\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

     # plot the box and whisker plot of the variation at each pressure
    plt.figure(figsize=(4.0, 3.0))
    sns.boxplot(x='Pressure', y='TranslationTotal_std', data=agg_df, showfliers=show_outliers)
    plt.xlabel('Pressure (psi)', fontproperties=source_sans)  
    plt.ylabel('Standard Deviation of Translation (pixels)', fontproperties=source_sans)
    plt.title('Variation of Translation at each Pressure', fontproperties=source_sans)
    plt.ylim(0, 40)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    plt.savefig('C:\\Users\\gt8mar\\capillary-flow\\results\\video_transl_by_pressure.png', dpi=400)
    plt.close()
    return 0

def calculate_correlations(agg_df):
    # Example: X-axis correlation with Pressure
    pearson_r_x, pearson_p_x = pearsonr(agg_df['TranslationX_std'], agg_df['Pressure'])
    spearman_r_x, spearman_p_x = spearmanr(agg_df['TranslationX_std'], agg_df['Pressure'])

    print(f"Pearson correlation (X_std vs Pressure) = {pearson_r_x:.3f}, p-value = {pearson_p_x:.3e}")
    print(f"Spearman correlation (X_std vs Pressure) = {spearman_r_x:.3f}, p-value = {spearman_p_x:.3e}")

    # Repeat for Y_std
    pearson_r_y, pearson_p_y = pearsonr(agg_df['TranslationY_std'], agg_df['Pressure'])
    spearman_r_y, spearman_p_y = spearmanr(agg_df['TranslationY_std'], agg_df['Pressure'])

    print(f"Pearson correlation (Y_std vs Pressure) = {pearson_r_y:.3f}, p-value = {pearson_p_y:.3e}")
    print(f"Spearman correlation (Y_std vs Pressure) = {spearman_r_y:.3f}, p-value = {spearman_p_y:.3e}")

    # And for Total_std
    pearson_r_t, pearson_p_t = pearsonr(agg_df['TranslationTotal_std'], agg_df['Pressure'])
    spearman_r_t, spearman_p_t = spearmanr(agg_df['TranslationTotal_std'], agg_df['Pressure'])

    print(f"Pearson correlation (Total_std vs Pressure) = {pearson_r_t:.3f}, p-value = {pearson_p_t:.3e}")
    print(f"Spearman correlation (Total_std vs Pressure) = {spearman_r_t:.3f}, p-value = {spearman_p_t:.3e}")
    return 0

# Function to plot Spearman trends with loess smoothing
def plot_spearman_with_loess(df, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(2.4, 2))

    source_sans = FontProperties(fname='C:\\Users\\gt8mar\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })

    # Calculate Spearman correlation
    spearman_corr, p_value = spearmanr(df[x_col], df[y_col])
    
    # Loess smoothing for trend line
    smoothed = lowess(df[y_col], df[x_col], frac=0.3)  # Adjust frac for smoothing level
    
    # Scatter plot
    sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.6, ax=ax)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', label='Loess Trend')
    
    # Annotations
    ax.set_title(f"{title}\nSpearman ρ = {spearman_corr:.3f}, p = {p_value:.3e}", fontproperties=source_sans)
    ax.set_xlabel("Pressure", fontproperties=source_sans)
    ax.set_ylabel(y_col, fontproperties=source_sans)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'C:\\Users\\gt8mar\\capillary-flow\\results\\{title}.png', dpi=400)
    return 0


def analyze_decay_rate(agg_df, results_dir):
    """
    Analyze the decay rate of translation standard deviation with pressure.
    
    Parameters:
    -----------
    agg_df : pandas.DataFrame
        DataFrame containing at least 'Pressure' and 'TranslationTotal_std' columns
    results_dir : str
        Directory path where to save the plot
        
    Returns:
    --------
    dict
        Dictionary containing the decay parameters: initial_amplitude, decay_rate, r_squared
    """
    from sklearn.linear_model import LinearRegression

    source_sans = FontProperties(fname='C:\\Users\\gt8mar\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')
    
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 5, 'lines.linewidth': 0.5
    })
    
    
    # Calculate log of translation std
    agg_df['log_TranslationTotal_std'] = np.log(agg_df['TranslationTotal_std']+1)
    
    # Fit exponential decay: y = A*exp(-Bx)
    # Taking log of both sides: ln(y) = ln(A) - Bx
    X = agg_df['Pressure'].values.reshape(-1, 1)
    y = agg_df['log_TranslationTotal_std'].values
    
    # Use linear regression to fit the log-transformed data
    reg = LinearRegression().fit(X, y)
    
    # Calculate decay rate (B) and initial amplitude (A)
    decay_rate = -reg.coef_[0]  # B
    initial_amplitude = np.exp(reg.intercept_)  # A
    r_squared = reg.score(X, y)
    
    # Plot the data and fit
    plt.figure(figsize=(4, 3))
    plt.scatter(agg_df['Pressure'], agg_df['TranslationTotal_std'], 
                alpha=0.5, label='Data')
    
    # Generate points for the fitted curve
    pressure_range = np.linspace(agg_df['Pressure'].min(), agg_df['Pressure'].max(), 100)
    fit_curve = initial_amplitude * np.exp(-decay_rate * pressure_range)
    
    plt.plot(pressure_range, fit_curve, 'r-', 
             label=f'Fit: {initial_amplitude:.2f}*exp(-{decay_rate:.2f}x)')
    
    plt.xlabel('Pressure', fontproperties=source_sans)
    plt.ylabel('Translation Total std', fontproperties=source_sans)
    plt.title('Exponential Decay Fit of Translation vs Pressure', fontproperties=source_sans)
    plt.legend()
    plt.yscale('log')
    
    # Save the plot
    decay_plot_path = os.path.join(results_dir, 'translation_decay_fit.png')
    plt.savefig(decay_plot_path, dpi=400)
    plt.close()
    
    # Create LaTeX table
    latex_table = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Parameter}} & \\textbf{{Value}} \\\\
\\hline
Initial Amplitude (A) & {initial_amplitude:.3f} \\\\
Decay Rate (B) & {decay_rate:.3f} \\\\
R² & {r_squared:.3f} \\\\
\\hline
\\end{{tabular}}
\\caption{{Exponential decay fit parameters for translation standard deviation vs pressure. The fit follows the form: $y = A e^{{-Bx}}$}}
\\label{{tab:decay_params}}
\\end{{table}}
"""
    
    print("\nLaTeX Table:")
    print(latex_table)
    
    # Print numerical results
    print(f"\nNumerical Results:")
    print(f"Initial Amplitude (A): {initial_amplitude:.3f}")
    print(f"Decay Rate (B): {decay_rate:.3f}")
    print(f"R²: {r_squared:.3f}")
    
    return {
        'initial_amplitude': initial_amplitude,
        'decay_rate': decay_rate,
        'r_squared': r_squared
    }

def analysis():
    final_df = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\combined_results.csv')
    results_dir = 'C:\\Users\\gt8mar\\capillary-flow\\results'
    # Analysis: quantify shaking vs pressure
    # Then perform the aggregation

    print(f'the number of frames is {len(final_df)}')
    # the number of vidoes is the number of unique video/participant pairs
    print(f'the number of videos is {len(final_df.groupby(["Participant", "Video"]))}')

    group_cols = ['Participant', 'Date', 'Location', 'Video', 'Pressure']
    agg_df = final_df.groupby(group_cols).agg({
        'TranslationX': ['mean', 'std'],
        'TranslationY': ['mean', 'std']
    }).reset_index()

  # Rename the columns correctly
    agg_df.columns = ['_'.join(col).rstrip('_') if isinstance(col, tuple) else col for col in agg_df.columns]

    # round all pressure values to the first decimal
    agg_df['Pressure'] = agg_df['Pressure'].round(1)

    plot_video_counts_by_pressure(agg_df, pressure_column='Pressure', video_column='Video')

    # exclude all pressure values that are not in the range of 0 to 1.2
    # agg_df = agg_df[(agg_df['Pressure'] >= 0) & (agg_df['Pressure'] <= 2.0)]

    # Now the columns will be named: 'TranslationX_mean', 'TranslationX_std', 'TranslationY_mean', 'TranslationY_std'

    agg_df['TranslationTotal_std'] = np.sqrt(agg_df['TranslationX_std']**2 + agg_df['TranslationY_std']**2)

    # Update your correlation calculation to use the correct column names
    cohort_corr_x = agg_df['TranslationX_std'].corr(agg_df['Pressure'])
    cohort_corr_y = agg_df['TranslationY_std'].corr(agg_df['Pressure'])
    cohort_corr_total = agg_df['TranslationTotal_std'].corr(agg_df['Pressure'])
    
    calculate_correlations(agg_df)

    plot_video_transl_by_pressure(agg_df, pressure_column='Pressure', video_column='Video')

    # Calculate the decay rate between the pressure and the log of the standard deviation of the translation
    # and print a table of the results in latex
    decay_params = analyze_decay_rate(agg_df, results_dir)

    # agg_df['log_TranslationTotal_std'] = np.log(agg_df['TranslationTotal_std']+1)
    # log_corr_total = agg_df['log_TranslationTotal_std'].corr(agg_df['Pressure'])
    # print(f"Correlation between log of Translation Total std and Pressure: {log_corr_total:.2f}")

    # Plot Spearman trends with loess smoothing
    # plot_spearman_with_loess(agg_df, 'Pressure', 'TranslationX_std', 'Spearman Trend X_std vs Pressure')
    # plot_spearman_with_loess(agg_df, 'Pressure', 'TranslationY_std', 'Spearman Trend Y_std vs Pressure')
    plot_spearman_with_loess(agg_df, 'Pressure', 'TranslationTotal_std', 'Spearman Trend Total_std vs Pressure')

  
def main():
    # Set up logging
    log_path = '/hpc/projects/capillary-flow/scripts/process_data.log'
    logging.basicConfig(filename=log_path, level=logging.INFO, 
                        format='%(asctime)s %(levelname)s:%(message)s')

    # Base directories
    data_dir = '/hpc/projects/capillary-flow/data'
    metadata_dir = '/hpc/projects/capillary-flow/metadata'
    output_dir = '/hpc/projects/capillary-flow'
    results_dir = '/hpc/projects/capillary-flow/results'
    os.makedirs(results_dir, exist_ok=True)

    # Participants to ignore
    ignore_participants = ['part24', 'part10']

    # Folders to ignore
    ignore_locations = ['locTemp', 'locEx', 'locScan']

    # Moco priority
    moco_priority = ['moco-contrasted', 'mocoslice', 'moco-split', 'moco']

    # Create a list of participants
    participants = [p for p in os.listdir(data_dir) if p.startswith('part') and os.path.isdir(os.path.join(data_dir, p))]
    participants = [p for p in participants if p not in ignore_participants]

    all_data = []

    for participant in participants:
        participant_path = os.path.join(data_dir, participant)
        if not os.path.isdir(participant_path):
            logging.warning(f"Participant path not found: {participant_path}")
            continue
        
        date_str = find_earliest_date_dir(participant_path)

    
        # Metadata file for this participant and date
        metadata_file = os.path.join(metadata_dir, f"{participant}_{date_str}.xlsx")
        if not os.path.isfile(metadata_file):
            logging.warning(f"Metadata file missing for {participant}, {date_str}: {metadata_file}")
            continue
        meta_df = pd.read_excel(metadata_file)
        
        date_path = os.path.join(participant_path, date_str)
        if not os.path.isdir(date_path):
            logging.warning(f"No date directory found for {participant}, {date_str}")
            continue
        
        # Locations
        locations = [loc for loc in os.listdir(date_path) 
                    if os.path.isdir(os.path.join(date_path, loc)) and loc not in ignore_locations]
        
        if len(locations) == 0:
            logging.info(f"No valid location directories for {participant}, {date_str}")
        
        for loc in locations:
            loc_path = os.path.join(date_path, loc)
            
            # Videos
            vids_path = os.path.join(loc_path, 'vids')
            if not os.path.isdir(vids_path):
                logging.info(f"No vids directory for {participant}, {date_str}, {loc}")
                continue
            
            videos = [v for v in os.listdir(vids_path) 
                    if v.startswith('vid') and os.path.isdir(os.path.join(vids_path, v))]
            
            for video in videos:
                # Pressure value
                # the location value is the integer from the location name form loc01: namely 1
                location_value = int(loc.replace('loc', ''))
                pressure_value = meta_df[(meta_df['Location'] == location_value) & (meta_df['Video'] == video)]['Pressure'].values
                if len(pressure_value) == 0:
                    logging.info(f"No pressure value found for {participant}, {date_str}, {loc}, {video}")
                    continue
                pressure_value = pressure_value[0] 
        
                video_path = os.path.join(vids_path, video)
                
                # Check moco directories in priority
                chosen_moco = None
                for moco_dir_name in moco_priority:
                    moco_path = os.path.join(video_path, moco_dir_name)
                    if os.path.isdir(moco_path):
                        chosen_moco = moco_dir_name
                        break
                
                if chosen_moco is None:
                    # No moco folder found
                    logging.info(f"No moco folder found for {participant}, {date_str}, {loc}, {video}")
                    continue
                
                metadata_path = os.path.join(video_path, 'metadata')
                if not os.path.isdir(metadata_path):
                    logging.info(f"No metadata directory under moco for {participant}, {date_str}, {loc}, {video}")
                    continue
                
                results_file = os.path.join(video_path, 'metadata', 'Results.csv')
                if not os.path.isfile(results_file):
                    logging.info(f"No Results.csv for {participant}, {date_str}, {loc}, {video}")
                    continue
                
                # Load Results.csv
                # Assume Results.csv has columns for translation shifts like "Frame", "TranslationX", "TranslationY", etc.
                try:
                    # Load Results.csv as a headerless file
                    results_df = pd.read_csv(results_file, header=None, names=['TranslationX', 'TranslationY'])
                except Exception as e:
                    logging.error(f"Error reading Results.csv {results_file}: {e}")
                    continue

                
                # Add identifying columns
                results_df['Participant'] = participant
                results_df['Date'] = date_str
                results_df['Location'] = loc
                results_df['Video'] = video
                results_df['Moco'] = chosen_moco
                # Add pressure
                results_df['Pressure'] = pressure_value
                
                all_data.append(results_df)

    # Combine all data
    if len(all_data) == 0:
        logging.error("No data collected.")
        exit(1)

    final_df = pd.concat(all_data, ignore_index=True)

    final_df = clean_translation_data(final_df)

    # Save the final dataframe
    final_csv_path = os.path.join(output_dir, 'combined_results.csv')
    final_df.to_csv(final_csv_path, index=False)

    # Analysis: quantify shaking vs pressure
    # Then perform the aggregation
    group_cols = ['Participant', 'Date', 'Location', 'Video', 'Pressure']
    agg_df = final_df.groupby(group_cols).agg({
        'TranslationX': ['mean', 'std'],
        'TranslationY': ['mean', 'std']
    }).reset_index()

  # Rename the columns correctly
    agg_df.columns = ['_'.join(col).rstrip('_') if isinstance(col, tuple) else col for col in agg_df.columns]

    # Now the columns will be named:
    # 'TranslationX_mean', 'TranslationX_std', 'TranslationY_mean', 'TranslationY_std'

    # Update your correlation calculation to use the correct column names
    cohort_corr_x = agg_df['TranslationX_std'].corr(agg_df['Pressure'])
    cohort_corr_y = agg_df['TranslationY_std'].corr(agg_df['Pressure'])

    # Plot full cohort
    plt.figure(figsize=(8,6))
    plt.scatter(agg_df['Pressure'], agg_df['TranslationX_std'], label='Translation X std', alpha=0.7)
    plt.scatter(agg_df['Pressure'], agg_df['TranslationY_std'], label='Translation Y std', alpha=0.7)
    plt.xlabel('Pressure')
    plt.ylabel('Standard Deviation of Translation')
    plt.title(f'Full Cohort: Shaking vs Pressure\nCorr X={cohort_corr_x:.2f}, Corr Y={cohort_corr_y:.2f}')
    plt.legend()
    cohort_plot_path = os.path.join(results_dir, 'cohort_shaking_vs_pressure.png')
    plt.savefig(cohort_plot_path)
    plt.close()

    # Individual participants
    for p in agg_df['Participant'].unique():
        sub_df = agg_df[agg_df['Participant'] == p]
        if len(sub_df) < 2:
            continue
        p_corr_x = sub_df['TranslationX_std'].corr(sub_df['Pressure'])
        p_corr_y = sub_df['TranslationY_std'].corr(sub_df['Pressure'])
        
        plt.figure(figsize=(8,6))
        plt.scatter(sub_df['Pressure'], sub_df['TranslationX_std'], label='Translation X std', alpha=0.7)
        plt.scatter(sub_df['Pressure'], sub_df['TranslationY_std'], label='Translation Y std', alpha=0.7)
        plt.xlabel('Pressure')
        plt.ylabel('Standard Deviation of Translation')
        plt.title(f'{p}: Shaking vs Pressure\nCorr X={p_corr_x:.2f}, Corr Y={p_corr_y:.2f}')
        plt.legend()
        p_plot_path = os.path.join(results_dir, f'{p}_shaking_vs_pressure.png')
        plt.savefig(p_plot_path)
        plt.close()

    logging.info("Processing completed successfully.")

if __name__ == '__main__':
    # main()
    analysis()
