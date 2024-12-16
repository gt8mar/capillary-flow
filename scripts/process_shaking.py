#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.tools.find_earliest_date_dir import find_earliest_date_dir

def clean_translation_data(df):
    # Convert TranslationX and TranslationY to numeric, replacing errors with NaN
    df['TranslationX'] = pd.to_numeric(df['TranslationX'], errors='coerce')
    df['TranslationY'] = pd.to_numeric(df['TranslationY'], errors='coerce')
    
    # Optionally, drop rows where either column is NaN
    df = df.dropna(subset=['TranslationX', 'TranslationY'])
    
    return df

def analysis():
    final_df = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\combined_results.csv')
    results_dir = 'C:\\Users\\gt8mar\\capillary-flow\\results'
    # Analysis: quantify shaking vs pressure
    # Then perform the aggregation
    group_cols = ['Participant', 'Date', 'Location', 'Video', 'Pressure']
    agg_df = final_df.groupby(group_cols).agg({
        'TranslationX': ['mean', 'std'],
        'TranslationY': ['mean', 'std']
    }).reset_index()

  # Rename the columns correctly
    agg_df.columns = ['_'.join(col).rstrip('_') if isinstance(col, tuple) else col for col in agg_df.columns]

    # round all pressure values to the first decimal
    agg_df['Pressure'] = agg_df['Pressure'].round(1)
    # exclude all pressure values that are not in the range of 0 to 1.2
    agg_df = agg_df[(agg_df['Pressure'] >= 0) & (agg_df['Pressure'] <= 2.0)]

    # Now the columns will be named:
    # 'TranslationX_mean', 'TranslationX_std', 'TranslationY_mean', 'TranslationY_std'

    # Update your correlation calculation to use the correct column names
    cohort_corr_x = agg_df['TranslationX_std'].corr(agg_df['Pressure'])
    cohort_corr_y = agg_df['TranslationY_std'].corr(agg_df['Pressure'])

    # # Plot full cohort
    # plt.figure(figsize=(8,6))
    # plt.scatter(agg_df['Pressure'], agg_df['TranslationX_std'], label='Translation X std', alpha=0.7)
    # plt.scatter(agg_df['Pressure'], agg_df['TranslationY_std'], label='Translation Y std', alpha=0.7)
    # plt.xlabel('Pressure')
    # plt.ylabel('Standard Deviation of Translation')
    # plt.title(f'Full Cohort: Shaking vs Pressure\nCorr X={cohort_corr_x:.2f}, Corr Y={cohort_corr_y:.2f}')
    # plt.legend()
    # cohort_plot_path = os.path.join(results_dir, 'cohort_shaking_vs_pressure.png')
    # # plt.savefig(cohort_plot_path)
    # plt.show()
    # plt.close()

    # plot the box and whisker plot of the variation at each pressure
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Pressure', y='TranslationX_std', data=agg_df)
    plt.xlabel('Pressure')  
    plt.ylabel('Standard Deviation of Translation X')
    plt.title('Variation of Translation X at each Pressure')
    plt.ylim(0, 20)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()
    return 0

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
