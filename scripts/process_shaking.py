#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

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

    # Get dates
    dates = [d for d in os.listdir(participant_path) if os.path.isdir(os.path.join(participant_path, d))]
    
    # Attempt to load participant-level metadata as needed
    # We'll load metadata per date since we have {participant}_{date}.xlsx
    for date_str in dates:
        # Metadata file for this participant and date
        metadata_file = os.path.join(metadata_dir, f"{participant}_{date_str}.xlsx")
        if not os.path.isfile(metadata_file):
            logging.warning(f"Metadata file missing for {participant}, {date_str}: {metadata_file}")
            continue
        
        # Load pressure data
        # Assume the metadata file has a column named "Pressure"
        try:
            meta_df = pd.read_excel(metadata_file)
            if 'Pressure' not in meta_df.columns:
                logging.warning(f"No 'Pressure' column in {metadata_file}")
                continue
            # We'll assume a single pressure value per participant_date combination,
            # or you might have a row for each condition. Adjust as needed.
            # If multiple rows, you may need to select or aggregate.
            pressure_value = meta_df['Pressure'].mean()  # Example: take mean if multiple entries
        except Exception as e:
            logging.error(f"Error reading metadata file {metadata_file}: {e}")
            continue
        
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
                
                metadata_path = os.path.join(video_path, chosen_moco, 'metadata')
                if not os.path.isdir(metadata_path):
                    logging.info(f"No metadata directory under moco for {participant}, {date_str}, {loc}, {video}")
                    continue
                
                results_file = os.path.join(metadata_path, 'Results.csv')
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

# Save the final dataframe
final_csv_path = os.path.join(output_dir, 'combined_results.csv')
final_df.to_csv(final_csv_path, index=False)

# Analysis: quantify shaking vs pressure
# For simplicity, let's assume "TranslationX", "TranslationY" columns exist.
# We'll compute a measure of "shaking" as the standard deviation of translation per video.
# Then correlate with pressure.

# Aggregate by Participant, Date, Location, Video
group_cols = ['Participant', 'Date', 'Location', 'Video', 'Pressure']
agg_df = final_df.groupby(group_cols).agg(
    mean_trans_x=('TranslationX', 'mean'),
    std_trans_x=('TranslationX', 'std'),
    mean_trans_y=('TranslationY', 'mean'),
    std_trans_y=('TranslationY', 'std')
).reset_index()

# Full cohort analysis: correlation between std of translation and Pressure
cohort_corr_x = agg_df['std_trans_x'].corr(agg_df['Pressure'])
cohort_corr_y = agg_df['std_trans_y'].corr(agg_df['Pressure'])

# Plot full cohort
plt.figure(figsize=(8,6))
plt.scatter(agg_df['Pressure'], agg_df['std_trans_x'], label='Std Translation X', alpha=0.7)
plt.scatter(agg_df['Pressure'], agg_df['std_trans_y'], label='Std Translation Y', alpha=0.7)
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
    p_corr_x = sub_df['std_trans_x'].corr(sub_df['Pressure'])
    p_corr_y = sub_df['std_trans_y'].corr(sub_df['Pressure'])
    
    plt.figure(figsize=(8,6))
    plt.scatter(sub_df['Pressure'], sub_df['std_trans_x'], label='Std Translation X', alpha=0.7)
    plt.scatter(sub_df['Pressure'], sub_df['std_trans_y'], label='Std Translation Y', alpha=0.7)
    plt.xlabel('Pressure')
    plt.ylabel('Standard Deviation of Translation')
    plt.title(f'{p}: Shaking vs Pressure\nCorr X={p_corr_x:.2f}, Corr Y={p_corr_y:.2f}')
    plt.legend()
    p_plot_path = os.path.join(results_dir, f'{p}_shaking_vs_pressure.png')
    plt.savefig(p_plot_path)
    plt.close()

logging.info("Processing completed successfully.")
