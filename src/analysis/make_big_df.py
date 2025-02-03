"""Module for creating and analyzing blood flow velocity data.

This module processes blood flow velocity data from multiple participants,
combining metadata and velocity measurements into a comprehensive DataFrame
for statistical analysis.
"""

import os
import datetime
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import platform

# Define path constants
cap_flow_path = 'C:\\Users\\gt8mar\\capillary-flow'

def extract_capillary(image_path):
    """Extracts capillary name from image path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Name of the capillary extracted from the path.
    """
    image_path = image_path.replace('.tiff', '').replace('.png', '')
    image_path_list = image_path.split('_')
    return image_path_list[-1]

def calculate_age(date, birthday):
    """Calculates age based on date and birthday.

    Args:
        date (str): Date in YYMMDD format.
        birthday (str): Birthday in YYYYMMDD format.

    Returns:
        int: Calculated age in years.
    """
    date = datetime.datetime.strptime(str(int(date)), '%y%m%d')
    birthday = datetime.datetime.strptime(str(int(birthday)), '%Y%m%d')
    age = date.year - birthday.year
    if date.month < birthday.month or (date.month == birthday.month and date.day < birthday.day):
        age -= 1
    return age

def compile_metadata():
    """Compiles metadata from various sources.

    Returns:
        pandas.DataFrame: Combined metadata for all participants.
    """
    # Implementation needed
    pass

def main(verbose=False):
    """Main function to process and combine blood flow velocity data.

    Args:
        verbose (bool): If True, prints additional debug information.

    Returns:
        pandas.DataFrame: Final processed DataFrame containing all measurements and metadata.
    """
    # Set up file paths based on operating system
    if platform.system() == 'Windows':
        path = os.path.join(cap_flow_path, 'results', 'summary_df_test.csv')
        classified_kymos_path = os.path.join(cap_flow_path, 'classified_kymos_real.csv')
    else:
        path = '/hpc/projects/capillary-flow/results/summary_df_test.csv'
        classified_kymos_path = '/hpc/projects/capillary-flow/results/classified_kymos.csv'

    # Load initial datasets
    summary_df = pd.read_csv(path)
    classified_kymos_df = pd.read_csv(classified_kymos_path)
    
    # Load and combine additional classified kymos data
    additional_kymos_files = [
        'classified_kymos_part28_to_part32.csv',
        'classified_kymos_part33_to_part81.csv',
        'classified_kymos_part40_to_part48.csv',
        'classified_kymos_part34_to_part80.csv'
    ]
    
    total_classified_kymos_df = pd.concat([
        pd.read_csv(os.path.join(cap_flow_path, file)) 
        for file in additional_kymos_files
    ], ignore_index=True)

    # Save combined kymos
    total_classified_kymos_df.to_csv(
        os.path.join(cap_flow_path, 'classified_kymos_part28_to_part81.csv'), 
        index=False
    )

    # Process metadata
    metadata_df = compile_metadata()
    total_classified_kymos_df = pd.merge(
        total_classified_kymos_df, 
        metadata_df, 
        on=['Participant', 'Date', 'Location', 'Video'], 
        how='left'
    )

    # Clean and process data
    total_classified_kymos_df = total_classified_kymos_df[
        total_classified_kymos_df['Second_Classification'] != 'Unclear'
    ]
    total_classified_kymos_df['Capillary'] = total_classified_kymos_df['Image_Path'].apply(extract_capillary)
    total_classified_kymos_df['Corrected Velocity'] = total_classified_kymos_df['Classified_Velocity']
    total_classified_kymos_df['Pressure'] = total_classified_kymos_df['Pressure'].round(1)

    # Sort and process participant data
    total_classified_kymos_df = total_classified_kymos_df.sort_values(
        by=['Participant', 'Date', 'Location', 'Video', 'Capillary']
    ).reset_index(drop=True)

    # Process demographic information
    demographic_columns = ['Birthday', 'Diabetes', 'Hypertension', 'HeartDisease', 
                         'Sex', 'Height', 'Weight']
    for col in demographic_columns:
        total_classified_kymos_df[col] = total_classified_kymos_df.groupby('Participant')[col].transform('first')

    # Calculate ages and blood pressure
    total_classified_kymos_df['Age'] = total_classified_kymos_df.apply(
        lambda x: calculate_age(x['Date'], x['Birthday']), axis=1
    )
    total_classified_kymos_df[['SYS_BP', 'DIA_BP']] = total_classified_kymos_df['BP'].str.split('/', expand=True).astype(int)

    # Remove duplicates
    total_classified_kymos_df = total_classified_kymos_df.drop_duplicates(
        subset=['Participant', 'Date', 'Location', 'Video', 'Capillary']
    ).reset_index(drop=True)

    # Final processing and saving
    summary_df_nhp_video_medians = process_final_dataframe(total_classified_kymos_df)
    summary_df_nhp_video_medians.to_csv(
        os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv'), 
        index=False
    )

    return summary_df_nhp_video_medians

def process_final_dataframe(df):
    """Processes the final DataFrame with additional calculations and categorizations.

    Args:
        df (pandas.DataFrame): Input DataFrame with raw data.

    Returns:
        pandas.DataFrame: Processed DataFrame with additional calculations and categories.
    """
    df['Age_Group'] = np.where(df['Age'] > 50, 'Above 50', 'Below 50')
    df['Sex_Group'] = np.where(df['Sex'] == 'M', 'M', 'F')
    df['BP_Group'] = np.where(df['SYS_BP'] > 120, '>120', '<=120')
    df['Video_Median_Velocity'] = df['Video Median Velocity']
    df['Log_Video_Median_Velocity'] = np.log((df['Video Median Velocity'])+1)

    # Convert categories
    category_mappings = {
        'Age_Group': ['Below 50', 'Above 50'],
        'Sex_Group': ['F', 'M'],
        'BP_Group': ['<=120', '>120']
    }
    
    for col, categories in category_mappings.items():
        df[col] = pd.Categorical(df[col], categories=categories, ordered=True)

    return df

if __name__ == '__main__':
    main()
    