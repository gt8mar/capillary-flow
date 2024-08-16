"""
Filename: make_summary_df.py
----------------------------
This script merges the capillary diameter measurements with the corrected velocity measurements.

By: Marcus Forst
"""

import os, platform
import pandas as pd
from src.tools.parse_filename import parse_filename

def compile_metadata():
    metadata_folder = 'C:\\Users\\gt8mar\\capillary-flow\\metadata'
    # Read the metadata files if they are csvs
    metadata_files = [f for f in os.listdir(metadata_folder) if f.endswith('.xlsx')]
    metadata_dfs = [pd.read_excel(os.path.join(metadata_folder, f)) for f in metadata_files]
    metadata_df = pd.concat(metadata_dfs)

    # make slice of metadata_df with only bp measurements
    non_bp_metadata = metadata_df[~metadata_df['Video'].str.contains('bp')]
   
    # add 'loc' and a leading zero to the location column
    non_bp_metadata['Location'] = 'loc' + non_bp_metadata['Location'].astype(str).str.zfill(2)

    # Convert 'Video' identifiers to integers for comparison
    non_bp_metadata['VideoID'] = non_bp_metadata['Video'].str.extract('(\d+)').astype(int)

    # remove all part09 videos greater than vid59:
    non_bp_metadata = non_bp_metadata[~((non_bp_metadata['Participant'] == 'part09') & (non_bp_metadata['VideoID'] > 59))]

    # keep only participant, date, location, and video columns
    non_bp_metadata = non_bp_metadata[['Participant', 'Date', 'Location', 'Video']]
    return non_bp_metadata

def compare_participants(df1, df2):
    """ 
    Compare the participants in two DataFrames and return a list of participants that are in one DataFrame 
    but not the other, or have different counts in the two DataFrames.

    Args:
        df1 (DataFrame): the first DataFrame
        df2 (DataFrame): the second DataFrame
    
    Returns:
        
    """
    # Group by 'Participant' and 'Capillary' and count the number of rows for each group
    df1_grouped = df1.groupby(['Participant', 'Video', 'Capillary']).size()
    df2_grouped = df2.groupby(['Participant', 'Video', 'Capillary']).size()

    # Group by 'Participant' and count the total number of rows for each participant
    df1_participant_counts = df1['Participant'].value_counts()
    df2_participant_counts = df2['Participant'].value_counts()

    different_rows = []
    different_participants = []


    for row in df1_grouped.index:
        if row not in df2_grouped or df1_grouped[row] != df2_grouped[row]:
            different_rows.append(row)

    for row in df2_grouped.index:
        if row not in df1_grouped:
            different_rows.append(row)
    
    # Check for different total counts for participants
    for participant in df1_participant_counts.index:
        if participant not in df2_participant_counts or df1_participant_counts[participant] != df2_participant_counts[participant]:
            different_participants.append(participant)

    for participant in df2_participant_counts.index:
        if participant not in df1_participant_counts:
            different_participants.append(participant)

    return different_rows, different_participants
def handle_dotted_evac(summary_df):
    # Fill NaN values in the 'Notes' column with an empty string
    summary_df['Notes'] = summary_df['Notes'].fillna('')
    # Now it is time to handle the added rows in velocity_df

    # if the row has 'dotted' in the notes column, that means that the velocity and area are zero.
    # if the row has 'evac' in the notes column, that means that the velocity is zero and the area is zero.

    # Condition to find rows with 'NaN' in 'Area' and 'dotted' in 'Notes'
    condition = summary_df['Area'].isna() & summary_df['Notes'].str.contains('dotted')
    condition_evac = summary_df['Area'].isna() & summary_df['Notes'].str.contains('evac')
    summary_df.loc[condition, 'Area'] = 0
    summary_df.loc[condition, 'Diameter'] = 0
    summary_df.loc[condition, 'Corrected Velocity'] = 0
    summary_df.loc[condition, 'Centerline'] = 0
    summary_df.loc[condition_evac, 'Area'] = 0
    summary_df.loc[condition_evac, 'Diameter'] = 0
    summary_df.loc[condition_evac, 'Corrected Velocity'] = 0
    summary_df.loc[condition_evac, 'Centerline'] = 0
    return summary_df

def check_inserted_rows(summary_df):
     # if the row has 'inserted' in the notes column, that means that the area is the same as the original area from the 'Capillary' column
    condition_inserted = summary_df['Notes'].str.contains('inserted')
    # print the rows that have 'inserted' in the notes column
    print(summary_df[condition_inserted][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Capillary_new', 'Area']])

    # print all part26 loc01 rows
    print(summary_df[(summary_df['Participant'] == 'part26') & (summary_df['Location'] == 'loc01') & (summary_df['Video'] == 'vid05')][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Capillary_new', 'Area', 'Corrected Velocity', 'Diameter']])
    return 0

def merge_vel_size(verbose=False):
    size_df = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\results\\cap_diameters.csv')
    # velocity_df = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\big_df - Copy.csv')
    velocity_df = pd.read_excel('C:\\Users\\gt8mar\\capillary-flow\\results\\big_df.xlsx')
    # velocity_df_old = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\big_df.csv')
    metadata_df = compile_metadata()
    print(metadata_df.head)

    # modify size_df to remove all bp measurements
    print(f'size_df shape: {size_df.shape}')
    
    # remove 'bp' from the video column
    size_df['Video'] = size_df['Video'].str.replace('bp', '')

    # use outer merge to find the rows in size_df that are not bp measurements
    size_df = size_df.merge(metadata_df, on=['Participant', 'Date', 'Location', 'Video'], how='inner', indicator=False)
    print(f'new size_df shape: {size_df.shape}')
    print(size_df.head())

    different_rows, different_participants = compare_participants(size_df, velocity_df)

    # save for testing
    # size_df.to_csv('C:\\Users\\gt8mar\\capillary-flow\\size_test.csv', index=False)
    # velocity_df.to_csv('C:\\Users\\gt8mar\\capillary-flow\\velocity_test.csv', index=False)

    # # remove part22 and part23 from different rows
    # different_rows = [row for row in different_rows if row[0] != 'part22' and row[0] != 'part23']
    print(different_rows)
    print(different_participants)

    # pd.set_option('display.max_rows', None)

    # print(velocity_df[velocity_df['Participant'] == 'part15'][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Corrected Velocity']])
    # velocity_part15_shape = velocity_df[velocity_df['Participant'] == 'part15'].shape
    # print(f'Velocity df shape: {velocity_part15_shape}')
    # print(size_df[size_df['Participant'] == 'part15'][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Diameter']])
    # size_part15_shape = size_df[size_df['Participant'] == 'part15'].shape
    # print(f'Size df shape: {size_part15_shape}')

    # remove SYS_BP column from size_df
    size_df = size_df.drop(columns=['SYS_BP'])

    # Merge the DataFrames
    summary_df = pd.merge(size_df, velocity_df, how='outer',on=['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Age'], indicator=True)
    
    summary_df = handle_dotted_evac(summary_df)

    if verbose: 
        check_inserted_rows(summary_df)
    
    # print any rows where area is NaN
    print("the following rows have NaN in the 'Area' column: ")
    print(summary_df[summary_df['Area'].isna()][['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Area', 'Corrected Velocity', 'Diameter']])
    
    # make the 'Drop' column strings
    summary_df['Drop'] = summary_df['Drop'].astype(str)

    # remove row if 'drop' is in the Drop column
    summary_df = summary_df[~summary_df['Drop'].str.contains('drop')]

    # if there is a number in "Manual" column, replace "Corrected Velocity" with "Manual"
    summary_df['Corrected Velocity'] = summary_df['Manual'].fillna(summary_df['Corrected Velocity'])
    # if there is a number in "Manual Velocity" column, replace "Corrected Velocity" with "Manual Velocity"
    summary_df['Corrected Velocity'] = summary_df['Manual Velocity'].fillna(summary_df['Corrected Velocity'])
   
    # save summary_df to csv
    summary_df.to_csv('C:\\Users\\gt8mar\\capillary-flow\\summary_df_test.csv', index=False)
    return summary_df

def main(participant_lead, participant_end_inclusive):
    if platform.system() == 'Windows':
        results_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results'
    else:
        results_folder = '/hpc/projects/capillary-flow/results'
    velocity_folder = os.path.join(results_folder, 'velocities')
    
    participant_list  = [f'part{i:02d}' for i in range(participant_lead, participant_end_inclusive+1)]
    print(participant_list)
    file_list = os.listdir(velocity_folder)
    velocity_csvs = []
    for participant in participant_list:
        # search for csvs with the participant name in the filename
        for file in file_list:
            if participant in file and file.endswith('.csv'):
                participant, date, location, __, __ = parse_filename(file)
                velocity_csvs.append([participant, date, location, file])

        # csv_path = os.path.join(velocity_folder, f'{participant}_velocities.csv')
        # update_velocities2(csv_path)
    # make velocity_csvs into a DataFrame
    velocity_csvs_df = pd.DataFrame(velocity_csvs, columns=['Participant', 'Date', 'Location', 'Filename'])
    # Sort first by participant, then by date, then by location
    velocity_csvs_df = velocity_csvs_df.sort_values(by=['Participant', 'Date', 'Location'])
    
    # Row by row, load the csvs into a big DataFrame
    big_df = pd.DataFrame()
    for index, row in velocity_csvs_df.iterrows():
        csv_path = os.path.join(velocity_folder, row['Filename'])
        df = pd.read_csv(csv_path)
        big_df = pd.concat([big_df, df])

    # save the big_df to a csv
    filename = f'velocity_df_part{participant_lead:02d}_to_part{participant_end_inclusive:02d}.csv'
    big_df.to_csv(f'C:\\Users\\gt8mar\\capillary-flow\\results\\{filename}', index=False)

if __name__ == '__main__':
    # merge_vel_size(verbose=True)
    main(33, 81)
    
