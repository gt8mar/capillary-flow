"""
Filename: make_big_csv.py
-------------------------------------------------
This file compiles size, velocity, and metadata data into one 
big csv file. This is useful for plotting and analysis.

By: Marcus Forst
"""

import os, numpy, platform, time
import pandas as pd
from src.tools.find_earliest_date_dir import find_earliest_date_dir

def main(participant, verbose=False, write = True):
    """
    This function takes in the participant path and compiles the 
    size, velocity, and metadata data into one big csv file.

    Args:
        participant (str): The participant number
        verbose (bool): Whether to print to terminal or not
    
    Returns:
        None

    Saves:
        total_csv (csv): A csv file with the compiled data

    """
    SET = 'set01'

    # Create output folders
    if platform.system() == 'Windows':
        capillary_path = 'C:\\Users\\gt8mar\\capillary-flow'
    else:
        capillary_path = '/hpc/projects/capillary-flow'

    results_path = os.path.join(capillary_path, 'results')
    metadata_path = os.path.join(capillary_path, 'metadata')

    size_path = os.path.join(results_path, 'size', 'size_data')
    vel_path = os.path.join(results_path, 'velocities')

    date = find_earliest_date_dir(os.path.join(capillary_path, 'data', participant))
    locations = os.listdir(os.path.join(capillary_path, 'data', participant, date))

    # Create a dataframe for each location
    dataframe_list = []

    for location in locations: 
        # Omit locScan, locTemp, and locEx from this analysis
        if location == 'locScan' or location == 'locTemp' or location == 'locEx':
            continue
        else:
            location_path = os.path.join(capillary_path, 'data', participant, date, location)
            centerline_folder = os.path.join(location_path, 'centerlines')
            size_df = pd.read_csv(os.path.join(size_path, f'{participant}_{date}_{location}_size_data.csv'))
            vel_df = pd.read_csv(os.path.join(vel_path, f'{SET}_{participant}_{date}_{location}_velocity_data.csv'))
            metadata_df = pd.read_excel(os.path.join(metadata_path, f'{participant}_{date}.xlsx'))
            
            # Rename columns of the size dataframe
            size_df.rename(columns={'participant': 'Participant',
                                    'date':'Date',
                                    'location':'Location',
                                    'capnum':'Capillary',
                                    'area':'Area',
                                    'pressure':'Pressure'
                                    }, inplace=True)
            # Rename columns of the velocity dataframe
            vel_df.rename(columns={'Weighted Average Slope': 'Velocity'
                                   }, inplace=True)
            # Replace column vidnum with 'Video' by adding 'vid' to the beginning of the number
            size_df['Video'] = 'vid' + size_df['vidnum'].astype(str).apply(lambda x: x.zfill(2))
            # Drop the vidnum column
            size_df.drop(columns=['vidnum'], inplace=True)
            # Convert capillary number to string in size_df
            size_df['Capillary'] = size_df['Capillary'].astype(str)
            # Drop a from end of capillary number in vel_df
            vel_df['Capillary'] = vel_df['Capillary'].replace({'a':''}, regex=True)
            # Convert capillary number to int64 in vel_df
            vel_df['Capillary'] = vel_df['Capillary'].astype(int).astype(str)

            # Convert location in metadata_df to string of 'loc' + location number with leading zeros
            metadata_df['Location'] = metadata_df['Location'].astype(str).str.zfill(2)
            metadata_df['Location'] = 'loc' + metadata_df['Location']

            print(size_df.dtypes)
            print(vel_df.dtypes)

            # Merge the size and velocity dataframes
            total_df = pd.merge(size_df, vel_df, on=['Participant', 'Date', 'Location', 'Video', 'Pressure','Capillary'], how='outer')

            # Reorder the columns
            total_df = total_df[['Participant', 'Date', 'Location', 'Video', 'Pressure', 'Capillary', 'Area',  'Velocity']]
            # Reorder entries by video number and then by capillary number including leading zeros
            total_df = total_df.sort_values(by=['Video', 'Capillary'], key=lambda x: x.str.zfill(2))
            print(total_df.head)

            # Append the dataframe to the dataframe list
            dataframe_list.append(total_df)

    # Concatenate all the csv files into one big csv file
    total_big_df = pd.concat(dataframe_list)
    
    # check if there are entries in the video column that have bp at the end. add these to a list:
    bp_list = []
    for index, row in metadata_df.iterrows():
        if row['Video'].endswith('bp'):
            bp_list.append(row['Video'])
    print(bp_list)
    bp_list_minus_bp = [item[:-2] for item in bp_list]  
    print(bp_list_minus_bp)

    # Add 'bp' to the end of the video number in the total_big_df if it is in the bp_list
    total_big_df['Video'] = total_big_df['Video'].apply(lambda x: x + 'bp' if x in bp_list_minus_bp else x)

    # Merge the metadata dataframe
    total_big_df = pd.merge(total_big_df, metadata_df, 
                            on=['Participant', 'Date', 'Location', 'Video', 'Pressure'], 
                            how='outer') #.apply(choose_bp, axis=1)
    
    # Print ['Participant', 'Date', 'Location', 'Video', 'Pressure'] column headers and first 5 rows
    print(total_big_df[['Participant', 'Date', 'Location', 'Video', 'Pressure', 'BP']].head)
    print(total_big_df.dtypes)
    
    if write:
        # Save the dataframe as a csv
        total_big_df.to_csv(os.path.join(results_path, 'total', f'{participant}_{date}_total_data.csv'), index=False)
    
    return 0 

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main('part09', verbose=True, write=True)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))


