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
from src.tools.load_name_map import load_name_map
from src.tools.parse_filename import parse_filename

def make_big_name_map(participant):
    if platform.system() == 'Windows':
        name_map_path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\size\\name_maps'
        data_path = 'C:\\Users\\gt8mar\\capillary-flow\\data'
    else:
        name_map_path = '/hpc/projects/capillary-flow/results/size/name_maps'
        data_path = '/hpc/projects/capillary-flow/data'
    # load name maps for each location for the participant, concatenate them
    name_map_list = []
    for filename in os.listdir(name_map_path):
        if filename.startswith(participant):
            # get date and location from filename
            date = filename.split('_')[1]
            location = filename.split('_')[2]
            name_map_list.append(load_name_map(participant, date, location, version = 'centerlines'))
    name_map_df = pd.concat(name_map_list)
    # print full text of first 5 entries of 'centerlines name' in name_map_df
    with pd.option_context('display.max_colwidth', None):
        print(name_map_df['centerlines name'].head)
        # print number of rows in name_map_df
        print(len(name_map_df))
    return name_map_df

def make_centerline_df(participant, date, location, verbose=False):
    """
    This function takes in the participant, date, and location
    and returns a dataframe with the length of capillaries.

    Args:
        participant (str): The participant number
        date (str): The date of the experiment
        location (str): The location on the finger
        verbose (bool): Whether to print to terminal or not

    Returns:
        centerline_df (pd.DataFrame): A dataframe with length of capillary centerlines

    """
    # Create output folders
    if platform.system() == 'Windows':
        capillary_path = 'C:\\Users\\gt8mar\\capillary-flow'
    else:
        capillary_path = '/hpc/projects/capillary-flow'

    results_path = os.path.join(capillary_path, 'results')
    centerline_path = os.path.join(results_path, 'centerlines')

    name_map = load_name_map(participant, date, location, version = 'centerlines')

    # Create a dataframe with columns 'Participant', 'Date', 'Location', 'Video', 'Capillary', 'Centerline Length'
    centerline_df = pd.DataFrame(columns=['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Centerline Length'])
    
    for filename in os.listdir(centerline_path):
        if filename.startswith(f'set_01_{participant}') and filename.endswith('.csv'):
            # switch to set01
            filename = filename.replace('set_01', 'set01')
        elif filename.startswith(f'set01_{participant}') and filename.endswith('.csv'):
            # Get the video number from the filename
            pass
        else:
            continue
        _, _, _, video, _ = parse_filename(filename)
        centerline_coords = pd.read_csv(os.path.join(centerline_path, filename))
        # The number of rows of the centerline_coords dataframe is the length of the capillary
        cap_length = len(centerline_coords)
        # Check if filename is in name_map
        if name_map['centerlines name'].str.contains(filename).any(): 
            if verbose:           
                print(f'{filename} in name_map')
            pass
        elif location in filename:
            print(f'{filename} in location {location} not in name_map')
            continue
        else:
            continue

        # Get the capillary number from the name map and filename
        cap_num = name_map[name_map['centerlines name'] == filename]['cap name short'].values[0]
        # Add a row to the centerline_df dataframe
        new_row_df = pd.DataFrame([{'Participant': participant, 
                                    'Date': date, 
                                    'Location': location, 
                                    'Video': video, 
                                    'Capillary': cap_num, 
                                    'Centerline Length': cap_length}])
        centerline_df = pd.concat([centerline_df, new_row_df], ignore_index=True)
    return centerline_df
            
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
    centerline_path = os.path.join(results_path, 'centerlines')
    size_path = os.path.join(results_path, 'size', 'size_data')
    vel_path = os.path.join(results_path, 'velocities')

    # load name maps for each location for the participant, concatenate them
    name_map_df = make_big_name_map(participant)

    # Get the earliest date for the participant
    date = find_earliest_date_dir(os.path.join(capillary_path, 'data', participant))

    # Get the locations for the participant
    locations = os.listdir(os.path.join(capillary_path, 'data', participant, date))

    # Create a dataframe for each location
    dataframe_list = []

    for location in locations: 
        # Omit locScan, locTemp, and locEx from this analysis
        if location == 'locScan' or location == 'locTemp' or location == 'locEx':
            continue
        else:
            print(f'constructing dataframe for {location}')
            location_path = os.path.join(capillary_path, 'data', participant, date, location)
            size_df = pd.read_csv(os.path.join(size_path, f'{participant}_{date}_{location}_size_data.csv'))
            vel_df = pd.read_csv(os.path.join(vel_path, f'{SET}_{participant}_{date}_{location}_velocity_data.csv'))
            metadata_df = pd.read_excel(os.path.join(metadata_path, f'{participant}_{date}.xlsx'))
            
            # Create a dataframe with length of capillaries
            centerline_df = make_centerline_df(participant, date, location)

            # Change date in centerline_df to int64 to match vel_df
            centerline_df['Date'] = centerline_df['Date'].astype(int)

            # Merge the centerline_df and vel_df dataframes
            vel_df = pd.merge(vel_df, centerline_df, on=['Participant', 'Date', 'Location', 'Video', 'Capillary'], how='outer')

            with pd.option_context('display.max_colwidth', None) and pd.option_context('display.max_rows', None):
                print(vel_df)
                print(vel_df.dtypes)

            # Rename columns of the size dataframe
            size_df.rename(columns={'participant': 'Participant',
                                    'date':'Date',
                                    'location':'Location',
                                    'capnum':'Capillary',
                                    'area':'Area',
                                    'pressure':'Pressure'
                                    }, inplace=True)
            # Check if 'Weighted Average Slope' is in the columns of the velocity dataframe
            if 'Weighted Average Slope' in vel_df.columns:
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
            vel_df['Capillary'] = vel_df['Capillary'].replace({'a':''}, regex=True).replace({'b':''}, regex=True).replace({'c':''}, regex=True) 
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
    # Reorder entries by video number and then by capillary number including leading zeros
    total_big_df = total_big_df.sort_values(by=['Video', 'Capillary'], key=lambda x: x.str.zfill(2))
    
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
        # Create a folder for the total files if it does not exist
        os.makedirs(os.path.join(results_path, 'total'), exist_ok=True)
        # Save the dataframe as a csv
        total_big_df.to_csv(os.path.join(results_path, 'total', f'{participant}_{date}_total_data.csv'), index=False)
    
    return 0 

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    for i in range(9,28):
        if i == 24:
            continue
        participant = 'part' + str(i).zfill(2)
        main(participant, verbose=True, write=True)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))


