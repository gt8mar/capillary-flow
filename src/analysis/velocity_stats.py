import os, platform
import pandas as pd
import matplotlib.pyplot as plt
from src.tools.load_name_map import load_name_map
from src.tools.parse_filename import parse_filename

def make_centerline_df(participant, date, location):
    """
    This function takes in the participant, date, and location
    and returns a dataframe with the length of capillaries.

    Args:
        participant (str): The participant number
        date (str): The date of the experiment
        location (str): The location on the finger

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
        if filename not in name_map['centerlines name'].values:
            print(f'{filename} not in name_map')
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
    print(name_map_df.head)

def check_correlations(csv_file_path, test=True):
    big_name_map = make_big_name_map('part17')
    if test:
        centerline_df = make_centerline_df('part17', '230502', 'loc01')
        
        # Adjust format of capillary number to match total data file 
        # Drop a from end of capillary number in vel_df
        centerline_df['Capillary'] = centerline_df['Capillary'].replace({'a':''}, regex=True).replace({'b':''}, regex=True)
        # Convert capillary number to int64 in vel_df
        centerline_df['Capillary'] = centerline_df['Capillary'].astype(int)
        # make date column an int
        centerline_df['Date'] = centerline_df['Date'].astype(int)

        
        print(centerline_df.head)
        # Import CSV into DataFrame
        df = pd.read_csv(csv_file_path)

        # merge centerline_df and df
        df = pd.merge(df, centerline_df, how='left', on=['Participant', 'Date', 'Location', 'Video', 'Capillary'])
    else:
        df = pd.read_csv(csv_file_path)
    
    # Use BP column to make systolic and diastolic blood pressure columns
    df['Systolic BP'] = df['BP'].apply(lambda x: x.split('/')[0])
    df['Diastolic BP'] = df['BP'].apply(lambda x: x.split('/')[1])
    # Make systolic and diastolic blood pressure columns integers
    df['Systolic BP'] = df['Systolic BP'].astype(int)
    df['Diastolic BP'] = df['Diastolic BP'].astype(int)
    # Print systolic and diastolic blood pressure columns
    print(df[['Systolic BP', 'Diastolic BP', 'Pulse']].head)

    # Use area and velocity columns to make velocity per diameter column
    df['Diameter'] = df['Area'] / df['Centerline Length']
    df['Velocity per Diameter'] = df['Velocity'] / df['Diameter']

 
    # Reorder columns to place systolic and diastolic blood pressure columns next to pulse column
    df = df[['Participant', 'Date', 'Location', 'Video', 'Pressure', 'Capillary', 'Diameter',  'Velocity', 
             'BP', 'Systolic BP', 'Diastolic BP', 'Pulse', 'Velocity per Diameter', 'Area','Centerline Length']]
    # list columns
    print(df.head)
    # make NaN values 0
    df.fillna(0, inplace=True)

    
    # Check for correlations between variables
    correlations = df.corr()
    fig, ax = plt.subplots(figsize=(10,10))
    ax.matshow(correlations)
    ax.set_xticks(range(len(correlations.columns)))
    ax.set_yticks(range(len(correlations.columns)))
    ax.set_xticklabels(correlations.columns)
    ax.set_yticklabels(correlations.columns)
    plt.xticks(rotation=90)
    plt.show()

    # run other analysis on the data
    df.describe()

    return correlations

# ------------------------- Main -------------------------
if __name__ == "__main__":
    data_path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\total\\part17_230502_total_data_MF231120.csv'
    correlations = check_correlations(data_path)
    print(correlations)