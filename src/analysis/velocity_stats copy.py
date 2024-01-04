import os, platform
import pandas as pd
import matplotlib.pyplot as plt
# from src.tools.load_name_map import load_name_map
from src.tools.parse_filename import parse_filename

def load_name_map(participant, date, location, version = 'centerlines'):
    """
    Loads the name map from the segmented folder, removes the prefix from the cap name,
    and adds a column with the actual capillary name.

    Args:
        participant (str): participant number
        date (str): date of the experiment
        location (str): location on the finger   
        version (str): version of the name map to load. Default: 'centerlines'
            options: 'centerlines', 'centerline', 'kymograph', 'kymo, 'kymographs'
    
    Returns:
        name_map (pd.DataFrame): DataFrame containing the name map 
            with the columns 'centerlines name', 'cap name', 'cap name short';
            'cap name short' gives the short index of the capillaries
    """
    column_names  = ['centerlines name', 'cap name']
    # name_map_folder = os.path.join(path, 'segmented')
    name_map_name = f'{participant}_{date}_{location}_name_map.csv'
    if platform.system() == 'Windows':
        name_map_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results\\archive\\231205\\size\\name_maps'
    else:
        name_map_folder = '/hpc/projects/capillary-flow/results/size/name_maps'
    name_map = pd.read_csv(os.path.join(name_map_folder, name_map_name), names = column_names)

    # Remove 'translated_' from all elements in the columns:
    name_map = name_map.apply(lambda x: x.str.replace('translated_', ''))
    if version in ('kymograph','kymo','kymographs'):
        name_map = name_map.apply(lambda x: x.str.replace('centerline_coords', 'kymograph'))
        name_map = name_map.apply(lambda x: x.str.replace('csv', 'tiff'))
    elif version in ('centerline','centerlines'):
        pass
    else:
        raise ValueError('Version not recognized. Use "centerline" or "kymograph".')
    # duplicate the cap name column
    name_map['cap name short'] = name_map['cap name']
    # remove the prefix from the short cap name
    name_map['cap name short'] = name_map['cap name short'].apply(lambda x: x.split("_")[-1].split(".")[0])
    return name_map

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

    results_path = os.path.join(capillary_path, 'results', 'archive', '231205')
    centerline_path = os.path.join(results_path, 'centerlines')

    name_map = load_name_map(participant, date, location, version = 'centerlines')
    print(name_map.head)
    # Create a dataframe with columns 'Participant', 'Date', 'Location', 'Video', 'Capillary', 'Centerline Length'
    centerline_df = pd.DataFrame(columns=['Participant', 'Date', 'Location', 'Video', 'Capillary', 'Centerline Length'])
    for filename in os.listdir(centerline_path):
        if filename.startswith(f'set_01_{participant}') and filename.endswith('.csv'):
            # switch to set01
            new_filename = filename.replace('set_01', 'set01')
        elif filename.startswith(f'set01_{participant}') and filename.endswith('.csv'):
            new_filename = filename
            _, _, _, video, _ = parse_filename(new_filename)
            print(f'video is {video}')
        else:
            continue
        centerline_coords = pd.read_csv(os.path.join(centerline_path, filename))
        # The number of rows of the centerline_coords dataframe is the length of the capillary
        cap_length = len(centerline_coords)
        print(f'capillary length is {cap_length}')
        # Check if filename is in name_map
        if new_filename not in name_map['centerlines name'].values:
            # print(f'{new_filename} not in name_map')
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
    print('works up to here ------------------------------------------------------------------------')
    print(centerline_df.head)
    return centerline_df

def make_big_name_map(participant, verbose=False):
    if platform.system() == 'Windows':
        name_map_path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\archive\\231205\\size\\name_maps'
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
    if verbose:
        print(name_map_df.head)
    return name_map_df

def check_correlations(csv_file_path, test=True):
    df = pd.read_csv(csv_file_path)

    if test:
        # Get participant values from filename
        basename = os.path.basename(csv_file_path)
        file_list = basename.split('.')[0]
        file_list = file_list.split('_')
        participant = file_list[0]
        date = file_list[1]
        big_name_map = make_big_name_map(participant)
    #     # big_name_map = make_big_name_map('part17')
    #     # centerline_df = make_centerline_df('part17', '230502', 'loc01')
        
        # Search big_name_map for files that have 'loc' in them and print value following the 'loc'
        # print(big_name_map)
        locations = big_name_map[big_name_map['centerlines name'].str.contains('loc')]['centerlines name'].str.split('_').str[3].values
        # remove redundant values
        locations = list(dict.fromkeys(locations))
        for location in locations:
            print(f'making centerline_df for {location}')
            centerline_df = make_centerline_df(participant, date, location)

            # Adjust format of capillary number to match total data file 
            # Drop a from end of capillary number in vel_df
            centerline_df['Capillary'] = centerline_df['Capillary'].replace({'a':''}, regex=True).replace({'b':''}, regex=True)
            # Convert capillary number to int64 in vel_df
            centerline_df['Capillary'] = centerline_df['Capillary'].astype(int)
            # make date column an int
            centerline_df['Date'] = centerline_df['Date'].astype(int)

            
            print(centerline_df.head)

            # merge centerline_df and df
            df = pd.merge(df, centerline_df, how='left', on=['Participant', 'Date', 'Location', 'Video', 'Capillary'])
    else:
        # participant, date, location, video, _ = parse_filename(csv_file_path)
        # big_name_map = make_big_name_map(participant)
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
    # correlations = 0
    return correlations

# ------------------------- Main -------------------------
if __name__ == "__main__":
    total_path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\archive\\231205\\total' # part17_230502_total_data_MF231120.csv
    for filename in os.listdir(total_path):
        if filename.endswith('.csv'):
            print(filename)
            data_path = os.path.join(total_path, filename)
            correlations = check_correlations(data_path, test=True)
            print(correlations)