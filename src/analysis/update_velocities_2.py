import os, platform
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import gaussian_filter
from src.analysis.make_velocities import find_slopes
from src.analysis.update_velocities import plot_velocities
from src.tools.parse_filename import parse_filename


PIX_UM = 2.44 #1.74


def calculate_age_on_date(date_str, birthday_str):
    """
    # # Example usage
# date = '230414'        # April 14, 2023
# birthday = '19981113'  # November 13, 1998
# age = calculate_age_on_date(date, birthday)
# print(f"Age on {date}: {age} years")

    """
    # Convert the date and birthday strings into date objects
    date_str = '20'+ date_str
    date = datetime.strptime(date_str, '%Y%m%d')
    birthday = datetime.strptime(birthday_str, '%Y%m%d')
    
    # Calculate the age
    age = date.year - birthday.year - ((date.month, date.day) < (birthday.month, birthday.day))

    return age


def update_velocities2(csv_path):
    if platform.system() == 'Windows':
        if 'gt8mar' in os.getcwd():
            results_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results'
            metadata_folder = 'C:\\Users\\gt8mar\\capillary-flow\\metadata'
        else:
            results_folder = 'C:\\Users\\gt8ma\\capillary-flow\\results'
            metadata_folder = 'C:\\Users\\gt8ma\\capillary-flow\\metadata'
    else:
        results_folder = '/hpc/projects/capillary-flow/results'
        metadata_folder = '/hpc/projects/capillary-flow/metadata'
# Import copy velocity file
# Read the csv file
    df = pd.read_csv(csv_path)

    # Get the participant, date, and part from the csv file name
    csv_filename = os.path.basename(csv_path)

    filename_no_ext = csv_filename.split('.')[0].replace('contrast_', '').replace('_background', '').replace('_seg', '')
    filename_list = filename_no_ext.split('_')
    # find participant
    participant = [item for item in filename_list if item.startswith('part')][0]
    date = [item for item in filename_list if (item.startswith('230') |
                                               item.startswith('231') |
                                               item.startswith('240') |
                                               item.startswith('241'))][0]
    
    metadata_name = f'{participant}_{date}.xlsx'
    # Read in the metadata
    metadata = pd.read_excel(os.path.join(metadata_folder, metadata_name), sheet_name = 'Sheet1') 

    Maxes = []
    # Iterate through each row
    for i in range(len(df)):
        # If the velocity is 0
        if df['Correct'][i] == 't':
            if df['Max'][i] == 't':
                Maxes.append([i, df['Velocity'][i]])
    # print(Maxes)
    if len(Maxes) == 0:
        print('No maxes found')
        # set max to highest velocity in df
        max = np.max(df['Velocity'])
    else:
        # max is the average of the maxes:
        max = np.mean([Max[1] for Max in Maxes])
    print(f'max = {max}')

    # make new column for corrected velocities
    df['Corrected Velocity'] = df['Velocity']

    for i in range(len(df)):
        video = df['Video'][i]
        video_metadata = metadata.loc[(metadata['Video'] == video) |
                                        (metadata['Video'] == video + 'bp') |
                                        (metadata['Video'] == video + 'scan')
                                        ]
        
        # Get the pressure for the video
        pressure = video_metadata['Pressure'].values[0]
        fps = video_metadata['FPS'].values[0]
        age = calculate_age_on_date(date, str(video_metadata['Birthday'].values[0]))
        df.loc[i, 'Age'] = age
        # print(f'age = {age}')
        sys_bp = video_metadata['BP'].values[0].split('/')[0]
        # print(f'sys_bp = {sys_bp}')
        df.loc[i, 'SYS_BP'] = sys_bp
        
            

        # Check each velocity row to see if it is true or false
        # if true, continue
        # if false, check if zero or max
        if (df['Correct'][i] == 'f' and df['Zero'][i]=='t'):
            # If zero, set to zero
            df.loc[i,'Corrected Velocity'] = 0
        elif(df['Correct'][i] == 'f' and df['Notes'][i]=='too slow'):
            if df['Max'][i] == 't':
                df.loc[i, 'Corrected Velocity'] = max
            else:
                ## elif too slow, load in kymograph
                filename = f"set01_{df['Participant'][i]}_{df['Date'][i]}_{df['Location'][i]}_{df['Video'][i]}_kymograph_{df['Capillary'][i]}.tiff"
                filepath = os.path.join(results_folder, 'kymographs', filename)
                kymograph = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                kymo_blur = gaussian_filter(kymograph, sigma = 2)
                # print(filename)
                slope = find_slopes(kymo_blur, filename, verbose=False, too_slow = True, write = False)
               
                if (slope == None):
                    # df.loc[i, 'Corrected Velocity'][i] = df.loc[i,'Velocity']
                    print('No slope found')
                elif (slope == 0):
                    print('no slope found')
                else:
                    um_slope = np.absolute(slope) *fps/PIX_UM
                    # print(f'velocity = {um_slope}')
                    df.loc[i,'Corrected Velocity'] = um_slope

        elif(df['Correct'][i] == 'f' and df['Max'][i]=='t'):
            ## elif max, set to max (set to 750? 1000? avg of maxes?), if greater than 1000 set to 1k?
            df.loc[i, 'Corrected Velocity'] = max

        elif(df['Correct'][i] == 'f' and df['Notes'][i]=='too fast'):
            ## elif too fast, load in kymograph
            filename = f"set01_{df['Participant'][i]}_{df['Date'][i]}_{df['Location'][i]}_{df['Video'][i]}_kymograph_{df['Capillary'][i]}.tiff"
            filepath = os.path.join(results_folder, 'kymographs', filename)
            kymograph = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            kymo_blur = gaussian_filter(kymograph, sigma = 2)
            # print(filename)
            slope = find_slopes(kymo_blur, filename, verbose=False, too_fast = True, write = True)
            
            

            if (slope == None):
                # df.loc[i, 'Corrected Velocity'][i] = df.loc[i,'Velocity']
                print('No slope found')                
            else:
                um_slope = np.absolute(slope) *fps/PIX_UM
                print(f'velocity = {um_slope}')
                df.loc[i,'Corrected Velocity'] = um_slope
    print(f'completed df for {csv_path}')        
    # print(df.head())

    # plot the corrected velocities organized by capillary:
    # plot_velocities(df, write = False, verbose = True)

    return df

if __name__ == '__main__':
    big_df_count = 0
    big_df = None
    # Usage example
    if platform.system() == 'Windows':
        if 'gt8mar' in os.getcwd():
            velocities_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\velocities'
        else:
            velocities_folder = 'C:\\Users\\gt8ma\\capillary-flow\\results\\velocities\\velocities'
        for csv_file in os.listdir(velocities_folder):
            if csv_file.endswith('Copy.csv'):
                csv_file_path = os.path.join(velocities_folder, csv_file)
                if big_df_count == 0:
                    big_df = update_velocities2(csv_file_path)
                    big_df_count += 1
                else:
                    big_df = pd.concat([big_df, update_velocities2(csv_file_path)])
        big_df.to_csv('C:\\Users\\gt8ma\\capillary-flow\\results\\velocities\\velocities\\big_df.csv', index=False)
    else:
        csv_file_path = '/hpc/projects/capillary-flow/results/velocities'
        if big_df_count == 0:
            big_df = update_velocities2(csv_file_path)
            big_df_count += 1
        else:
            big_df = pd.concat([big_df, update_velocities2(csv_file_path)])
        big_df.to_csv('/hpc/projects/capillary-flow/results/velocities/velocities/big_df.csv', index=False)
