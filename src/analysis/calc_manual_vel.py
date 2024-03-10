import os, platform
import pandas as pd
import numpy as np
from src.tools.parse_filename import parse_filename


def calc_manual_velocities(slopes_df_name, pixel_per_um, verbose = False):
    from math import tan, radians
    slopes_df = pd.read_csv(slopes_df_name)  # Replace 'your_data.csv' with the path to your data file
    participant, date, location, video, __= parse_filename(slopes_df_name)
    capillary = slopes_df_name.replace('..', '.').replace('_Results', '').replace('.csv', '').split('_')[-1]
    
    # Get the fps from the metadata
    metadata_folder = 'C:\\Users\\gt8mar\\capillary-flow\\metadata'
    # Get the metadata file name
    metadata_name = f'{participant}_{date}.xlsx'
    metadata = pd.read_excel(os.path.join(metadata_folder, metadata_name), sheet_name = 'Sheet1') 
    video_metadata = metadata[metadata['Video'] == video]
    fps = video_metadata['FPS'].values[0]
    pixel_per_um = 2.44
    
    # Calculate the average of the Angle column
    average_angle = slopes_df['Angle'].mean()
    # Convert average angle to radians for the tangent function
    average_angle_radians = radians(average_angle)

    # Calculate the velocity using the tangent of the average angle
    manual_velocity = tan(average_angle_radians) * fps / pixel_per_um
    # take the absolute value of the velocity
    manual_velocity = abs(manual_velocity)

    if verbose:
        print(f'Participant: {participant}, Date: {date}, Location: {location}, Video: {video}, Capillary: {capillary}, Manual: {manual_velocity}')

    # make a dataframe with the participant, date, location, video, capillary, and manual_velocity
    manual_velocity_df = pd.DataFrame({'Participant': [participant], 'Date': [date], 'Location': [location], 'Video': [video], 'Capillary': [capillary], 'Manual Velocity': [manual_velocity]})
    # set date to an int
    manual_velocity_df['Date'] = manual_velocity_df['Date'].astype(int)
    return manual_velocity_df

if __name__ == '__main__':
    participants = ['part22', 'part23']
    manual_dir_parent = 'C:\\Users\\gt8mar\\capillary-flow\\results\\kymographs\\by_hand'
    big_df_22_23_path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\big_df_22_23 240309 Copy.csv'
    big_df_22_23 = pd.read_csv(big_df_22_23_path)
    manual_df = pd.DataFrame()
    for participant in participants:
        manual_dir = os.path.join(manual_dir_parent, participant)
        for manual in os.listdir(manual_dir):
            if manual.endswith('.csv'):
                manual_path = os.path.join(manual_dir, manual)
                manual_velocity_df = calc_manual_velocities(manual_path, 2.44)
                # concatenate the manual_velocity_df to the manual_df
                manual_df = pd.concat([manual_df, manual_velocity_df], ignore_index=True)

                # big_df_22_23.merge(manual_velocity_df, 
                #                                on=['Participant', 'Date', 'Location', 'Video', 'Capillary'], 
                #                                how='left')
    # merge the manual_df with the big_df
    big_df_22_23 = big_df_22_23.merge(manual_df,
                                    on=['Participant', 'Date', 'Location', 'Video', 'Capillary'], 
                                    how='left')
    
    # # Print the dataframe with no skipping rows or columns
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(big_df_22_23)
    #     print(big_df_22_23.head())
    # # print(manual_df)

    # Save the big_df_22_23 to a csv
    new_path = big_df_22_23_path.replace(' 240309', ' 240309 manual')
    big_df_22_23.to_csv(new_path, index=False)
    
            
