"""
Filename: gettingradii.py
-------------------------
This file calculates the average radii of the capillaries using the centerline data. All averages are put into a new csv file.
By: Marcus Forst
"""   

import os
import pandas as pd   

DATA_PATH = 'J:\\frog\\data'
RESULTS_PATH = 'J:\\frog\\results\\average_vessel_thickness.csv'

average_vessel_thickness = []

for date_folder in os.listdir(DATA_PATH):
    date_folder_path = os.path.join(DATA_PATH, date_folder)
    if os.path.isdir(date_folder_path):
        for root, dirs, files in os.walk(date_folder_path):
            if "archive" in root.split(os.path.sep):
                continue
            if os.path.basename(root).startswith("Frog"):
                # print(date_folder_path) #printed the righ amount of times
                for subdir in dirs:
                    if "archive" in date_folder_path.split(os.path.sep):
                        continue
                    if subdir.startswith('Left') or subdir.startswith('Right'):
                        centerlines_path = os.path.join(root, subdir, 'centerlines', 'coords')
                        print(centerlines_path) #prints the path to the centerlines folder
                        rbc_count_path = os.path.join(root, subdir, "rbc_count")
                        # print(f'it is priting the path that brings you to the csv file: {rbc_count_path}') #printed the righ amount of times

                        if os.path.exists(centerlines_path):
                            for file in os.listdir(centerlines_path):
                                if file.endswith('.csv'):
                                    centerline_file_path = os.path.join(centerlines_path, file)
                                    # print(centerline_file_path)
                                    # for file in os.listdir(centerlines_path):
                                    try:
                                        df = pd.read_csv(centerlines_path) #creates a dataframe by reading the csv file 
                                        average_value = df.iloc[:, 2].mean()           
                                        df.to_csv(centerlines_path, index=False)
                                        try:
                                            df.to_csv(centerlines_path, index=False)       
                                            print("done")    
                                        except PermissionError:                 
                                            print("permission error")
                                    except PermissionError as e:
                                        print(f'permission denied: {centerlines_path} - {e}')
                                    except Exception as e:
                                        print(f'error processing file {centerlines_path} - {e}')
                                
                                           
                                     

                        if os.path.exists(rbc_count_path):
                            # print("reached this") #yes, the file path now exists
                            for file in os.listdir(rbc_count_path):
                                if file.endswith('.csv'):
                                    rbc_file_path = os.path.join(rbc_count_path, file)
                                    # print(rbc_file_path) #prints the file path to the csv file we ultimately want
                                    df = pd.read_csv(rbc_file_path) #creates a dataframe by reading the csv file 
                                    df['Date'] = date_folder #'Date' is name of new column, date_folder is the value
                                    df['Frog'] = os.path.basename(root) #'Frog' is name of new column, os.path.basename(root) is the value
                                    df['Side'] = subdir
                                    # df['Concat'] = average_path
                                    average_vessel_thickness.append(df)

master_df = pd.concat(average_vessel_thickness, ignore_index=True)
master_df.to_csv(RESULTS_PATH, index=False)
print("Done!")

                            