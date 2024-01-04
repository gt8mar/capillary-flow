"""
Filename: centerline_lengths.py
------------------------------------------------------
This function takes a directory of centerline files and returns a DataFrame with the 
filenames and the number of rows in each file.

By: Marcus Forst
"""

import os, platform 
import pandas as pd

def centerline_length_summary(directory):
    # List to store file details
    file_details = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            
            # Read the CSV file and count the rows
            try:
                df = pd.read_csv(file_path)
                row_count = len(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            # Append the details to the list
            file_details.append({'Filename': filename, 'RowCount': row_count})

    # Convert the list to a DataFrame
    summary_df = pd.DataFrame(file_details)

    return summary_df

if __name__ == '__main__':
    if platform.system() == 'Windows':
        # Define the directory
        directory = 'C:\\Users\\gt8mar\\capillary-flow\\results\\archive\\231205\\centerlines'

    # Call the function
    summary_df = centerline_length_summary(directory)

    # Print the DataFrame
    print(summary_df)