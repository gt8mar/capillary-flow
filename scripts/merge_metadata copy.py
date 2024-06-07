"""
Filename: merge_metadata.py
-----------------------
This script loads metadata and merges it into a single file 
using the pandas library.

By: Juliette Levy
"""

import os
import pandas as pd

METADATA_FOLDER_PATH = 'C:\\Users\\gt8mar\\capillary-flow\\metadata'
MERGED_FOLDER_PATH = os.path.join(METADATA_FOLDER_PATH, 'merged')

def load_first_row(metadata_path, verbose=False):
    """
    The following code will load metadata filenames into a list

    Args:
        metadata_path (str): The path to the metadata file

    Returns:
        first_row (pandas.Series): The first row of the metadata dataframe
    """
    metadata_df = pd.read_excel(metadata_path)
    first_row = metadata_df.iloc[0] # extracts the first row of the dataframe
    if verbose:
        print(first_row)
    return first_row

def load_metadata():
    final_metadata_df = pd.DataFrame()
    
    for filename in os.listdir(METADATA_FOLDER_PATH):
        if filename == "part10_230516.xlsx":
            print("excluding part10_230516.xlsx")
            continue
        if filename.endswith('.xlsx'):
            metadata_path = os.path.join(METADATA_FOLDER_PATH, filename) # full path to metadata file, merges folder path with the filename
            first_row = load_first_row(metadata_path)
            final_metadata_df = pd.concat([final_metadata_df, first_row.to_frame().T], ignore_index=True)
    
    if not os.path.exists(MERGED_FOLDER_PATH):
        os.makedirs(MERGED_FOLDER_PATH)
    
    final_metadata_df.set_index('Participant', inplace=True)
    final_metadata_df.to_csv(os.path.join(MERGED_FOLDER_PATH, 'merged_metadata.csv'))
    print("Merged metadata saved to:", os.path.join(MERGED_FOLDER_PATH, 'merged_metadata.csv'))
    
    return final_metadata_df

def main():
    """
    1. load metadata
    """
    final_metadata_df = load_metadata()
    print(final_metadata_df.head())
    return 0

"""
No need to edit below here
"""
if __name__ == "__main__":
    main()
