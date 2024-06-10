""" 
Filename: merge_metadata.py
-----------------------
This script loads metadata and merges it into a single file 
using the pandas library.

By: Juliette Levy
"""

import os
import time
import pandas as pd

"""
useful functions:

os.listdir(path) - returns a list of all files in the directory
os.makedirs(path) - creates a new directory
os.path.join(path, filename) - joins the path and filename together
function to check if a file is a folder: os.path.isdir(filename)
function to check if a file is a file: os.path.isfile(filename)
sorted(list, key=lambda x: x) - sorts a list of strings

string methods:
split() - splits a string into a list of strings
zfill() - pads a string with zeros
len() - returns the length of a string
example: filename.split('_') - splits a string on the underscore character so that 'filename_1' becomes ['filename', '1']

write filename strings using f'' notation:
f'filename_{variable}.csv'
example:
f'{participant}_{date}.csv'



function to get first row of a dataframe: df.iloc[0]
pd.read_csv(filename) - reads a csv file into a pandas dataframe
pd.concat([df1, df2]) - concatenates two dataframes
df[['col1', 'col2']] - slices a dataframe to include only the columns specified
df.median() - returns the median of all numerical columns in the dataframe
pd.to_csv(filename) - saves a dataframe to a csv file
pd.read_excel(filename) - reads an excel file into a pandas dataframe
print(df) - prints the dataframe to the console
df.head() - prints the first 5 rows of the dataframe to the console


"""

"""
def load metadata
go to /capillary-flow/metadata folder
for each file in the folder (in numerical order)
load the file into pandas

extract the first row (see below)
split systolic and diastolic blood pressure into separate columns
find median of systolic and diastolic blood pressure and pulse
# slice columns to include into new dataframe: 
df_new = df[['Participant','Date', 'FPS', 'SYS_BP_MED', 'DIA_BP_MED', 'Pulse Median',
    'Birthday', 'Height', 'Weight','Sex','Hypertension','Diabetes', 'Reynauds', 
    'Sickle Cell', 'SC Trait', 'Heart Disease', 'Other']] 

compile (concat) all dataframes into one
make a new folder /capillary-flow/metadata/merged (if it doesn't exist)
save it to /capillary-flow/metadata/merged/merged_metadata.csv
"""

METADATA_FOLDER_PATH = 'C:\\Users\\gt8mar\\capillary-flow\\metadata'

def load_first_row(metadata_path, verbose = False):
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
    """
    The following code will load metadata filenames into a list
    """
    for filename in os.listdir(METADATA_FOLDER_PATH):
        if filename == "part10_230516.xlsx":
            print("excluding part10_230516.xlsx")
            continue
        metadata_path = os.path.join(METADATA_FOLDER_PATH, filename) # full path to metadata file, merges folder path with the filename
        first_row = load_first_row(metadata_path)
        split_bp(first_row)
        

def split_bp(first):
    print(first_row["BP"])
    BP = first_row["BP"]
    SplitBP = BP.split('/')
    print(SplitBP)

    

        



    return 0



    


def main():
    """
    1. load metadata
    """
    load_metadata()
    return 0






"""
No need to edit below here
"""
if __name__ == "__main__":
    main()
    # print("--------------------")