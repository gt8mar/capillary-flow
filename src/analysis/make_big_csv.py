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

def main(participant, verbose=False):
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
    # Create output folders
    if platform.system() == 'Windows':
        capillary_path = 'C:\\Users\\gt8ma\\capillary-flow'
    else:
        capillary_path = '/hpc/projects/capillary-flow'

    participant_path = os.path.join(capillary_path, 'data', participant)
    date = find_earliest_date_dir(participant_path)
    locations = os.listdir(os.path.join(capillary_path, 'data', participant, date))
    # centerline_folder = os.path.join(path, 'centerlines')

    metadata_path = os.path.join(capillary_path, 'metadata')
    results_path = os.path.join(capillary_path, 'results')
    size_path = os.path.join(results_path, 'size', 'size_data')
    vel_path = os.path.join(results_path, 'velocities')

    # Load the CSV files into pandas dataframes
    size_df = pd.read_csv(size_path)
    vel_df = pd.read_csv(vel_path)
    metadata_df = pd.read_csv(os.path.join(metadata_path, f'{participant}_{date}.xlsx'))

    if verbose:
        print(size_df.head)
        print(vel_df.head)
        print(metadata_df.head)

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main('part09', verbose=True)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))


