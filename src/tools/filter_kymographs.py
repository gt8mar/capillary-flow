import time
import os
import pandas as pd
import shutil


def filter_kymographs(input_csv='/hpc/project/capillary-flow/results/ML/240521_filename_df.csv', kymograph_dir='/hpc/project/capillary-flow/results/kymographs'):
    ml_kymograph_dir = '/hpc/project/capillary-flow/results/ML/kymographs'
    os.makedirs(ml_kymograph_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Iterate through the rows in the CSV
    for _, row in df.iterrows():
        filename = row['Filename']
        source_path = os.path.join(kymograph_dir, filename)
        destination_path = os.path.join(ml_kymograph_dir, filename)

        shutil.copy2(source_path, destination_path)

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    filter_kymographs()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))