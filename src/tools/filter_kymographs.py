import time
import os
import pandas as pd
import shutil
import cv2


def filter_kymographs(input_csv='/hpc/projects/capillary-flow/results/ML/240521_filename_df.csv', kymograph_dir='/hpc/projects/capillary-flow/results/ML/big_kymographs'):
    ml_kymograph_dir = '/hpc/projects/capillary-flow/results/ML/kymographs'
    os.makedirs(ml_kymograph_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Iterate through the rows in the CSV
    rows_to_remove = []  # List to store indices of rows to be removed
    for index, row in df.iterrows():
        filename = row['Filename']
        source_path = os.path.join(kymograph_dir, filename)
        destination_path = os.path.join(ml_kymograph_dir, filename)

        # Check if source_path exists
        if os.path.exists(source_path):
            img = cv2.imread(source_path)
            width, height = img.shape[:2]
            if width > 128 and height > 128:
                # Copy the file
                shutil.copy2(source_path, destination_path)
            else:
                rows_to_remove.append(index)
        else:
            # If source_path does not exist, add the index to rows_to_remove
            rows_to_remove.append(index)

    # Remove rows from DataFrame
    df = df.drop(rows_to_remove)
    
    # Write filtered DataFrame back to CSV
    df.to_csv(input_csv, index=False)



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