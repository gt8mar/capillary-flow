import os
import shutil
import pandas as pd


# Set your source and destination directories
source_dir = 'C:\\Users\\gt8mar\\capillary-flow\\results\\kymographs'
destination_dir = 'C:\\Users\\gt8mar\\capillary-flow\\results\\kymographs\\tricky_kymographs'
tricky_kymo_df = 'C:\\Users\\gt8mar\\capillary-flow\\tricky_kymos.csv'

# Create the destination folder if it does not exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Function to generate filenames and copy files
def copy_images(df, source_dir, destination_dir):
    for _, row in df.iterrows():
        filename = f"set01_{row['Participant']}_{row['Date']}_{row['Location']}_{row['Video']}_kymograph_{row['Capillary']}.tiff"
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)
        
        # Copy the file if it exists
        if os.path.exists(source_path):
            shutil.copy2(source_path, destination_path)
            print(f"Copied: {filename}")
        else:
            print(f"File not found: {filename}")

# Apply the function
df = pd.read_csv(tricky_kymo_df)
copy_images(df, source_dir, destination_dir)