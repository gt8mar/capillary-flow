import pandas as pd
import os

def filter_csv_by_filenames(source_csv, source_folder):
    """
    Filter a CSV file based on the filenames present in a target folder.

    Args:
        source_csv (str): The path to the source CSV file.
        source_folder (str): The path to the target folder containing the files.

    Returns:    
        None
    """
    # Define the target folder based on the source folder
    target_folder = f"{os.path.dirname(source_folder)}/big_{os.path.basename(source_folder)}"
    
    # Load the CSV file
    df = pd.read_csv(source_csv)
    
    # List all files in the target folder
    files_in_target = {file for file in os.listdir(target_folder)}
    
    # Filter the DataFrame to only include files that exist in the target folder
    filtered_df = df[df['Filename'].apply(lambda x: x in files_in_target)]
    
    # Create the new CSV file path
    new_csv_filename = f"big_{os.path.basename(source_csv)}"
    new_csv_path = os.path.join(os.path.dirname(source_csv), new_csv_filename)
    
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(new_csv_path, index=False)
    print(f"Filtered CSV has been saved to {new_csv_path}")

# Usage
source_csv = '/hpc/projects/capillary-flow/results/ML/240521_filename_df.csv'  # Replace with your actual CSV filename
source_folder = '/hpc/projects/capillary-flow/results/ML/kymographs'
filter_csv_by_filenames(source_csv, source_folder)
