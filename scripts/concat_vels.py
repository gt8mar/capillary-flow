import os
import pandas as pd

def concatenate_csv_files(folder_path, output_file):
    # List to hold dataframes
    dataframes = []

    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # Read the CSV file and append to the list
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Concatenate all dataframes in the list
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Save the concatenated dataframe to a new CSV file
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    folder_path = 'C:\\Users\\gt8ma\\capillary-flow\\results\\velocities\\part40-48'
    output_file = 'C:\\Users\\gt8ma\\capillary-flow\\results\\velocity_df_part40_to_part48.csv'
    concatenate_csv_files(folder_path, output_file)
