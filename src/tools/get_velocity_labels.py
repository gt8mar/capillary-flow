import time
import pandas as pd

def filter_csv(input_path="C:\\Users\\Luke\\Downloads\\big_df - Sheet1.csv"):
    filtered_output_path = "C:\\Users\\Luke\\Downloads\\filtered_df.csv"
    filename_output_path = "C:\\Users\\Luke\\Downloads\\filename_df.csv"

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_path)
    
    # Filter rows where 'Correct' or 'Zero' columns contain 't'
    filtered_df = df[(df['Correct'] == 't') | (df['Zero'] == 't')]

    # Further filter out rows where 'Notes', 'Notes2', or 'Drop' contain 'drop' or 'unclear'
    filtered_df = filtered_df[
        ~(filtered_df['Notes'].str.contains('drop', case=False, na=False) |
          filtered_df['Notes2'].str.contains('drop', case=False, na=False) |
          filtered_df['Drop'].str.contains('drop', case=False, na=False) |
          filtered_df['Notes'].str.contains('unclear', case=False, na=False) |
          filtered_df['Notes2'].str.contains('unclear', case=False, na=False) |
          filtered_df['Drop'].str.contains('unclear', case=False, na=False))
    ]

    # Save the filtered rows to a new CSV file
    filtered_df.to_csv(filtered_output_path, index=False)

    # Create the additional CSV with 'Filename' and 'Velocity' columns
    filename_df = pd.DataFrame({
        'Filename': 'set01_' + filtered_df['Participant'].astype(str) + '_' + filtered_df['Date'].astype(str) + '_' 
        + filtered_df['Location'].astype(str) + '_' + filtered_df['Video'].astype(str) + '_kymograph_' + filtered_df['Capillary'].astype(str) +'.tiff',
        'Corrected Velocity': filtered_df['Corrected Velocity']
    })
    
    # Save the additional DataFrame to the second CSV file
    filename_df.to_csv(filename_output_path, index=False)




"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    filter_csv()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))