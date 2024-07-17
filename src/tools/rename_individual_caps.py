import time
import os
import pandas as pd
import shutil

def rename(path):  
    individual_caps_translated_fp = os.path.join(path, "segmented", "hasty", "individual_caps_translated")
    individual_caps_original_fp = os.path.join(path, "segmented", "hasty", "individual_caps_original")

    participant = os.path.basename(os.path.dirname(os.path.dirname(path)))
    date = os.path.basename(os.path.dirname(path))
    location = os.path.basename(path)
    cap_names_csv = os.path.join('/hpc/projects/capillary-flow/results/size/name_csvs', participant + '_' + date + '_' + location + '_cap_names.csv')

    df = pd.read_csv(cap_names_csv)

    # Make new named folders
    renamed_original_fp = os.path.join(path, "segmented", "hasty", "renamed_individual_caps_original")
    os.makedirs(renamed_original_fp, exist_ok=True)
    renamed_translated_fp = os.path.join(path, "segmented", "hasty", "renamed_individual_caps_translated")
    os.makedirs(renamed_translated_fp, exist_ok=True)

    # Iterate through every row in the csv
    for index, row in df.iterrows():
        file = row.iloc[0]
        original_path = os.path.join(individual_caps_original_fp, file)
        translated_path = os.path.join(individual_caps_translated_fp, file)
        
        # Check if column 1 is empty
        if len(row) == 1:
            # Copy the file into the new folder
            shutil.copy(original_path, os.path.join(renamed_original_fp, file))
            shutil.copy(translated_path, os.path.join(renamed_translated_fp, file))
        elif pd.isnull(row.iloc[1]):
            shutil.copy(original_path, os.path.join(renamed_original_fp, file))
            shutil.copy(translated_path, os.path.join(renamed_translated_fp, file))
        else:
            # Copy the file into the new folder with the new name
            new_filename = file[:-7] + f"{row.iloc[1]:02d}" + 'a.png' 
            shutil.copy(original_path, os.path.join(renamed_original_fp, new_filename))
            shutil.copy(translated_path, os.path.join(renamed_translated_fp, new_filename))


if __name__ == "__main__":
    ticks = time.time()
    rename()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))