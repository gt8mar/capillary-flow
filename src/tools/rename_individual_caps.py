import time
import os
import pandas as pd
import platform

def rename(path='D:\\gabby_debugging\\part10\\230425\\loc02'):
    individual_caps_translated_fp = os.path.join(path, "segmented", "hasty", "individual_caps_translated")
    individual_caps_original_fp = os.path.join(path, "segmented", "hasty", "individual_caps_original")

    if platform.system() == 'Windows':
        rename_map_fp = "D:\\gabby_debugging\\part10\\230425\\loc02\\rename_map.csv"
    else:
        rename_map_fp = "/hpc/projects/capillary-flow/results/size/rename_map.csv"
    df = pd.read_csv(rename_map_fp, header=None)

    #rename translated in loc folders
    for file in os.listdir(individual_caps_translated_fp):
        matching_row = df[df.iloc[:, 0] == file]
        if not matching_row.empty:
            new_filename = matching_row.iloc[0, 1]
            original_path = os.path.join(individual_caps_translated_fp, file)
            new_path = os.path.join(individual_caps_translated_fp, new_filename)
            os.rename(original_path, new_path)

    #rename original in loc folders
    for file in os.listdir(individual_caps_original_fp):
        matching_row = df[df.iloc[:, 0] == file]
        if not matching_row.empty:
            new_filename = matching_row.iloc[0, 1]
            original_path = os.path.join(individual_caps_original_fp, file)
            new_path = os.path.join(individual_caps_original_fp, new_filename)
            os.rename(original_path, new_path)

if __name__ == "__main__":
    ticks = time.time()
    rename()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))