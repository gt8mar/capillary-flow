import os
import numpy as np

def load_csv_list(path, dtype = int):
    """ Loads csvs from a folder into a list """
    print(f"Loading csvs from {path}")
    csv_names = [file for file in os.listdir(path) if file.endswith(".csv")]
    csv_list = []
    for name in csv_names:
        file = np.loadtxt(os.path.join(path, name), delimiter = ',', dtype = dtype)
        csv_list.append(file) 
    print(f"list shape: {np.shape(csv_list)}")
    return csv_list