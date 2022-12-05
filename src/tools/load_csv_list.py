import os
import numpy as np

def load_csv_list(path):
    """ Loads csvs from a folder into a list """
    csv_names = [file for file in os.listdir(path) if file.endswith(".csv")]
    csv_list = []
    for name in csv_names:
        file = np.loadtxt(os.path.join(path, name), delimiter = ',', dtype = int)
        csv_list.append(file) 
    return csv_list