"""
Filename: load_csv_list.py
------------------------------------------------------
This program loads csv files from a folder into a list.

By: Marcus Forst
"""

import os
import numpy as np

def load_csv_list(path, dtype = int):
    """ 
    Loads csv files from a folder into a list 

    Args:
        path (str): the path to the folder to be searched

    Returns:
        list: a list of csv files (dtype= 2D np.array) in the given path
    """
    csv_names = [file for file in os.listdir(path) if file.endswith(".csv")]
    csv_list = []
    for name in csv_names:
        file = np.loadtxt(os.path.join(path, name), delimiter = ',', dtype = dtype)
        csv_list.append(file) 
    print(f"{path} list length: {len(csv_list)}")
    return csv_list