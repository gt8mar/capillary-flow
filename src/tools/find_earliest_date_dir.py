"""
File: find_earliest_date_dir.py
------------------------------------------------------
This program finds the earliest date directory in the current directory.

By: Marcus Forst
"""

import os

def find_earliest_date_dir(path='F:\\Marcus\\data\\part13'):
    """
    Finds the earliest date directory in the current directory.
    
    Args:
        path (str): the path to the participant directory
    
    Returns:
        str: the earliest date directory in the participant directory
    """
    folder_list = os.listdir(path)
    # remove 'part_metadata' from the list if it is in the list
    if 'part_metadata' in folder_list:
        folder_list.remove('part_metadata')
    # read remaining folders as integers 
    folder_list = [int(folder) for folder in folder_list]
    # return the minimum value
    return str(min(folder_list))

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    find_earliest_date_dir()