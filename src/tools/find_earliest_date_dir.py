"""
File: find_earliest_date_dir.py
------------------------------------------------------
This program finds the earliest date directory in the current directory.

By: Marcus Forst
"""

import os

def find_earliest_date_dir(path):
    """
    Finds the earliest date directory in the current directory.
    
    Args:
        path (str): the path to the participant directory
    
    Returns:
        str: the earliest date directory in the participant directory
    """
    numeric_directories = [directory for directory in path if directory.isdigit()]

    if len(numeric_directories) == 0:
        return None
    elif len(numeric_directories) == 1:
        return numeric_directories[0]
    else:
        numeric_values = [int(directory) for directory in numeric_directories]
        return str(min(numeric_values))

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    find_earliest_date_dir()
