"""
Filename: get_shifts.py
-------------------------
This file uses the imagej log file to get the shifts of the image. 
By: Marcus Forst
"""   

import os
import pandas as pd   
    
def get_shifts(input_folder):
    """ Calculates and then returns max shifts """
    shifts = pd.read_csv(os.path.join(input_folder, 'Results.csv'))
    gap_left = shifts['x'].max()
    gap_right = shifts['x'].min()
    gap_bottom = shifts['y'].min()
    gap_top = shifts['y'].max()
    return gap_left, gap_right, gap_bottom, gap_top