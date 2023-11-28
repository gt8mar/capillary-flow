"""
Filename: load_name_map.py
-------------------------------------------- 

By: Marcus Forst
"""

import os
import pandas as pd
import platform
from src.tools.parse_path import parse_path


# Rename centerline files:
def load_name_map(participant, date, location, version = 'centerlines'):
    """
    Loads the name map from the segmented folder, removes the prefix from the cap name,
    and adds a column with the actual capillary name.

    Args:
        participant (str): participant number
        date (str): date of the experiment
        location (str): location on the finger   
        version (str): version of the name map to load. Default: 'centerlines'
            options: 'centerlines', 'centerline', 'kymograph', 'kymo, 'kymographs'
    
    Returns:
        name_map (pd.DataFrame): DataFrame containing the name map 
            with the columns 'centerlines name', 'cap name', 'cap name short';
            'cap name short' gives the short index of the capillaries
    """
    column_names  = ['centerlines name', 'cap name']
    # name_map_folder = os.path.join(path, 'segmented')
    name_map_name = f'{participant}_{date}_{location}_name_map.csv'
    if platform.system() == 'Windows':
        name_map_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results\\size\\name_maps'
    else:
        name_map_folder = '/hpc/projects/capillary-flow/results/size/name_maps'
    name_map = pd.read_csv(os.path.join(name_map_folder, name_map_name), names = column_names)

    # Remove 'translated_' from all elements in the columns:
    name_map = name_map.apply(lambda x: x.str.replace('translated_', ''))
    if version in ('kymograph','kymo','kymographs'):
        name_map = name_map.apply(lambda x: x.str.replace('centerline_coords', 'kymograph'))
        name_map = name_map.apply(lambda x: x.str.replace('csv', 'tiff'))
    elif version in ('centerline','centerlines'):
        pass
    else:
        raise ValueError('Version not recognized. Use "centerline" or "kymograph".')
    # duplicate the cap name column
    name_map['cap name short'] = name_map['cap name']
    # remove the prefix from the short cap name
    name_map['cap name short'] = name_map['cap name short'].apply(lambda x: x.split("_")[-1].split(".")[0])
    return name_map