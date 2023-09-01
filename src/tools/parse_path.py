"""
Filename: parse_path.py
------------------------------------
This file takes in a path and returns the variables set, participant, date, and video.

By: Marcus Forst
"""

import os

def parse_path(path, video_path=False):
    """
    This file takes in a path and returns the variables set, participant, date, and video.

    Args:
        path (str): path to a video folder
    Returns:
        SET (str): the set from which the data came
        participant (str): the participant who made the videos
        date (str): the date the data was collected
        video (str): the video number for that day
    """
    SET = 'set_01'

    # Split the path into its directory names
    dir_names = path.split(os.path.sep)
    
    # Extract the variables from the directory names
    participant = [item for item in dir_names if item.startswith('part')][0]

    date = [item for item in dir_names if (item.startswith('230') |
                                               item.startswith('231') |
                                               item.startswith('240') |
                                               item.startswith('241'))][0]   
    location = [item for item in dir_names if item.startswith('loc')][0]
    # remove bp from the video name
    if video_path:
        video = [item for item in dir_names if item.startswith('vid')][0]
        video = dir_names[-1].replace('bp', '')
        video = video.replace('scan', '')
        file_prefix = f'{SET}_{participant}_{date}_{location}_{video}'
    else: 
        video=None
        file_prefix = f'{SET}_{participant}_{date}_{location}'
    return participant, date, location, video, file_prefix