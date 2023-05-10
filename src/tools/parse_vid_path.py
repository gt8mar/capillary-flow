"""
Filename: parse_vid_path.py
------------------------------------
This file takes in a path and returns the variables set, participant, date, and video.

By: Marcus Forst
"""

def parse_vid_path(path):
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
    # Split the path into its directory names
    dir_names = path.split(os.path.sep)
    
    # Extract the variables from the directory names
    set = dir_names[-4]
    participant = dir_names[-3]
    date = dir_names[-2]
    video = dir_names[-1]
    return SET, participant, date, video