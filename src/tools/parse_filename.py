"""
Filename: parse_filename.py
---------------------------
This module contains a function that parses the filename of an image into its 
participant, date, location, and video number.

By: Marcus Forst
"""

import os
import pandas as pd

def parse_filename(filename):
    """
    Parses the filename of an image into its participant, date, location and video number.

    Args:
        filename (str): Filename of the image. format: set_participant_date_video_background.tiff
    
    Returns:
        participant (str): Participant number
        date (str): Date of the video
        location (str): Location of the video
        video (str): Video number
        file_prefix (str): Prefix of the filename. format: set_participant_date_location_video
    """
    filename_no_ext = filename.split('.')[0].replace('contrast_', '').replace('_background', '').replace('_seg', '')
    filename_list = filename_no_ext.split('_')
    # find participant
    participant = [item for item in filename_list if item.startswith('part')][0]
    date = [item for item in filename_list if (item.startswith('230') |
                                               item.startswith('231') |
                                               item.startswith('240') |
                                               item.startswith('241'))][0]
    video = [item for item in filename_list if item.startswith('vid')] [0]
    # get location from metadata
    metadata = pd.read_excel(os.path.join("/hpc/projects/capillary-flow/metadata", str(participant) + "_" + str(date) + ".xlsx"))
    # make location column entries into strings
    location = metadata.loc[(metadata['Video'] == video )| 
                            (metadata["Video"]== video + 'bp')| 
                            (metadata['Video']== video +'scan')]['Location'].values[0]
    
    if str(location) == "Temp" or str(location) == "Ex":
        location = "loc" + str(location)
    else:
        location = "loc" + str(location).zfill(2)
    
    file_prefix = f'set01_{participant}_{date}_{location}_{video}'
    return participant, date, location, video, file_prefix

if __name__ == "__main__":
    participant, date,location, video, file_prefix = parse_filename(
        "set01_part01_230414_loc02_vid01bp_contrast_seg.tiff")
