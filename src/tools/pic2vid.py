"""
Filename: pic2vid.py
-------------------------------------------------------------
This file turns a group of images into a video. it is primarily used as a
function but can be run in standalone versions by using the command 
line or an IDE. 
by: Marcus Forst
"""

import os
import time
import numpy as np
import cv2
import pandas as pd
from src.tools.get_images import get_images
import matplotlib.pyplot as plt

def frames_to_timecode(frame_number, frame_rate):
    """
    Method that converts frames to SMPTE timecode.
    Args:
        frame_number (int): Number of frames
        frame_rate (int/float): frames per second
        drop (bool): true if time code should drop frames, false if not
    
    Returns: SMPTE timecode as string, e.g. '01:02:12:32' or '01:02:12;32'
    """
    fps_int = int(round(frame_rate))
    # now split our frames into time code
    hours = int(frame_number / (3600 * fps_int) % 24)
    minutes = int((frame_number / (60 * fps_int)) % 60)
    seconds = int((frame_number / fps_int) % 60)
    return f'{str(hours).zfill(2)}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}:{str(frame_number).zfill(4)}'
def add_overlay(img, text, location):
    """ Add overlay to video frame with specific style"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 1
    line_type = 2
    cv2.putText(img, text, location, font, font_scale, font_color, thickness, line_type)
def add_focus_bar(img, focus):
    """ Add focus bar to video (work in progress)"""
    add_overlay(img, f'F:{round(focus, 1)}', (img.shape[1] - 175, 80))
    return 0
def add_scale_bar(img):
    add_overlay(img, f'100 um', (img.shape[1] -250, img.shape[0]-50))
    img[-100:-85,-274:-100] = 255
    return 0
def calculate_focus_measure(image,method='LAPE'):
    """ Quantify the focus of an image using the laplacian transform """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # optional
    if method == 'LAPE':
        if image.dtype == np.uint16:
            lap = cv2.Laplacian(image, cv2.CV_32F)
        else:
            lap = cv2.Laplacian(image, cv2.CV_16S)
        focus_measure = np.mean(np.square(lap))
    elif method == 'GLVA':
        focus_measure = np.std(image,axis=None)# GLVA
    else:
        focus_measure = np.std(image,axis=None)# GLVA
    return focus_measure
def extract_metadata(path, video):
    """input path: string; outputs pressure: string, frame rate: integer"""
    metadata = pd.read_excel(path)
    pressure = metadata.loc[(metadata['Video'] == video )| 
                            (metadata["Video"]== video + 'bp')| 
                            (metadata['Video']== video +'scan')]['Pressure'].values[0]
    frame_rate = metadata.loc[(metadata['Video'] == video )| 
                            (metadata["Video"]== video + 'bp')| 
                            (metadata['Video']== video +'scan')]['FPS'].values[0]
    
    

    return pressure, frame_rate

def pic2vid(path, images, participant = 'part_11', date = '230427', location = 'loc01',
            video_folder = 'vid1', color = False, compress = True, overlay = True):
    """
    Takes a list of image files or numpy array and makes a movie with overlays
    
    Args:
        images (list/np.array): The image data to be made into a video.
        participant (str): the participant who made the videos
        date (str): the date the data was collected
        location (str): the location of the data
        video_folder (str): the video number for that day
        color: bool
        compress: bool, whether to compress the video or not

    Returns: 
        int: 0 if successful

    Saves: video file in results folder
    """
    SET = 'set_01'
    images = np.array(images)
    output_path = '/hpc/projects/capillary-flow/results/videos'
    if overlay:
        metadata_path = os.path.join('hpc/projects/capillary-flow/data', participant, date, 'part_metadata', f'{participant}_{date}.xlsx')
        pressure, frame_rate = extract_metadata(metadata_path, video)
        print(frame_rate)
    else:
        frame_rate = 227.8/2
        pressure = 'TBD'
    if color:
        video_name = f'{SET}_{participant}_{date}_{location}_{video_folder}_color.avi'
    else:
        video_name = f'{SET}_{participant}_{date}_{location}_{video_folder}_gray.avi'
    frame = images[0]
    if compress:
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # avi compression
    else: 
        fourcc = 0
    if color:
        video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, 60, (frame.shape[1], frame.shape[0]), True)    
    else: 
        video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, 60, (frame.shape[1], frame.shape[0]), False)    

    print(frame.shape)
    for i in range(images.shape[0]):
        timecode = frames_to_timecode(i, frame_rate)
        img = images[i]
        focus_measure = calculate_focus_measure(img)
        # add pressure overlay
        add_overlay(img, f'P:{pressure}', (frame.shape[1]-150, 50))
        # add frame counter
        add_overlay(img, timecode, (frame.shape[1]//2 - 100, 50))
        # add set and sample overlay details
        set_string = str(SET).split('_')[0] + ": " + str(SET).split('_')[1]
        participant_string = str(participant)
        date_string = str(date)
        location_string = str(location)
        video_string = str(video_folder)

        add_overlay(img, f'{set_string}', (50, 50))
        add_overlay(img, f'{participant_string}', (50, 80))
        # add version overlay
        add_overlay(img, f'{date_string}', (50, 110))
        add_overlay(img, f'{location_string}', (50, 140))
        add_overlay(img, f'{video_string}', (50, 170))
        # TODO: add focus bar
        add_focus_bar(img, focus_measure)
        # TODO: add scale bar
        add_scale_bar(img)
        if color:
            img_color = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
            video.write(img_color.astype('uint8'))
            # plt.imshow(img_color)
            # plt.show()
        else:
            video.write(img.astype('uint8'))
    cv2.destroyAllWindows()
    video.release()
    return 0

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    input_folder = 'C:\\Users\\gt8mar\\capillary-flow\\data\\part_11\\230427\\vid1\\moco'
    images = get_images(input_folder)
    image_files = []
    for i in range(len(images)): 
        image = np.array(cv2.imread(os.path.join(input_folder, images[i]), cv2.IMREAD_GRAYSCALE))
        image_files.append(image)
    pic2vid(image_files, color= True)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
