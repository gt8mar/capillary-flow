"""
Filename: pic2vid.py
-------------------------------------------------------------
This file turns a group of files into a video. It correctly orders misordered files.
by: Marcus Forst
"""

import os
import time
import numpy as np
import cv2
from src.tools.get_images import get_images

# UMBRELLA_FOLDER = 'C:\\Users\\ejerison\\Desktop\\data\\221010'
FILEFOLDER_PATH = "C:\\Users\\gt8mar\\Desktop\\data\\230425\\vid15bp"
DATE = "230425"
PARTICIPANT = "Participant10"
FOLDER_NAME = 'vid15bp'
SET = '01'
SAMPLE = '0077'
FRAME_RATE = 227.3

def frames_to_timecode(frame_number, frame_rate):
    """
    Method that converts frames to SMPTE timecode.
    :param frame_number: Number of frames
    :param frame_rate: frames per second
    :param drop: true if time code should drop frames, false if not
    :returns: SMPTE timecode as string, e.g. '01:02:12:32' or '01:02:12;32'
    """
    fps_int = int(round(frame_rate))
    # now split our frames into time code
    hours = int(frame_number / (3600 * fps_int) % 24)
    minutes = int((frame_number / (60 * fps_int)) % 60)
    seconds = int((frame_number / fps_int) % 60)
    return f'{hours}:{minutes}:{seconds}:{frame_number}'
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
    add_overlay(img, f'F:{focus}', (1050, 900))
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

def main(filefolder = FILEFOLDER_PATH, folder = FOLDER_NAME, date = DATE,
         participant = PARTICIPANT, frame_rate = FRAME_RATE):
    """
    this takes an image folder and a list of image files and makes a movie
    :param image_folder: string
    :param images: list of image filenames (strings)
    :return:
    """
    images = get_images(filefolder)
    video_name = f'{date}_{participant}_{folder}.avi'
    frame = cv2.imread(os.path.join(filefolder, images[0]))
    video = cv2.VideoWriter(video_name, 0, 60, (frame.shape[1], frame.shape[0]))
    print(frame.shape)
    for i in range(len(images)):
        timecode = frames_to_timecode(i, frame_rate)
        img = cv2.imread(os.path.join(filefolder, images[i]))
        focus_measure = calculate_focus_measure(img)
        # add pressure overlay
        add_overlay(img, 'P: 1.2 psi', (1050, 50))
        # add frame counter
        add_overlay(img, timecode, (200, 50))
        # add set and sample overlay
        add_overlay(img, f'{SET}.{SAMPLE}:', (50, 50))
        # add version overlay
        add_overlay(img, "HW: 01", (50, 80))
        add_overlay(img, "SW: 01", (50, 110))
        # TODO: add focus bar
        add_focus_bar(img, focus_measure)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()
    return 0

# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
