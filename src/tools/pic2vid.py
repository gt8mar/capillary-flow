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
from src.tools import get_images
import matplotlib.pyplot as plt
from matplotlib.pylab import cm

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
    add_overlay(img, f'F:{round(focus, 1)}', (img.shape[1] - 175, 900))
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
def extract_metadata(path):
    """input path: string; outputs pressure: string, frame rate: integer"""
    shifts = pd.read_csv(path, header=None).transpose()
    shifts.columns  = shifts.iloc[0]
    shifts = shifts.drop(shifts.index[0])
    pressure = shifts['Pressure'][1].strip(';')
    exposure = shifts['Exposure'][1].strip(';')
    frame_rate = 100000//int(exposure)  # this rounds the frame-rate
    return pressure, frame_rate

def main(images, SET = 'set_01', sample = 'sample_001', color = False, compress = True):
    """
    takes a list of image files or numpy array and makes a movie with overlays
    :param images: list of images or numpy array of images
    :param SET: string
    :param sample: string
    :param color: bool
    :param compress: bool, whether to compress the video or not
    :return: 0
    """
    images = np.array(images)
    metadata_path = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\raw', str(SET), str(sample), 'metadata.txt')
    output_path = 'C:\\Users\\gt8mar\\capillary-flow\\results'
    pressure, frame_rate = extract_metadata(metadata_path)
    print(frame_rate)
    video_name = f'{SET}_{sample}.avi'
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
        # add set and sample overlay
        set_string = str(SET).split('_')[0] + ": " + str(SET).split('_')[1]
        sample_string = str(sample).split('_')[0] + ": " + str(sample).split('_')[1]
        add_overlay(img, f'{set_string}', (50, 50))
        add_overlay(img, f'{sample_string}', (50, 80))
        # add version overlay
        add_overlay(img, "HW: 01", (50, 110))
        add_overlay(img, "SW: 01", (50, 140))
        # TODO: add focus bar
        add_focus_bar(img, focus_measure)
        # TODO: add scale bar
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
    SET = 'set_01'
    sample = 'sample_001'
    input_folder = str(os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'B_stabilized\\vid'))
    images = get_images.main(input_folder)
    image_files = []
    for i in range(len(images)):
        image = np.array(cv2.imread(os.path.join(input_folder, images[i]), cv2.IMREAD_GRAYSCALE))
        image_files.append(image)
    main(image_files, SET, sample, color= True)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
