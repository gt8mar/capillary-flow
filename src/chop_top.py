"""
Filename: chop_top.py
-------------------------------------------------------------
This file turns a group of files into a group of files that are slightly smaller
by: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
"""


import cv2
import os
import glob
import re
import time
import numpy as np

UMBRELLA_FOLDER = 'C:\\Users\\gt8mar\\Desktop\\data\\221010'
DATE = "221010"
PARTICIPANT = "Participant3"

def tryint(s):
    try:
        return int(s)
    except:
        return s
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]
def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
def get_images(FILEFOLDER):
    """
    this function grabs image names, sorts them, and puts them in a list.
    :param FILEFOLDER: string
    :return: images: list of images
    """
    images = [img for img in os.listdir(FILEFOLDER) if img.endswith(".tif") or img.endswith(
        ".tiff")]  # if this came out of moco the file suffix is .tif otherwise it's tiff
    sort_nicely(images)
    return images
def pic2vid(image_folder, images, video_name):
    """
    this takes an image folder and a list of image files and makes a movie
    :param image_folder: string
    :param images: list of image filenames (strings)
    :return:
    """
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 60, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()
    return 0

def main(umbrella_folder = UMBRELLA_FOLDER, date = DATE, participant = PARTICIPANT):
    for folder in os.listdir(umbrella_folder):
        path = os.path.join(umbrella_folder, folder)
        print(path)
        images = get_images(path)
        new_folder_name = date + "_" + participant + "_" + folder + "_chopped_10"
        path_new = os.path.join(umbrella_folder, new_folder_name)
        os.mkdir(path_new)
        # Make a list of image files
        # image_files = []
        for i in range(len(images)):
            picture = np.array(cv2.imread(os.path.join(path, images[i])))
            # # This chops the image into smaller pieces (important if there has been motion correction)
            new_new_picture = picture[10:]
            # image_files.append(new_new_picture)
            cv2.imwrite(os.path.join(path_new, images[i]), new_new_picture)
        print("finished folder " + folder)
    return 0

"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))