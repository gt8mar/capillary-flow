"""
Filename: pic2vid_color.py
-------------------------------------------------------------
This file turns a group of files into a video. It correctly orders misordered files. It makes the video colored with
matplotlib colormaps

by: Marcus Forst
sort_nicely credit: Ned B (https://nedbatchelder.com/blog/200712/human_sorting.html)
"""


import cv2
import os
import glob
import re
import matplotlib.pyplot as plt

FILEFOLDER = 'C:\\Users\\Luke\\Documents\\Marcus\\Data\\220513\\pointer2_slice_stable'
BORDER = 50

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

images = [img for img in os.listdir(FILEFOLDER) if img.endswith(".tif")]
sort_nicely(images)

"""
------------------------------------------------------------------------------------------------------------------ 
"""


video_name = FILEFOLDER
video_name += str(images[0].strip(".tif"))
video_name += "2.avi"
print(video_name)

im = cv2.imread(os.path.join(FILEFOLDER, images[0]))
height, width, layers = im.shape

im_slice = im[BORDER:-BORDER, BORDER:-BORDER]
# plot image in color
plt.imshow(im_slice.mean(2), cmap="viridis")
#save image in color
plt.imsave("color.png", im_slice.mean(2), cmap="viridis")
plt.show()

# write new folder of colored images:
cwd = os.getcwd()
folder = FILEFOLDER + "_color"
path = os.path.join(cwd, folder)
if folder not in os.listdir(cwd):
    os.mkdir(path)



for image in images:
    im2 = cv2.imread(os.path.join(FILEFOLDER, image))
    im2 = im2[BORDER:-BORDER, BORDER:-BORDER]
    plt.imsave(image + "color.png", im2.mean(2), cmap="viridis")


