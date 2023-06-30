"""
Filename: video_separator.py
-----------------------
By: Gabby Rincon
"""

import os
import time
import shutil

def main(vidnum_start = 0):
    """
    This function takes a directory of images and separates them into video folders
    based on the image number.

    Args:
        vidnum_start (int): the video number to start with
    
    Returns:
        int: 0 if executed
    """
    directory_path = 'C:\\Users\\gt8mar\\Desktop\\data\\part25\\230601'
    # directory_path = 'D:\\Marcus\\backup\\data\\part24\\230601'

    filenames = os.listdir(directory_path)
    filenames = sorted(filenames, key=lambda x: os.path.getctime(os.path.join(directory_path, x))) # TODO: i'm not sure we should use creation time
    # print(filenames)
    files = []
    for i in range(len(filenames)):
        imgnum = filenames[i].split('_')[-1]
        files.append([filenames[i],filenames[i].split('_')[-2],imgnum[:len(imgnum)-5]]) #list where each element is a list of filename, vid number, image number TODO don't hardcode .tiff size. instead remove '.tiff'
    
    vidnum = vidnum_start
    for i in range(len(files)):
        if (int(files[i][2]) == 0):
            vidnum += 1
            dest_path = os.path.join(directory_path, "vid" + str(vidnum).zfill(2))
            os.makedirs(dest_path)
            source_path = os.path.join(directory_path, str(files[i][0]))
            shutil.move(source_path, dest_path)
            
        else:
            source_path = os.path.join(directory_path, str(files[i][0]))
            dest_path = os.path.join(directory_path, "vid" + str(vidnum).zfill(2))
            shutil.move(source_path, dest_path)

    return 0        

if __name__ == "__main__":
    ticks = time.time()
    main(vidnum_start=0)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))