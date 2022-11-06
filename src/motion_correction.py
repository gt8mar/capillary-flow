from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging

from src.tools import get_images
from src.tools import get_image_paths

SET = "set_01"
SAMPLE = "sample_000"

try:
    cv2.setNumThreads(0)
except:
    pass

# try:
#     if __IPYTHON__:
#         get_ipython().magic('load_ext autoreload')
#         get_ipython().magic('autoreload 2')
# except NameError:
#     pass

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.DEBUG)

import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.utils.utils import download_demo

# # fnames = 'C:\\Users\\gt8mar\\capillary-flow\\data\\processed\\set_01\\sample_000\\A_cropped\\Basler_acA1300-200um__23253950__20220513_155354922_0000.tiff'
# fnames = [download_demo(fnames)]     # the file will be downloaded if it doesn't already exist

set = "set_01"
sample = "sample_000"

input_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(set), str(sample), 'A_cropped\\vid')
fnames = get_image_paths.main(input_folder)
frame = cv2.imread(os.path.join(input_folder, fnames[0]), cv2.IMREAD_GRAYSCALE)
m_orig = cm.load_movie_chain(fnames)
# downsample_ratio = .2  # motion can be perceived better when downsampling in time
# m_orig.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=30, magnification=2)   # play movie (press q to exit)





# max_shifts = (6, 6)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
# strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
# overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
# num_frames_split = 100  # length in frames of each chunk of the movie (to be processed in parallel)
# max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
# pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
# shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
# border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)

# if 'dview' in locals():
#     cm.stop_server(dview=dview)
# c, dview, n_processes = cm.cluster.setup_cluster(
#     backend='local', n_processes=None, single_thread=False)

# mc = MotionCorrect(fnames, dview=dview, max_shifts=max_shifts,
#                   strides=strides, overlaps=overlaps,
#                   max_deviation_rigid=max_deviation_rigid, 
#                   shifts_opencv=shifts_opencv, nonneg_movie=True,
#                   border_nan=border_nan)

# # correct for rigid motion correction and save the file (in memory mapped form)
# mc.motion_correct(save_movie=True)

# m_rig = cm.load(mc.mmap_file)
# bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)


# plt.figure(figsize = (20,10))
# plt.imshow(mc.total_template_rig, cmap = 'gray')
# plt.show()

# m_rig.resize(1, 1, downsample_ratio).play(
#     q_max=99.5, fr=30, magnification=2, bord_px = 0*bord_px_rig) # press q to exit

# plt.close()
# plt.figure(figsize = (20,10))
# plt.plot(mc.shifts_rig)
# plt.legend(['x shifts','y shifts'])
# plt.xlabel('frames')
# plt.ylabel('pixels')
# plt.show()