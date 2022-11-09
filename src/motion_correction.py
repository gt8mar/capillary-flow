from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import logging

import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.utils.utils import download_demo

try:
    cv2.setNumThreads(0)
except:
    pass

os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1' 

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

def main():
    SET = "set_01"
    sample = "sample_000"

    # Note: we must use grayscale (2D) tif files. 
    input_folder = os.path.join('C:\\Users\\gt8mar\\capillary-flow\\data\\processed', str(SET), str(sample), 'A_cropped')
    fname = os.path.join(input_folder, f'{SET}_{sample}_stack.tif')

    # m_orig = cm.load_movie_chain(fpaths)
    # # m_orig.play(q_max = 99.5, fr = 60, magnification = 1)
    downsample_ratio = .2  # motion can be perceived better when downsampling in time
    # m_orig.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=30, magnification=2)   # play movie (press q to exit)

    # # Code from a helpful friend
    # cm.load(fpaths[0:28]).save('bar.tif')

    max_shifts = (10, 10)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
    overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
    num_frames_split = 100  # length in frames of each chunk of the movie (to be processed in parallel)
    max_deviation_rigid = 6   # maximum deviation allowed for patch with respect to rigid shifts
    pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
    shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)

    # Start cluster
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    mc = MotionCorrect(fname, dview = dview, max_shifts=max_shifts,
                    strides=strides, overlaps=overlaps,
                    max_deviation_rigid=max_deviation_rigid, 
                    shifts_opencv=shifts_opencv, nonneg_movie=True,
                    border_nan=border_nan, pw_rigid=True, is3D=False)

    # correct for rigid motion correction and save the file (in memory mapped form)
    mc.motion_correct(save_movie=True)
    m_rig = cm.load(mc.mmap_file)
    bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(int)

    plt.figure(figsize = (20,10))
    plt.imshow(mc.total_template_rig)
    plt.show()

    m_rig.resize(1, 1, downsample_ratio).play(
        q_max=99.5, fr=30, magnification=2, bord_px = 0*bord_px_rig) # press q to exit
    # m_rig.play(q_max=99.5, fr=60, magnification=2, bord_px = 0*bord_px_rig) # press q to exit

    plt.figure(figsize = (20,10))
    plt.plot(mc.shifts_rig)
    plt.legend(['x shifts','y shifts'])
    plt.xlabel('frames')
    plt.ylabel('pixels')
    plt.show()
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