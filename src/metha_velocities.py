"""
Filename: metha_velocities.py
------------------------------------------------------
This file takes a series of images and calculates the standard deviation image before 
processing each pixel to find the maximum forward and backward displacements. This
information is then used to display a velocity map.

By: Marcus Forst
"""


import numpy as np
import scipy.io
import scipy.stats
from matplotlib import pyplot as plt

def load_data(filename):
    """Load MATLAB data file.
    
    Args:
        filename (str): Path to the MATLAB file to load.

    Returns:
        dict: Dictionary containing arrays loaded from the MATLAB file.
    """
    return scipy.io.loadmat(filename)

def calculate_standard_deviation(windowArray, maskIm):
    """Calculate standard deviation image.
    
    Args:
        windowArray (numpy.ndarray): Array containing window data.
        maskIm (numpy.ndarray): Binary image mask.

    Returns:
        numpy.ndarray: Standard deviation image.
    """
    ySize, xSize = maskIm.shape
    return np.reshape(np.std(windowArray, axis=1), (ySize, xSize))

def initialize_arrays(numPix_loop):
    """Initialize arrays for displacement and Z-values.

    Args:
        numPix_loop (int): Number of pixels to loop over.

    Returns:
        tuple: A tuple containing initialized numpy arrays for forward and backward calculations.
    """
    return (np.zeros(numPix_loop) for _ in range(10))

def compute_inverse_rms_differences(this_sig, fwd_array, bak_array, numPix_loop):
    """Compute inverse RMS differences and convert to Z-scores.

    Args:
        this_sig (numpy.ndarray): Signal array for the current pixel.
        fwd_array (numpy.ndarray): Forward shifted array.
        bak_array (numpy.ndarray): Backward shifted array.
        numPix_loop (int): Number of pixels to loop over.

    Returns:
        tuple: Forward and backward inverse RMS differences converted to Z-scores.
    """
    this_rep = np.tile(this_sig, (numPix_loop, 1))
    fwd_im_inv = np.mean((this_rep - fwd_array) ** 2, axis=1) ** -0.5
    bak_im_inv = np.mean((this_rep - bak_array) ** 2, axis=1) ** -0.5
    fwd_im_inv_z = (fwd_im_inv - np.nanmean(fwd_im_inv)) / np.nanstd(fwd_im_inv)
    bak_im_inv_z = (bak_im_inv - np.nanmean(bak_im_inv)) / np.nanstd(bak_im_inv)
    return fwd_im_inv_z, bak_im_inv_z

def update_displacements(fwd_im_inv_z, bak_im_inv_z, stdIm_inv_z, loopPix, xSize):
    """Find and update displacements based on Z-score differences.

    Args:
        fwd_im_inv_z (numpy.ndarray): Forward inverse RMS Z-scores.
        bak_im_inv_z (numpy.ndarray): Backward inverse RMS Z-scores.
        stdIm_inv_z (numpy.ndarray): Inverse of the standard deviation Z-scores.
        loopPix (tuple): Indices of pixels to loop over.
        xSize (int): The size of the x dimension.

    Returns:
        tuple: Values and indices for the maximum forward and backward displacements.
    """
    fwd_dif_inv_z = fwd_im_inv_z - stdIm_inv_z
    bak_dif_inv_z = bak_im_inv_z - stdIm_inv_z
    fwd_val, fwd_i = np.max(fwd_dif_inv_z), np.argmax(fwd_dif_inv_z)
    bak_val, bak_i = np.max(bak_dif_inv_z), np.argmax(bak_dif_inv_z)
    return fwd_val, bak_val, divmod(fwd_i, xSize), divmod(bak_i, xSize)

def main(filename):
    """Main function to run the image analysis.

    Args:
        filename (str): Path to the data file to load.
    """
    data = load_data(filename)
    
    # print(data.keys())
    # print(data['maskIm'].shape)
    # plt.imshow(data['maskIm'])
    # plt.show()

    # print(data['windowArray'].shape)
    # video_array = data['windowArray'].reshape(388, 220, 30)
    # # for i in range(30):
    # #     plt.imshow(video_array[:, :, i])
    # #     plt.show()
    # # transpose the array from 388, 220, 30 to 220, 388, 30
    # video_array = np.transpose(video_array, (1, 0, 2))
    # # for i in range(30):
    # #     plt.imshow(video_array[:, :, i])
    # #     plt.show()
    
    # # now video_array and maskIm have the same shape and orientation. 
    


    maskIm = data['maskIm']
    fps = data['fps'][0, 0]
    pixel_diam_mm = data['pixel_diam_mm'][0, 0]
    windowArray = data['windowArray']
    maskIm[:] = 1  # Set entire mask to 1     totally unclear why this is done

    # Parameters
    v_max_mms = 4.5  # mm/sec
    p_criterion = 0.025  # Bonferroni adjusted one-tailed criterion

    # Standard deviation and initialization
    stdIm = calculate_standard_deviation(windowArray, maskIm)
    loopPix = np.where(maskIm & ~np.isnan(stdIm))
    numPix_loop = len(loopPix[0])
    (dr_fwd, dc_fwd, dr_bak, dc_bak, fwd_zvals, bak_zvals, min_zvals,
     fwd_array, bak_array, stdIm_inv_z) = initialize_arrays(numPix_loop)

    print("windowArray shape:", windowArray.shape)
    print("Sliced windowArray shape:", windowArray[:, 2:].shape)
    print("loopPix:", loopPix)
    # reshape loopPix to a list of tuples
    loopPix = list(zip(loopPix[0], loopPix[1]))
    print("loopPix:", loopPix)

    # # Pre-compute arrays
    # fwd_array = windowArray[:, 2:][loopPix]
    # bak_array = windowArray[:, :-2][loopPix]

    # # Process each pixel
    # for pp_i in range(numPix_loop):
    #     pp = loopPix[0][pp_i], loopPix[1][pp_i]
    #     this_sig = windowArray[pp][:, 1:-1]
    #     fwd_im_inv_z, bak_im_inv_z = compute_inverse_rms_differences(this_sig, fwd_array, bak_array, numPix_loop)
    #     fwd_val, bak_val, (fwd_r, fwd_c), (bak_r, bak_c) = update_displacements(fwd_im_inv_z, bak_im_inv_z, stdIm_inv_z, loopPix, windowArray.shape[1])
    #     # Update Z-values and displacements (further code as required)

    # # Display velocity map
    # plt.figure()
    # plt.imshow(fwd_array)  # Example placeholder
    # plt.colorbar()
    # plt.title('Velocity (mm/s)')
    # plt.show()

if __name__ == "__main__":
    main('C:\\Users\\gt8mar\\capillary-flow\\tests\\demo_data.mat')
