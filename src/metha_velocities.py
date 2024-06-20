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
from scipy.stats import norm


FPS = 113.9 #227.8 #169.3
PIX_UM = 2.44 #1.74
V_MAX_MMS = 4.5  # mm/sec

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

def main(filename, plot = False):
    """Main function to run the image analysis.

    Args:
        filename (str): Path to the data file to load.
    """
    # Load required data: an image volume, a binary mask, the frame rate, and pixel size
    data = scipy.io.loadmat(filename)

    # Extract data from the loaded .mat file
    windowArray = data['windowArray']  # Assuming the variable names are the same in the MATLAB file
    maskIm = data['maskIm']

    # Set all values in the binary mask to 1
    # maskIm[:] = 1

    # Tunable parameters
    v_max_mms = 4.5  # mm/sec
    p_criterion = 0.025  # Bonferroni adjusted one-tailed criterion

    # Calculate the size of the mask image
    ySize, xSize = maskIm.shape
    print(f"Mask image size: {ySize} x {xSize}")
    if plot:
        plt.imshow(maskIm)
        plt.show()

    # Calculate standard deviation image (in vector orientation)
    print("windowArray shape:", windowArray.shape)
    flattened_stdIm = np.std(windowArray, axis=1)
    print("flattened_stdIm shape:", flattened_stdIm.shape)
    stdIm = np.reshape(flattened_stdIm, (xSize, ySize))
    # Transpose the image to match the mask image
    stdIm = np.transpose(stdIm)
    if plot:
        plt.imshow(stdIm)
        plt.show()

    valid_mask = (maskIm == 1) & ~np.isnan(stdIm)
    loopPix = np.where(valid_mask)
    row_indices, column_indices = loopPix
    flat_indices = row_indices * xSize + column_indices  # 388 is the number of columns in the original matrix


    # Pre-compute arrays 
    fwd_array = windowArray[:, 2:]
    bak_array = windowArray[:, :-2]

    selected_fwd_array = fwd_array[flat_indices,:]
    selected_bak_array = bak_array[flat_indices,:]

    stdIm[maskIm == 0] = np.nan

    # Flatten stdIm to operate on all values at once and compute the inverse
    stdIm_inv = 1.0 / stdIm.flatten()

    # Compute the mean and standard deviation of the non-NaN values for normalization
    mean_stdIm_inv = np.nanmean(stdIm_inv)
    std_stdIm_inv = np.nanstd(stdIm_inv)

    # Normalize the inverted values to Z-scores
    stdIm_inv_z = (stdIm_inv - mean_stdIm_inv) / std_stdIm_inv

    # Assuming numPix_loop and other required variables are already defined
    # Initialize arrays
    numPix_loop = len(loopPix[0]) 
    disp_row_fwd = np.zeros(numPix_loop)
    disp_col_fwd = np.zeros(numPix_loop)
    disp_row_bak = np.zeros(numPix_loop)
    disp_col_bak = np.zeros(numPix_loop)
    fwd_zvals = np.zeros(numPix_loop)
    bak_zvals = np.zeros(numPix_loop)
    min_zvals = np.zeros(numPix_loop)

    # Precompute coordinates of looped pixels
    xmesh, ymesh = np.meshgrid(np.arange(1, xSize+1), np.arange(1, ySize+1), indexing='xy')  # Ensure 'xy' indexing for consistency with MATLAB
    xmeshvec = xmesh[loopPix]
    ymeshvec = ymesh[loopPix]

    print("xmeshvec shape:", xmeshvec.shape)
    print("ymeshvec shape:", ymeshvec.shape)

    # Assuming pixel_diam_mm, fps, and v_max_mms are already defined
    scaleFactor = (1/PIX_UM) * FPS/1000  # converts between displacement and velocity
    rcutoff = V_MAX_MMS / scaleFactor  # radius corresponding to physiologically plausible values

    # Print to console
    print('Processing...')

    # Initialize a counter for progress tracking
    perc_count = 0

    

    # Assuming all arrays and parameters have been initialized and loaded appropriately
    print("Processing...")
    perc_count = 0

    for pixel_idx in range(numPix_loop):
        perc_count += 1
        if perc_count >= numPix_loop / 100:
            print('.', end='')  # Print progress dots
            perc_count = 0

        # Index conversion assumed already correct for Python zero-based indexing
        pp = loopPix[0][pixel_idx], loopPix[1][pixel_idx]
        rr, cc = pp  # row and column indices

        # Signal replication for pointwise multiplication
        this_sig = windowArray[pp, 1:-1]  # Adjust slicing for Python
        this_rep = np.tile(this_sig, (numPix_loop, 1))

        # Calculate inverse RMS difference
        fwd_im_inv = np.mean((this_rep - fwd_array) ** 2, axis=1) ** -0.5
        bak_im_inv = np.mean((this_rep - bak_array) ** 2, axis=1) ** -0.5

        # Z-score normalization
        fwd_im_inv_z = (fwd_im_inv - np.nanmean(fwd_im_inv)) / np.nanstd(fwd_im_inv)
        bak_im_inv_z = (bak_im_inv - np.nanmean(bak_im_inv)) / np.nanstd(bak_im_inv)

        # Subtraction of standard deviation image
        fwd_dif_inv_z = fwd_im_inv_z - stdIm_inv_z
        bak_dif_inv_z = bak_im_inv_z - stdIm_inv_z

        # Find and store the biggest peak
        fwd_val, fwd_i = np.max(fwd_dif_inv_z), np.argmax(fwd_dif_inv_z)
        bak_val, bak_i = np.max(bak_dif_inv_z), np.argmax(bak_dif_inv_z)

        fwd_zvals[pixel_idx] = fwd_val
        bak_zvals[pixel_idx] = bak_val

        # Convert to row and column displacements
        fwd_r, fwd_c = np.unravel_index(loopPix[fwd_i], (ySize, xSize))
        bak_r, bak_c = np.unravel_index(loopPix[bak_i], (ySize, xSize))

        disp_row_fwd[pixel_idx] = fwd_r - rr
        disp_col_fwd[pixel_idx] = fwd_c - cc
        disp_row_bak[pixel_idx] = bak_r - rr
        disp_col_bak[pixel_idx] = bak_c - cc

        # Distance calculation and statistical correction
        rdist = np.sqrt((xmeshvec[pixel_idx] - xmeshvec) ** 2 + (ymeshvec[pixel_idx] - ymeshvec) ** 2)
        goodPix = rdist <= rcutoff
        this_numPix_p = np.nansum(goodPix)
        this_p_criterion = p_criterion / this_numPix_p
        min_zvals[pixel_idx] = abs(norm.ppf(this_p_criterion))

    print("\nProcessing complete.")





    # Now they are the same shape. 

    # maskIm = data['maskIm']
    # fps = data['fps'][0, 0]
    # pixel_diam_mm = data['pixel_diam_mm'][0, 0]
    # windowArray = data['windowArray']
    # # maskIm[:] = 1  # Set entire mask to 1     totally unclear why this is done

    # # Parameters
    # v_max_mms = 4.5  # mm/sec
    # p_criterion = 0.025  # Bonferroni adjusted one-tailed criterion

    # # Standard deviation and initialization
    # stdIm = calculate_standard_deviation(windowArray, maskIm)
    # # loopPix is the pixels where the mask is true and the standard deviation is not, i.e. the pixels are available for blood to be in.
    # loopPix = np.where(maskIm & ~np.isnan(stdIm))
    # print("loopPix:", loopPix)

    # # Initialize arrays
    # numPix_loop = len(loopPix[0]) # number of pixels to loop over
    # (dr_fwd, dc_fwd, dr_bak, dc_bak, fwd_zvals, bak_zvals, min_zvals,
    #  fwd_array, bak_array, stdIm_inv_z) = initialize_arrays(numPix_loop)

    # print("windowArray shape:", windowArray.shape)
    # print("Sliced windowArray shape:", windowArray[:, 2:].shape)
    # print("loopPix:", loopPix)

    # # reshape loopPix to a list of tuples
    # # loopPix_list = list(zip(loopPix[0], loopPix[1]))
    # # print("loopPix:", len(loopPix_list))

    # Pre-compute arrays 
    fwd_array = windowArray[:, 2:]
    bak_array = windowArray[:, :-2]

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
