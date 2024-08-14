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


# FPS = 113.9 #227.8 #169.3
# PIX_UM = 2.44 #1.74
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

def main(filename, plot = False, marcus = True):
    """Main function to run the image analysis.

    Args:
        filename (str): Path to the data file to load.
    """
    # Load required data: an image volume, a binary mask, the frame rate, and pixel size
    data = scipy.io.loadmat(filename)
    print(data.keys())

    # Extract data from the loaded .mat file
    windowArray = data['windowArray']  # Assuming the variable names are the same in the MATLAB file
    maskIm = data['maskIm']
    fps = data['fps']
    pixel_diam_mm = data['pixel_diam_mm']

    print("windowArray shape:", windowArray.shape)
    print("maskIm shape:", maskIm.shape)
    print("Number of nan values in windowArray:", np.sum(np.isnan(windowArray)))

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
    flattened_maskIm = maskIm.flatten()

    # Calculate standard deviation image (in vector orientation)
    print("windowArray shape:", windowArray.shape)
    flattened_stdIm = np.std(windowArray, axis=1)

    # reshape windowArray to have the same shape as maskIm plus a time dimension
    windowArray_3D = np.reshape(windowArray, (xSize, ySize, -1))
    # transpose windowArray to be in the same orientationas maskIm
    windowArray_3D = np.transpose(windowArray_3D, (1, 0, 2))
    
    stdIm = windowArray_3D.std(axis=2)
    # plt.imshow(stdIm)
    # plt.show()
 
    # Mask the standard deviation image
    stdIm_masked = stdIm*maskIm
    # plt.imshow(stdIm_masked)
    # plt.show()

    # set all values in stdIm_masked that are not 1 to nan
    stdIm_masked[maskIm == 0] = np.nan

    # make 3D array of maskIm to have the same shape as windowArray
    maskIm_3D = np.repeat(maskIm[:, :, np.newaxis], windowArray_3D.shape[2], axis=2)
    print("maskIm shape:", maskIm.shape)
    print("windowArray shape:", windowArray_3D.shape)

    masked_windowArray = windowArray_3D * maskIm_3D

    loopPix_coords = np.where(maskIm == 1)
    


    # valid_mask = ((maskIm == 1) & (~np.isnan(stdIm)))
   
    # loopPix_coords = np.where(valid_mask)

    # Convert the coordinates to linear indices
    loopPix = np.ravel_multi_index(loopPix_coords, maskIm.shape)
    windowArray = np.reshape(windowArray_3D, (xSize*ySize, -1))
    print("loopPix shape:", loopPix.shape)
    print("flattened_windowArray shape:", windowArray.shape)

    # check to see if there are nan values in windowArray where loopPix is
    print(f"Number of nan values in windowArray where loopPix is: {np.sum(np.isnan(windowArray[loopPix]))}")
    
    # slice flattened_stdIm to get the values at loopPix
    stdIm_slice = flattened_stdIm[loopPix]
    print("stdIm shape:", stdIm_slice.shape)
    print("Number of nan values in stdIm_slice:", np.sum(np.isnan(stdIm_slice)))
    
    # Pre-compute arrays 
    fwd_array = windowArray[:, 2:]
    bak_array = windowArray[:, :-2]

    selected_fwd_array = fwd_array[loopPix,:]
    selected_bak_array = bak_array[loopPix,:]
    print("Selected forward array shape:", selected_fwd_array.shape)
    print("Selected backward array shape:", selected_bak_array.shape)

    # print number of nan values in selected_fwd_array and selected_bak_array
    print(f"Number of nan values in selected_fwd_array: {np.sum(np.isnan(selected_fwd_array))}")
    print(f"Number of nan values in selected_bak_array: {np.sum(np.isnan(selected_bak_array))}")

    stdIm[maskIm == 0] = np.nan

    # Flatten stdIm to operate on all values at once and compute the inverse
    stdIm_inv = 1.0 / stdIm.flatten()

    # Compute the mean and standard deviation of the non-NaN values for normalization
    mean_stdIm_inv = np.nanmean(stdIm_inv)
    std_stdIm_inv = np.nanstd(stdIm_inv)

    # Normalize the inverted values to Z-scores
    stdIm_inv_z = (stdIm_inv - mean_stdIm_inv) / std_stdIm_inv
    print("stdIm_inv_z shape:", stdIm_inv_z.shape)
    stdIm_inv_z = stdIm_inv_z[loopPix]
    print("stdIm_inv_z shape:", stdIm_inv_z.shape)

    # Assuming numPix_loop and other required variables are already defined
    # Initialize arrays
    numPix_loop = len(loopPix) 
    disp_row_fwd = np.zeros(numPix_loop)
    disp_col_fwd = np.zeros(numPix_loop)
    disp_row_bak = np.zeros(numPix_loop)
    disp_col_bak = np.zeros(numPix_loop)
    fwd_zvals = np.zeros(numPix_loop)
    bak_zvals = np.zeros(numPix_loop)
    min_zvals = np.zeros(numPix_loop)

    # Precompute coordinates of looped pixels
    xmesh, ymesh = np.meshgrid(np.arange(xSize), np.arange(ySize), indexing='xy')  # Ensure 'xy' indexing for consistency with MATLAB
    xmesh_flat = xmesh.flatten()
    ymesh_flat = ymesh.flatten()
    xmeshvec = loopPix_coords[1]
    ymeshvec = loopPix_coords[0]

    # Assuming pixel_diam_mm, fps, and v_max_mms are already defined
    # scaleFactor = (1/PIX_UM) * FPS/1000  # converts between displacement and velocity
    # rcutoff = V_MAX_MMS / scaleFactor  # radius corresponding to physiologically plausible values
    scaleFactor = pixel_diam_mm * fps
    rcutoff = v_max_mms / scaleFactor
    # Print to console
    print('Processing...')

    for i in range(len(loopPix)):
        # Get the linear index from loopPix (assuming loopPix is a flat array of indices)
        pixel_index = loopPix[i]

        # Convert linear index to 2D index (row, column)
        row, col = divmod(pixel_index, xSize)  # divmod gives the quotient and remainder, useful for index conversion
        
        # Extract the signal from the windowArray
        # Python indexing is zero-based and slice end is exclusive
        pixel_signal = windowArray[pixel_index, 1:-1]  # omitting the first and last elements just like MATLAB
        signal_repeated = np.tile(pixel_signal, (numPix_loop, 1))
        
       
        # Calculate inverse RMS difference
        fwd_im_inv = np.mean((signal_repeated - selected_fwd_array) ** 2, axis=1) ** -0.5
        bak_im_inv = np.mean((signal_repeated - selected_bak_array) ** 2, axis=1) ** -0.5
        
        # plt.plot(fwd_im_inv)
        # plt.plot(bak_im_inv)
        # plt.show()

        # print number of values in fwd_im_inv and bak_im_inv that are nan
        if np.sum(np.isnan(fwd_im_inv)) ==10806:
            print(f"Number of nan values in fwd_im_inv: {i} {np.sum(np.isnan(fwd_im_inv))}")
        
        if np.sum(np.isnan(bak_im_inv)) ==3152:
            print(f"Number of nan values in bak_im_inv: {i} {np.sum(np.isnan(bak_im_inv))}")    

        # if fwd_im_inv is empty, print i
        # if np.isnan(fwd_im_inv).all():
            # print("fwd_im_inv is empty at i = ", i)
        # if fwd_im_inv.size == 0:
        #     print("fwd_im_inv is empty at i = ", i)
        # if np.isnan(bak_im_inv).all():
        #     print("bak_im_inv is empty at i = ", i)
        # if bak_im_inv.size == 0:
        #     print("bak_im_inv is empty at i = ", i)

        # Z-score normalization
        fwd_im_inv_z = (fwd_im_inv - np.nanmean(fwd_im_inv)) / np.nanstd(fwd_im_inv)
        bak_im_inv_z = (bak_im_inv - np.nanmean(bak_im_inv)) / np.nanstd(bak_im_inv)

        # Subtraction of standard deviation image
        fwd_dif_inv_z = fwd_im_inv_z - stdIm_inv_z
        bak_dif_inv_z = bak_im_inv_z - stdIm_inv_z

        # Find and store the biggest peak
        fwd_val, fwd_i = np.max(fwd_dif_inv_z), np.argmax(fwd_dif_inv_z)
        bak_val, bak_i = np.max(bak_dif_inv_z), np.argmax(bak_dif_inv_z)

        fwd_zvals[i] = fwd_val
        bak_zvals[i] = bak_val

        # Convert to row and column displacements
        fwd_r, fwd_c = np.unravel_index(loopPix[fwd_i], (ySize, xSize))
        bak_r, bak_c = np.unravel_index(loopPix[bak_i], (ySize, xSize))

        disp_row_fwd[i] = fwd_r - row
        disp_col_fwd[i] = fwd_c - col
        disp_row_bak[i] = bak_r - row
        disp_col_bak[i] = bak_c - col

        # Distance calculation and statistical correction
        rdist = np.sqrt((xmeshvec[i] - xmeshvec) ** 2 + (ymeshvec[i] - ymeshvec) ** 2)
        goodPix = rdist <= rcutoff
        this_numPix_p = np.nansum(goodPix)
        this_p_criterion = p_criterion / this_numPix_p
        min_zvals[i] = abs(norm.ppf(this_p_criterion))
        
    print("\nProcessing complete.")
    
    # write the results to a .csv file
    # np.savetxt('C:\\Users\\gt8ma\\capillary-flow\\displacements.csv', np.column_stack((disp_row_fwd, disp_col_fwd, disp_row_bak, disp_col_bak)), delimiter=',')
        
    # Assuming necessary arrays and scaleFactor, v_max_mms, min_zvals are already defined
    # Calculate raw forward and backward velocities
    v_raw_fwd = np.sqrt(disp_row_fwd**2 + disp_col_fwd**2) * scaleFactor
    v_raw_bak = np.sqrt(disp_row_bak**2 + disp_col_bak**2) * scaleFactor

    # Identify invalid measurements
    invalid_fwd = (v_raw_fwd > v_max_mms) | (fwd_zvals < min_zvals)
    invalid_bak = (v_raw_bak > v_max_mms) | (bak_zvals < min_zvals)
    print(invalid_fwd.shape)
    print(invalid_fwd)

    invalid_fwd = np.squeeze(invalid_fwd)
    invalid_bak = np.squeeze(invalid_bak)

    # Handle invalid measurements
    disp_row_fwd[invalid_fwd] = np.nan
    disp_col_fwd[invalid_fwd] = np.nan
    disp_row_bak[invalid_bak] = np.nan
    disp_col_bak[invalid_bak] = np.nan

    # Average forward and backward displacements
    dr = np.nanmean(np.column_stack((disp_row_fwd, -disp_row_bak)), axis=1)
    dc = np.nanmean(np.column_stack((disp_col_fwd, -disp_col_bak)), axis=1)

    # Compute velocity
    dv = np.sqrt(dr**2 + dc**2) * scaleFactor

    # Convert from vector to 2D image
    vMap = np.full((ySize, xSize), np.nan)  # Create an empty array filled with NaN
    vMap[loopPix_coords] = dv  # Place velocities into the 2D map


    # Display the velocity map
    plt.figure()
    plt.imshow(vMap, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.title('Velocity (mm/s)')
    plt.show()

    # Adjust the colormap to set zero velocity to a specific color if needed
    cust_map = plt.cm.jet
    cust_map.set_under('black')  # Set velocities of zero to black
    plt.imshow(vMap, cmap=cust_map, interpolation='nearest', vmin=0.01)  # Adjust vmin to slightly above 0 to use set_under
    plt.colorbar()



    

if __name__ == "__main__":
<<<<<<< HEAD
    main('C:\\Users\\gt8mar\\capillary-flow\\tests\\demo_data.mat')
=======
    main('C:\\Users\\ejerison\\capillary-flow\\tests\\demo_data.mat')

    #TODO: Add the following to the main function
    # load our data
    # load mask 
    # update velocity boundary conditions
>>>>>>> 2034049368b1ba30a5c0a3ff3f8f71c440cddc76
