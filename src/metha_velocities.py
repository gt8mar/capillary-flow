"""
Filename: metha_velocities.py
------------------------------------------------------
This file takes a series of images and calculates the standard deviation image before 
processing each pixel to find the maximum forward and backward displacements. This
information is then used to display a velocity map.

By: Marcus Forst (adapted from MATLAB code from Metha Group)
"""

import numpy as np
import scipy.io
import scipy.stats
from matplotlib import pyplot as plt
from scipy.stats import norm
import cv2
from src.tools import get_images, load_image_array


# FPS = 113.9 #227.8 #169.3
# PIX_UM = 0.8 #2.44 #1.74
V_MAX_MMS = 4.5  # mm/sec

def calculate_standard_deviation(video_array, maskIm):
    """Calculate standard deviation image.
    
    Args:
        video_array (numpy.ndarray): Array containing window data.
        maskIm (numpy.ndarray): Binary image mask.

    Returns:
        numpy.ndarray: Standard deviation image.
    """
    ySize, xSize = maskIm.shape
    return np.reshape(np.std(video_array, axis=1), (ySize, xSize))

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

def compare_data_shapes(filepath_marcus, filepath_mat):
    """
    Compares the shapes of arrays in two different data files: Marcus' data format and a .mat file.

    Args:
        filepath_marcus (str): Path to the data file in Marcus' format.
        filepath_mat (str): Path to the .mat data file.

    Returns:
        dict: Dictionary containing the shapes of the arrays in both files for comparison.
    """
    # Load Marcus data
    data_marcus = load_marcus_data()
    video_array_marcus = data_marcus['video_array']
    maskIm_marcus = data_marcus['maskIm']
    
    # Load .mat data
    data_mat = scipy.io.loadmat(filepath_mat)
    video_array_mat = data_mat['windowArray']
    maskIm_mat = data_mat.get('maskIm', None)  # Check if maskIm exists in .mat data

    # Check the presence of maskIm in .mat file, handle if it’s missing
    if maskIm_mat is None:
        print("Warning: 'maskIm' not found in the .mat file. Defaulting to None.")
    
    # Store shapes for comparison
    shapes_comparison = {
        "Marcus Data": {
            "video_array_shape": video_array_marcus.shape,
            "maskIm_shape": maskIm_marcus.shape,
        },
        ".mat Data": {
            "video_array_shape": video_array_mat.shape,
            "maskIm_shape": maskIm_mat.shape if maskIm_mat is not None else "Not available",
        }
    }

    # Print out the shapes for easy comparison
    print("Shape Comparison:")
    for data_type, shapes in shapes_comparison.items():
        print(f"{data_type}:")
        for array_name, shape in shapes.items():
            print(f"  {array_name}: {shape}")

    return shapes_comparison

def load_marcus_data():
    """
    Load data from human capillaries in the same form as the
    Metha MATLAB data. 

    Args:
        filepath (str): Path to the data file.

    Returns:
        dict: Dictionary containing:
            - maskIm (numpy.ndarray): Binary mask image.
            - pixel_diam_mm (float): Pixel diameter in millimeters.
            - fps (int): Frames per second.
            - video_array (numpy.ndarray): Array of video data.
        """
    # What do we want this function to output? To gather? 
    # what information are we trying to get here?
    # mask, pixel_diam_mm, video_array, fps
    # Maybe need header, version, global etc, unclear.
    mask_path = 'D:\\frog\\masks\\SD_24-07-29_CalFrog4fps100Lankle_mask.png'
    maskIm = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pixel_diam_um = 0.8
    pixel_diam_mm = pixel_diam_um / 1000
    # fps = mask_path.replace('.png', '').split('Lankle')[0].split('fps')[-1]
    fps = 100
    images = get_images.get_images('D:\\frog\\vids\\24-07-29_CalFrog4fps100Lankle')
    video_array = load_image_array.load_image_array(images, 'D:\\frog\\vids\\24-07-29_CalFrog4fps100Lankle')
    # mask_template = video_array[0]
    # maskIm = np.ones_like(mask_template)
    data = {'maskIm': maskIm,
            'pixel_diam_mm': pixel_diam_mm,
            'fps': fps,
            'video_array': video_array}
    return data

def main(filename, plot = False, write = False, marcus = True):
    """
    Main function to run the image analysis.

    Args:
        filename (str): Path to the data file to load.
    """
    if marcus:
        data = load_marcus_data()
        video_array_3D = data['video_array']  # Assuming the variable names are the same in the MATLAB file
        # print(video_array_3D.shape)
        # reshape video_array so that it goes from (t, row, col), to (row, col, t)
        video_array_3D = video_array_3D[:100, :, :] # downsize for testing
        # print(video_array_3D.shape)
        downsampled_video_array = np.zeros((video_array_3D.shape[0], video_array_3D.shape[1]//4, video_array_3D.shape[2]//4))
        # downsample the spatial data 2x2 using an average filter
        for i in range(video_array_3D.shape[0]):
            frame = video_array_3D[i]
            frame = np.array(frame, dtype=np.uint8)
            # print(frame.shape)
            downsampled_frame = cv2.resize(frame, (frame.shape[1]//4 ,frame.shape[0]//4), interpolation=cv2.INTER_AREA)
            downsampled_video_array[i] = downsampled_frame

        # video_array_3D = np.array([cv2.resize(frame, (frame.shape[1]//2 ,frame.shape[0]//2), interpolation=cv2.INTER_AREA) for frame in video_array_3D])
        video_array_3D = np.transpose(downsampled_video_array, (1, 2, 0))
        video_array = np.reshape(video_array_3D, (-1, video_array_3D.shape[2]))
        flattened_stdIm = np.std(video_array, axis=1)
        maskIm = data['maskIm']
        # resize mask
        maskIm = cv2.resize(maskIm, (maskIm.shape[1]//4,maskIm.shape[0]//4), interpolation=cv2.INTER_AREA)
        fps = data['fps']
        pixel_diam_mm = data['pixel_diam_mm']
    else:
        # Load required data: an image volume, a binary mask, the frame rate, and pixel size
        data = scipy.io.loadmat(filename)
        data['video_array'] = data['windowArray']
        print(data.keys())

        # Extract data from the loaded .mat file
        video_array = data['video_array']  # Assuming the variable names are the same in the MATLAB file
        maskIm = data['maskIm']
        fps = data['fps']
        pixel_diam_mm = data['pixel_diam_mm']

        # Calculate standard deviation image (in vector orientation)
        print("video_array shape:", video_array.shape)
        flattened_stdIm = np.std(video_array, axis=1)
        ySize, xSize = maskIm.shape

        # reshape video_array to have the same shape as maskIm plus a time dimension
        video_array_3D = np.reshape(video_array, (xSize, ySize, -1))
        # transpose video_array to be in the same orientationas maskIm
        video_array_3D = np.transpose(video_array_3D, (1, 0, 2))
        
    stdIm = video_array_3D.std(axis=2)
    # plt.imshow(stdIm)
    # plt.show()

    # mean_frame = np.mean(video_array_3D, axis=2)
    # plt.figure(figsize=(10,5))
    # plt.subplot(121)
    # plt.imshow(mean_frame, cmap='gray')
    # plt.title('Mean Frame')
    # plt.subplot(122)
    # plt.imshow(mean_frame, cmap='gray')
    # plt.imshow(maskIm, alpha=0.5, cmap='jet')
    # plt.title('Mask Overlay')
    # plt.show()
    

    print("video_array shape:", video_array.shape)
    print("maskIm shape:", maskIm.shape)
    print("Number of nan values in video_array:", np.sum(np.isnan(video_array)))

    # Set all values in the binary mask greater than 0 to 1
    maskIm[maskIm > 0] = 1

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

    
 
    # Mask the standard deviation image
    stdIm_masked = stdIm*maskIm
    # plt.imshow(stdIm_masked)
    # plt.show()

    # set all values in stdIm_masked that are not 1 to nan
    stdIm_masked[maskIm == 0] = np.nan

    # make 3D array of maskIm to have the same shape as video_array
    maskIm_3D = np.repeat(maskIm[:, :, np.newaxis], video_array_3D.shape[2], axis=2)
    print("maskIm shape:", maskIm.shape)
    print("video_array shape:", video_array_3D.shape)

    masked_video_array = video_array_3D * maskIm_3D

    loopPix_coords = np.where(maskIm == 1)
    print("Unique values in maskIm:", np.unique(maskIm))

    


    # valid_mask = ((maskIm == 1) & (~np.isnan(stdIm)))
   
    # loopPix_coords = np.where(valid_mask)

    # Convert the coordinates to linear indices
    loopPix = np.ravel_multi_index(loopPix_coords, maskIm.shape)
    video_array = np.reshape(video_array_3D, (xSize*ySize, -1))
    print("loopPix shape:", loopPix.shape)
    print("flattened_video_array shape:", video_array.shape)

    # check to see if there are nan values in video_array where loopPix is
    print(f"Number of nan values in video_array where loopPix is: {np.sum(np.isnan(video_array[loopPix]))}")
    
    # slice flattened_stdIm to get the values at loopPix
    stdIm_slice = flattened_stdIm[loopPix]
    print("stdIm shape:", stdIm_slice.shape)
    print("Number of nan values in stdIm_slice:", np.sum(np.isnan(stdIm_slice)))
    
    # Pre-compute arrays 
    fwd_array = video_array[:, 2:]
    bak_array = video_array[:, :-2]

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

    """ --------- loop replacement --------- """

    # Replace the previous vectorized code with this chunked version
    print("Processing in chunks...")
    chunk_size = 2000  # Adjust this based on your available memory

    # Initialize arrays for results
    fwd_zvals = np.zeros(len(loopPix))
    bak_zvals = np.zeros(len(loopPix))
    disp_row_fwd = np.zeros(len(loopPix))
    disp_col_fwd = np.zeros(len(loopPix))
    disp_row_bak = np.zeros(len(loopPix))
    disp_col_bak = np.zeros(len(loopPix))
    min_zvals = np.zeros(len(loopPix))

    # Add before the chunk processing:
    max_search_radius = 5  # pixels
    print(f"\nInitializing with max search radius of {max_search_radius} pixels")

    # Process in chunks
    for chunk_start in range(0, len(loopPix), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(loopPix))
        print(f"\nProcessing chunk {chunk_start//chunk_size + 1} of {(len(loopPix)-1)//chunk_size + 1}")
        
        # Get chunk pixel positions
        chunk_rows, chunk_cols = np.unravel_index(loopPix[chunk_start:chunk_end], (ySize, xSize))
        
        # Create distance masks for all pixels in chunk at once
        row_dists = np.abs(chunk_rows[:, np.newaxis] - ymeshvec[np.newaxis, :])
        col_dists = np.abs(chunk_cols[:, np.newaxis] - xmeshvec[np.newaxis, :])
        valid_correlations = (row_dists <= max_search_radius) & (col_dists <= max_search_radius)
        
        print(f"Valid correlation statistics for first pixel in chunk:")
        print(f"Number of pixels within search radius: {np.sum(valid_correlations[0])}")
        print(f"Distance range to valid pixels: {np.min(row_dists[0][valid_correlations[0]]):.1f} to {np.max(row_dists[0][valid_correlations[0]]):.1f} rows")
        print(f"                                {np.min(col_dists[0][valid_correlations[0]]):.1f} to {np.max(col_dists[0][valid_correlations[0]]):.1f} cols")
        
        # Get chunk signals
        chunk_signals = video_array[loopPix[chunk_start:chunk_end], 1:-1]
        print(f"Chunk signals shape: {chunk_signals.shape}")
        
        # Calculate RMS differences only for valid pixels
        chunk_fwd_inv = np.full((chunk_end-chunk_start, len(loopPix)), np.nan)
        chunk_bak_inv = np.full((chunk_end-chunk_start, len(loopPix)), np.nan)
        
        # Create mask for broadcasting
        valid_mask = valid_correlations[:, :, np.newaxis]
        print(f"Valid mask shape: {valid_mask.shape}")
        
        # Calculate correlations only for valid pixels
        chunk_signals_expanded = chunk_signals[:, np.newaxis, :].repeat(len(loopPix), axis=1)
        chunk_signals_masked = np.where(valid_mask, chunk_signals_expanded, np.nan)
        print(f"Masked signals shape: {chunk_signals_masked.shape}")
        
        # Calculate RMS differences
        chunk_fwd_inv = np.mean((chunk_signals_masked - selected_fwd_array[np.newaxis, :, :]) ** 2, axis=2) ** -0.5
        chunk_bak_inv = np.mean((chunk_signals_masked - selected_bak_array[np.newaxis, :, :]) ** 2, axis=2) ** -0.5
        
        print("\nCorrelation statistics:")
        print(f"Forward correlations range: {np.nanmin(chunk_fwd_inv):.3f} to {np.nanmax(chunk_fwd_inv):.3f}")
        print(f"Backward correlations range: {np.nanmin(chunk_bak_inv):.3f} to {np.nanmax(chunk_bak_inv):.3f}")
        print(f"Number of valid correlations: {np.sum(~np.isnan(chunk_fwd_inv))}")
        
        # Z-score normalization
        chunk_fwd_inv_z = (chunk_fwd_inv - np.nanmean(chunk_fwd_inv, axis=1)[:, np.newaxis]) / np.nanstd(chunk_fwd_inv, axis=1)[:, np.newaxis]
        chunk_bak_inv_z = (chunk_bak_inv - np.nanmean(chunk_bak_inv, axis=1)[:, np.newaxis]) / np.nanstd(chunk_bak_inv, axis=1)[:, np.newaxis]
        
        # Subtract standard deviation image
        chunk_fwd_dif_z = chunk_fwd_inv_z - stdIm_inv_z[np.newaxis, :]
        chunk_bak_dif_z = chunk_bak_inv_z - stdIm_inv_z[np.newaxis, :]
        
        print("\nZ-score statistics:")
        print(f"Forward Z-scores range: {np.nanmin(chunk_fwd_dif_z):.3f} to {np.nanmax(chunk_fwd_dif_z):.3f}")
        print(f"Backward Z-scores range: {np.nanmin(chunk_bak_dif_z):.3f} to {np.nanmax(chunk_bak_dif_z):.3f}")
        
        # Find peaks (only consider valid correlations)
        chunk_fwd_dif_z[~valid_correlations] = -np.inf
        chunk_bak_dif_z[~valid_correlations] = -np.inf
        
        fwd_zvals[chunk_start:chunk_end] = np.nanmax(chunk_fwd_dif_z, axis=1)
        bak_zvals[chunk_start:chunk_end] = np.nanmax(chunk_bak_dif_z, axis=1)
        chunk_fwd_indices = np.argmax(chunk_fwd_dif_z, axis=1)
        chunk_bak_indices = np.argmax(chunk_bak_dif_z, axis=1)
        
        # Calculate displacements
        fwd_rows, fwd_cols = np.unravel_index(loopPix[chunk_fwd_indices], (ySize, xSize))
        bak_rows, bak_cols = np.unravel_index(loopPix[chunk_bak_indices], (ySize, xSize))
        
        disp_row_fwd[chunk_start:chunk_end] = fwd_rows - chunk_rows
        disp_col_fwd[chunk_start:chunk_end] = fwd_cols - chunk_cols
        disp_row_bak[chunk_start:chunk_end] = bak_rows - chunk_rows
        disp_col_bak[chunk_start:chunk_end] = bak_cols - chunk_cols
        
        print("\nDisplacement statistics for this chunk:")
        print(f"Forward displacements: ({np.min(disp_row_fwd[chunk_start:chunk_end]):.1f}, {np.min(disp_col_fwd[chunk_start:chunk_end]):.1f}) to ({np.max(disp_row_fwd[chunk_start:chunk_end]):.1f}, {np.max(disp_col_fwd[chunk_start:chunk_end]):.1f})")
        print(f"Backward displacements: ({np.min(disp_row_bak[chunk_start:chunk_end]):.1f}, {np.min(disp_col_bak[chunk_start:chunk_end]):.1f}) to ({np.max(disp_row_bak[chunk_start:chunk_end]):.1f}, {np.max(disp_col_bak[chunk_start:chunk_end]):.1f})")
        
        # Calculate statistical correction for chunk
        chunk_rdist = np.sqrt((xmeshvec[chunk_start:chunk_end, np.newaxis] - xmeshvec) ** 2 + 
                            (ymeshvec[chunk_start:chunk_end, np.newaxis] - ymeshvec) ** 2)
        chunk_goodPix = chunk_rdist <= rcutoff
        chunk_numPix_p = np.sum(chunk_goodPix, axis=1)
        min_zvals[chunk_start:chunk_end] = np.abs(norm.ppf(p_criterion / chunk_numPix_p))
        
        print(f"\nStatistical threshold range: {np.min(min_zvals[chunk_start:chunk_end]):.3f} to {np.max(min_zvals[chunk_start:chunk_end]):.3f}")
    print("\nFinal displacement statistics:")
    print(f"Forward displacement range: {np.nanmin(disp_row_fwd):.2f} to {np.nanmax(disp_row_fwd):.2f} (rows)")
    print(f"                           {np.nanmin(disp_col_fwd):.2f} to {np.nanmax(disp_col_fwd):.2f} (cols)")
    print(f"Backward displacement range: {np.nanmin(disp_row_bak):.2f} to {np.nanmax(disp_row_bak):.2f} (rows)")
    print(f"                            {np.nanmin(disp_col_bak):.2f} to {np.nanmax(disp_col_bak):.2f} (cols)")

    """ --------- end loop replacement --------- """
        
    print("\nProcessing complete.")
    
    # write the results to a .csv file
    # np.savetxt('C:\\Users\\gt8ma\\capillary-flow\\displacements.csv', np.column_stack((disp_row_fwd, disp_col_fwd, disp_row_bak, disp_col_bak)), delimiter=',')
        
    # Assuming necessary arrays and scaleFactor, v_max_mms, min_zvals are already defined
    # for testing set scaleFactor = 1
    if marcus:
        scaleFactor = 1
    # Calculate raw forward and backward velocities
    v_raw_fwd = np.sqrt(disp_row_fwd**2 + disp_col_fwd**2) * scaleFactor
    v_raw_bak = np.sqrt(disp_row_bak**2 + disp_col_bak**2) * scaleFactor

    
    # Identify invalid measurements
    invalid_fwd = (v_raw_fwd > v_max_mms) | (fwd_zvals < min_zvals)
    invalid_bak = (v_raw_bak > v_max_mms) | (bak_zvals < min_zvals)
    print(invalid_fwd.shape)
    print(invalid_fwd)

    # After calculating v_raw_fwd and v_raw_bak:
    masked_v_raw_fwd = v_raw_fwd[~np.isnan(v_raw_fwd)]
    masked_v_raw_bak = v_raw_bak[~np.isnan(v_raw_bak)]

    print("\nRaw velocity statistics (masked pixels only):")
    print(f"Forward velocities: mean={np.nanmean(masked_v_raw_fwd):.3f}, max={np.nanmax(masked_v_raw_fwd):.3f}")
    print(f"Backward velocities: mean={np.nanmean(masked_v_raw_bak):.3f}, max={np.nanmax(masked_v_raw_bak):.3f}")
    print(f"Percent invalid forward: {100 * np.sum(invalid_fwd[~np.isnan(v_raw_fwd)])/len(masked_v_raw_fwd):.1f}%")
    print(f"Percent invalid backward: {100 * np.sum(invalid_bak[~np.isnan(v_raw_bak)])/len(masked_v_raw_bak):.1f}%")

    if not marcus:
        invalid_fwd = np.squeeze(invalid_fwd)
        invalid_bak = np.squeeze(invalid_bak)

        # Handle invalid measurements
        disp_row_fwd[invalid_fwd] = np.nan
        disp_col_fwd[invalid_fwd] = np.nan
        disp_row_bak[invalid_bak] = np.nan
        disp_col_bak[invalid_bak] = np.nan

    # Average forward and backward displacements
    dr = np.nanmean(np.column_stack((disp_row_fwd, disp_row_bak)), axis=1)
    dc = np.nanmean(np.column_stack((disp_col_fwd, disp_col_bak)), axis=1)
    print("dr shape:", dr.shape)
    print("dc shape:", dc.shape)

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
    if write:
        plt.savefig('C:\\Users\\ejerison\\capillary-flow\\velocity_map.png')
    if plot:
        plt.show()
    else:
        plt.close()
    return 0

if __name__ == "__main__":
    # main('C:\\Users\\gt8mar\\capillary-flow\\tests\\demo_data.mat', plot=True, marcus=False)
    main('C:\\Users\\ejerison\\capillary-flow\\tests\\demo_data.mat', plot=True, write=False,  marcus=True)
    # Example usage:
    # compare_data_shapes(
    #     'C:\\Users\\ejerison\\capillary-flow\\tests\\demo_data_marcus_format.mat',
    #     'C:\\Users\\ejerison\\capillary-flow\\tests\\demo_data.mat'
    # )
    #TODO: Add the following to the main function
    # load our data
    # load mask 
    # update velocity boundary conditions
