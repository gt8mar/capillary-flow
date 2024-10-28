import os
import cupy as cp
import numpy as np
import scipy.io
import scipy.stats
from matplotlib import pyplot as plt
from scipy.stats import norm
import cv2
from src.tools import get_images, load_image_array

V_MAX_MMS = 4.5  # mm/sec

def load_marcus_data():
    """
    Load data from human capillaries. Now returns all data on GPU.

    Returns:
        dict: Dictionary containing loaded data (mask, pixel diameter, video array, fps).
    """
    mask_path = '/hpc/projects/capillary-flow/frog/240729/Frog4/Left/masks/SD_24-07-29_CalFrog4fps100Lankle_mask.png'
    maskIm = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Move mask to GPU immediately after loading
    maskIm = cp.array(maskIm > 0, dtype=cp.int32)
    
    pixel_diam_um = 0.8
    pixel_diam_mm = pixel_diam_um / 1000
    fps = 100
    
    images = get_images.get_images('/hpc/projects/capillary-flow/frog/240729/Frog4/Left/vids/24-07-29_CalFrog4fps100Lankle')
    video_array = load_image_array.load_image_array(images, '/hpc/projects/capillary-flow/frog/240729/Frog4/Left/vids/24-07-29_CalFrog4fps100Lankle')
    # Move video array to GPU immediately after loading
    video_array = cp.array(video_array)
    
    data = {
        'maskIm': maskIm,
        'pixel_diam_mm': pixel_diam_mm,
        'fps': fps,
        'video_array': video_array
    }
    print("Data loaded successfully on GPU")
    return data

def process_pixel(i, loopPix, maskIm, video_array, selected_fwd_array, selected_bak_array, 
                 loopPix_coords, V_MAX_MMS, pixel_diam_mm, fps, 
                 disp_row_fwd, disp_col_fwd, disp_row_bak, disp_col_bak):
    """
    Process a single pixel for velocity calculation. Modified for parallel processing.
    
    Args:
        i (int): Local index within this GPU's assigned pixels
        loopPix (cupy.ndarray): Array of pixel indices for this GPU
        maskIm (cupy.ndarray): Binary image mask
        video_array (cupy.ndarray): Full video array
        selected_fwd_array (cupy.ndarray): Pre-computed forward arrays for this GPU's pixels
        selected_bak_array (cupy.ndarray): Pre-computed backward arrays for this GPU's pixels
        loopPix_coords (tuple): Coordinates for pixels assigned to this GPU
        V_MAX_MMS (float): Maximum velocity in mm/sec
        pixel_diam_mm (float): Pixel diameter in mm
        fps (float): Frames per second
        disp_row_fwd (cupy.ndarray): Array to store forward row displacements
        disp_col_fwd (cupy.ndarray): Array to store forward column displacements
        disp_row_bak (cupy.ndarray): Array to store backward row displacements
        disp_col_bak (cupy.ndarray): Array to store backward column displacements
    
    Returns:
        tuple: Displacement values and z-scores for the processed pixel
    """
    print(f"Processing pixel {i} of {len(loopPix)}")
    # Get the current pixel index
    pixel_index = loopPix[i]
    row, col = cp.unravel_index(pixel_index, maskIm.shape)
    
    # Get the signal for current pixel
    pixel_signal = video_array[pixel_index, 1:-1]
    
    # Calculate inverse RMS differences
    signal_repeated = cp.tile(pixel_signal, (len(loopPix), 1))
    fwd_im_inv = cp.mean((signal_repeated - selected_fwd_array) ** 2, axis=1) ** -0.5
    bak_im_inv = cp.mean((signal_repeated - selected_bak_array) ** 2, axis=1) ** -0.5
    
    # Convert to Z-scores
    fwd_im_inv_z = (fwd_im_inv - cp.nanmean(fwd_im_inv)) / cp.nanstd(fwd_im_inv)
    bak_im_inv_z = (bak_im_inv - cp.nanmean(bak_im_inv)) / cp.nanstd(bak_im_inv)
    
    # Calculate standard deviation Z-scores
    stdIm = cp.std(video_array, axis=1)
    stdIm_inv = 1.0 / stdIm
    stdIm_inv_z = (stdIm_inv - cp.nanmean(stdIm_inv)) / cp.nanstd(stdIm_inv)
    stdIm_inv_z = stdIm_inv_z[loopPix]
    
    # Calculate difference Z-scores
    fwd_dif_inv_z = fwd_im_inv_z - stdIm_inv_z
    bak_dif_inv_z = bak_im_inv_z - stdIm_inv_z
    
    # Find maximum values and their indices
    fwd_val, fwd_i = cp.max(fwd_dif_inv_z), cp.argmax(fwd_dif_inv_z)
    bak_val, bak_i = cp.max(bak_dif_inv_z), cp.argmax(bak_dif_inv_z)
    
    # Get coordinates for maximum values
    fwd_r, fwd_c = loopPix_coords[0][fwd_i], loopPix_coords[1][fwd_i]
    bak_r, bak_c = loopPix_coords[0][bak_i], loopPix_coords[1][bak_i]
    
    # Calculate displacements
    disp_row_fwd[i] = fwd_r - row
    disp_col_fwd[i] = fwd_c - col
    disp_row_bak[i] = bak_r - row
    disp_col_bak[i] = bak_c - col
    
    # Calculate minimum Z-value for significance
    rdist = cp.sqrt((loopPix_coords[1][i] - loopPix_coords[1]) ** 2 + 
                    (loopPix_coords[0][i] - loopPix_coords[0]) ** 2)
    goodPix = rdist <= (V_MAX_MMS / (pixel_diam_mm * fps))
    this_numPix_p = cp.nansum(goodPix)
    this_p_criterion = 0.025 / this_numPix_p
    min_zval = abs(norm.ppf(this_p_criterion))
    
    return (disp_row_fwd[i], disp_col_fwd[i], 
            disp_row_bak[i], disp_col_bak[i], 
            fwd_val, bak_val, min_zval)

def calculate_standard_deviation(video_array, maskIm):
    """Calculate standard deviation image.
    
    Args:
        video_array (cupy.ndarray): Array containing window data.
        maskIm (cupy.ndarray): Binary image mask.

    Returns:
        cupy.ndarray: Standard deviation image.
    """
    ySize, xSize = maskIm.shape
    return cp.reshape(cp.std(video_array, axis=1), (ySize, xSize))

def initialize_arrays(numPix_loop):
    """Initialize arrays for displacement and Z-values.

    Args:
        numPix_loop (int): Number of pixels to loop over.

    Returns:
        tuple: A tuple containing initialized cupy arrays for forward and backward calculations.
    """
    return (cp.zeros(numPix_loop) for _ in range(10))

def compute_inverse_rms_differences(this_sig, fwd_array, bak_array, numPix_loop):
    """Compute inverse RMS differences and convert to Z-scores.

    Args:
        this_sig (cupy.ndarray): Signal array for the current pixel.
        fwd_array (cupy.ndarray): Forward shifted array.
        bak_array (cupy.ndarray): Backward shifted array.
        numPix_loop (int): Number of pixels to loop over.

    Returns:
        tuple: Forward and backward inverse RMS differences converted to Z-scores.
    """
    this_rep = cp.tile(this_sig, (numPix_loop, 1))
    fwd_im_inv = cp.mean((this_rep - fwd_array) ** 2, axis=1) ** -0.5
    bak_im_inv = cp.mean((this_rep - bak_array) ** 2, axis=1) ** -0.5
    fwd_im_inv_z = (fwd_im_inv - cp.nanmean(fwd_im_inv)) / cp.nanstd(fwd_im_inv)
    bak_im_inv_z = (bak_im_inv - cp.nanmean(bak_im_inv)) / cp.nanstd(bak_im_inv)
    return fwd_im_inv_z, bak_im_inv_z

def save_velocity_map(vMap, output_path, plot=False):
    """
    Save the velocity map as an image and optionally display it.
    
    Args:
        vMap (cupy.ndarray): Velocity map array on GPU
        output_path (str): Path to save the image
        plot (bool): Whether to display the plot
    """
    # Convert CuPy array to NumPy for matplotlib
    vMap_np = cp.asnumpy(vMap)
    
    # Create figure with specific size
    plt.figure(figsize=(10, 8))
    
    # Create the plot
    im = plt.imshow(vMap_np, cmap='jet', interpolation='nearest')
    plt.colorbar(im, label='Velocity (mm/s)')
    plt.title('Velocity Map')
    
    # Save the figure with high DPI
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if plot:
        plt.show()
    else:
        plt.close()

def split_work(total_pixels, num_gpus, gpu_id):
    """
    Split pixels among GPUs.
    
    Args:
        total_pixels (int): Total number of pixels to process
        num_gpus (int): Number of available GPUs
        gpu_id (int): Current GPU ID
        
    Returns:
        tuple: (start_idx, end_idx) for current GPU
    """
    pixels_per_gpu = total_pixels // num_gpus
    start_idx = gpu_id * pixels_per_gpu
    end_idx = start_idx + pixels_per_gpu if gpu_id < num_gpus - 1 else total_pixels
    return start_idx, end_idx

def save_partial_results(output_data, output_path, gpu_id):
    """
    Save partial results from each GPU.
    
    Args:
        output_data (dict): Dictionary containing partial results
        output_path (str): Base path for saving results
        gpu_id (int): Current GPU ID
    """
    partial_path = f"{output_path}_partial_{gpu_id}.npz"
    np.savez(partial_path, **output_data)

def main(filename, plot=False, write=True, marcus=True, gpu_id=0, num_gpus=1):
    """
    Main function to run the image analysis.
    """
    # Set the GPU device
    cp.cuda.Device(gpu_id).use()
    print(f"Processing on GPU {gpu_id} of {num_gpus}")

    if marcus:
        data = load_marcus_data()
        video_array_3D = data['video_array']
        video_array_3D = cp.transpose(video_array_3D, (1, 2, 0))
        video_array = cp.reshape(video_array_3D, (-1, video_array_3D.shape[2]))
        maskIm = data['maskIm']
        fps = data['fps']
        pixel_diam_mm = data['pixel_diam_mm']
    else:
        data = scipy.io.loadmat(filename)
        video_array = cp.array(data['windowArray'])
        maskIm = cp.array(data['maskIm'])
        fps = data['fps'][0][0]
        pixel_diam_mm = data['pixel_diam_mm'][0][0]
        video_array_3D = cp.reshape(video_array, (maskIm.shape[1], maskIm.shape[0], -1))
        video_array_3D = cp.transpose(video_array_3D, (1, 0, 2))

    print("Finding loop pixel coordinates...")
    loopPix_coords = cp.where(maskIm == 1)
    loopPix = cp.ravel_multi_index(loopPix_coords, maskIm.shape)
    
    # Split work among GPUs
    start_idx, end_idx = split_work(len(loopPix), num_gpus, gpu_id)
    loopPix = loopPix[start_idx:end_idx]
    loopPix_coords = (loopPix_coords[0][start_idx:end_idx], 
                     loopPix_coords[1][start_idx:end_idx])

    video_array = cp.reshape(video_array_3D, (maskIm.shape[0] * maskIm.shape[1], -1))

    print(f"GPU {gpu_id}: Processing pixels {start_idx} to {end_idx}")
    
    # Pre-compute arrays for this GPU's portion
    fwd_array = video_array[:, 2:]
    bak_array = video_array[:, :-2]
    selected_fwd_array = fwd_array[loopPix, :]
    selected_bak_array = bak_array[loopPix, :]

    # Initialize arrays for this GPU's portion
    numPix_loop = len(loopPix)
    disp_row_fwd, disp_col_fwd = cp.zeros(numPix_loop), cp.zeros(numPix_loop)
    disp_row_bak, disp_col_bak = cp.zeros(numPix_loop), cp.zeros(numPix_loop)

    # Process pixels assigned to this GPU
    for i in range(numPix_loop):
        if i % 1000 == 0:
            print(f"GPU {gpu_id}: Processing pixel {i + start_idx}/{end_idx}")
        
        # Process pixel (existing process_pixel function logic here)
        pixel_results = process_pixel(
            i, loopPix, maskIm, video_array, selected_fwd_array, selected_bak_array,
            loopPix_coords, V_MAX_MMS, pixel_diam_mm, fps,
            disp_row_fwd, disp_col_fwd, disp_row_bak, disp_col_bak)
        
        disp_row_fwd[i], disp_col_fwd[i], disp_row_bak[i], disp_col_bak[i] = pixel_results[:4]

    # Calculate velocities for this GPU's portion
    v_raw_fwd = cp.sqrt(disp_row_fwd ** 2 + disp_col_fwd ** 2) * (pixel_diam_mm * fps)
    v_raw_bak = cp.sqrt(disp_row_bak ** 2 + disp_col_bak ** 2) * (pixel_diam_mm * fps)
    velocities = cp.nanmean(cp.column_stack((v_raw_fwd, -v_raw_bak)), axis=1)

    # Save partial results
    if write:
        output_data = {
            'velocities': cp.asnumpy(velocities),
            'coordinates': (cp.asnumpy(loopPix_coords[0]), cp.asnumpy(loopPix_coords[1])),
            'shape': maskIm.shape
        }
        save_partial_results(output_data, '/hpc/projects/capillary-flow/frog/results/velocity', gpu_id)

    print(f"GPU {gpu_id}: Processing complete")
    return True

def combine_results(num_gpus, output_path):
    """
    Combine partial results from all GPUs into final velocity map.
    """
    # Initialize final velocity map
    first_data = np.load(f"{output_path}_partial_0.npz")
    shape = tuple(first_data['shape'])
    final_map = np.full(shape, np.nan)

    # Combine results from all GPUs
    for gpu_id in range(num_gpus):
        data = np.load(f"{output_path}_partial_{gpu_id}.npz")
        velocities = data['velocities']
        coords = data['coordinates']
        final_map[coords[0], coords[1]] = velocities
        
        # Clean up partial files
        os.remove(f"{output_path}_partial_{gpu_id}.npz")

    # Save final velocity map
    plt.figure(figsize=(10, 8))
    plt.imshow(final_map, cmap='jet', interpolation='nearest')
    plt.colorbar(label='Velocity (mm/s)')
    plt.title('Velocity Map')
    plt.savefig(f"{output_path}_final.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python script.py <gpu_id> <num_gpus>")
        sys.exit(1)
        
    try:
        gpu_id = int(sys.argv[1])
        num_gpus = int(sys.argv[2])
        success = main('/hpc/projects/capillary-flow/frog/demo_data.mat', 
                      plot=False, marcus=True, gpu_id=gpu_id, 
                      num_gpus=num_gpus, write=True)
        
        # Only GPU 0 combines results after all GPUs are done
        if gpu_id == 0 and success:
            combine_results(num_gpus, '/hpc/projects/capillary-flow/frog/results/velocity')
            
    except Exception as e:
        print(f"Error on GPU {gpu_id}: {str(e)}")
        sys.exit(1)