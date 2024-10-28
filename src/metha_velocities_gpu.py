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
        dict: Dictionary containing all data on GPU.
    """
    mask_path = '/hpc/projects/capillary-flow/frog/240729/Frog4/Left/masks/SD_24-07-29_CalFrog4fps100Lankle_mask.png'
    maskIm = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    maskIm[maskIm > 0] = 1
    # Convert to binary mask on GPU where anything greater than 0 is 1
    maskIm = cp.array(maskIm > 0, dtype=cp.uint8)
    
    pixel_diam_um = 0.8
    pixel_diam_mm = pixel_diam_um / 1000
    fps = 100
    
    images = get_images.get_images('/hpc/projects/capillary-flow/frog/240729/Frog4/Left/vids/24-07-29_CalFrog4fps100Lankle')
    video_array = load_image_array.load_image_array(images, '/hpc/projects/capillary-flow/frog/240729/Frog4/Left/vids/24-07-29_CalFrog4fps100Lankle')
    video_array = cp.array(video_array, dtype=cp.float32)
    
    data = {
        'maskIm': maskIm,
        'pixel_diam_mm': pixel_diam_mm,
        'fps': fps,
        'video_array': video_array
    }
    print("Data loaded successfully on GPU")
    return data


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
    
def process_pixel(i, loopPix, maskIm, video_array, selected_fwd_array, selected_bak_array, 
                 loopPix_coords, V_MAX_MMS, pixel_diam_mm, fps, 
                 disp_row_fwd, disp_col_fwd, disp_row_bak, disp_col_bak):
    """
    Process a single pixel for velocity calculation.
    """
    try:
        # Get the current pixel index - convert to numpy scalar for indexing
        pixel_index = int(loopPix[i].get())  # Convert CuPy array element to Python int
        
        # Use divmod for row, col calculation instead of unravel_index
        height = maskIm.shape[0]
        row = pixel_index // maskIm.shape[1]
        col = pixel_index % maskIm.shape[1]
        
        # Convert to CuPy arrays for further calculations
        row = cp.array(row)
        col = cp.array(col)
        
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
        fwd_val = float(cp.max(fwd_dif_inv_z).get())
        fwd_i = int(cp.argmax(fwd_dif_inv_z).get())
        bak_val = float(cp.max(bak_dif_inv_z).get())
        bak_i = int(cp.argmax(bak_dif_inv_z).get())
        
        # Get coordinates for maximum values
        fwd_r = int(loopPix_coords[0][fwd_i].get())
        fwd_c = int(loopPix_coords[1][fwd_i].get())
        bak_r = int(loopPix_coords[0][bak_i].get())
        bak_c = int(loopPix_coords[1][bak_i].get())
        
        # Calculate displacements
        disp_row_fwd[i] = float(fwd_r - row.get())
        disp_col_fwd[i] = float(fwd_c - col.get())
        disp_row_bak[i] = float(bak_r - row.get())
        disp_col_bak[i] = float(bak_c - col.get())
        
        # Calculate minimum Z-value for significance
        rdist = cp.sqrt((loopPix_coords[1][i] - loopPix_coords[1]) ** 2 + 
                      (loopPix_coords[0][i] - loopPix_coords[0]) ** 2)
        goodPix = rdist <= (V_MAX_MMS / (pixel_diam_mm * fps))
        this_numPix_p = float(cp.nansum(goodPix).get())
        this_p_criterion = 0.025 / this_numPix_p
        min_zval = abs(norm.ppf(this_p_criterion))
        
        return (disp_row_fwd[i], disp_col_fwd[i], 
                disp_row_bak[i], disp_col_bak[i], 
                fwd_val, bak_val, min_zval)
    except Exception as e:
            print(f"Error processing pixel {i}: {str(e)}")
            raise


def main(filename, plot=False, write=True, marcus=True, gpu_id=0, num_gpus=1):
    """
    Main function to run the image analysis.
    """
    try:
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
            video_array = cp.array(data['windowArray'], dtype=cp.float32)
            maskIm = cp.array(data['maskIm'] > 0, dtype=cp.uint8)
            fps = float(data['fps'][0][0])
            pixel_diam_mm = float(data['pixel_diam_mm'][0][0])
            video_array_3D = cp.reshape(video_array, (maskIm.shape[1], maskIm.shape[0], -1))
            video_array_3D = cp.transpose(video_array_3D, (1, 0, 2))

        print("Finding loop pixel coordinates...")
        print(f"Mask shape: {maskIm.shape}")
        print(f"Mask dtype: {maskIm.dtype}")
        print(f"Number of masked pixels: {cp.sum(maskIm > 0)}")
        
        # Convert boolean mask to indices
        loopPix_coords = cp.where(maskIm > 0)
        if len(loopPix_coords[0]) == 0:
            raise ValueError("No pixels found in mask!")
            
        print(f"Found {len(loopPix_coords[0])} pixels in mask")
        
        # Calculate linear indices
        loopPix = cp.ravel_multi_index(loopPix_coords, maskIm.shape)
        video_array = cp.reshape(video_array_3D, (maskIm.shape[0] * maskIm.shape[1], -1))

        # Process full array on single GPU
        start_idx = 0
        end_idx = len(loopPix)
        loopPix = loopPix[start_idx:end_idx]
        loopPix_coords = (loopPix_coords[0][start_idx:end_idx], 
                         loopPix_coords[1][start_idx:end_idx])

        print(f"GPU {gpu_id}: Processing pixels {start_idx} to {end_idx}")
        
        # Pre-compute arrays
        fwd_array = video_array[:, 2:]
        bak_array = video_array[:, :-2]
        selected_fwd_array = fwd_array[loopPix, :]
        selected_bak_array = bak_array[loopPix, :]

        # Initialize arrays
        numPix_loop = len(loopPix)
        disp_row_fwd = cp.zeros(numPix_loop, dtype=cp.float32)
        disp_col_fwd = cp.zeros(numPix_loop, dtype=cp.float32)
        disp_row_bak = cp.zeros(numPix_loop, dtype=cp.float32)
        disp_col_bak = cp.zeros(numPix_loop, dtype=cp.float32)

        # Process pixels
        for i in range(numPix_loop):
            if i % 1000 == 0:
                print(f"Processing pixel {i} of {numPix_loop}")
            
            pixel_results = process_pixel(
                i, loopPix, maskIm, video_array, selected_fwd_array, selected_bak_array,
                loopPix_coords, V_MAX_MMS, pixel_diam_mm, fps,
                disp_row_fwd, disp_col_fwd, disp_row_bak, disp_col_bak)
            
            disp_row_fwd[i], disp_col_fwd[i], disp_row_bak[i], disp_col_bak[i] = pixel_results[:4]

        # Calculate velocities
        print("Calculating velocities...")
        v_raw_fwd = cp.sqrt(disp_row_fwd ** 2 + disp_col_fwd ** 2) * (pixel_diam_mm * fps)
        v_raw_bak = cp.sqrt(disp_row_bak ** 2 + disp_col_bak ** 2) * (pixel_diam_mm * fps)
        
        # Create velocity map
        vMap = cp.full(maskIm.shape, cp.nan, dtype=cp.float32)
        velocities = cp.nanmean(cp.column_stack((v_raw_fwd, -v_raw_bak)), axis=1)
        vMap[loopPix_coords] = velocities

        if write:
            print("Saving velocity map...")
            vMap_np = cp.asnumpy(vMap)
            plt.figure(figsize=(10, 8))
            plt.imshow(vMap_np, cmap='jet', interpolation='nearest')
            plt.colorbar(label='Velocity (mm/s)')
            plt.title('Velocity Map')
            plt.savefig('/hpc/projects/capillary-flow/frog/results/velocity_map.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

        print(f"GPU {gpu_id}: Processing complete")
        return True

    except Exception as e:
        print(f"Error on GPU {gpu_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python script.py <gpu_id> <num_gpus>")
        sys.exit(1)
        
    try:
        gpu_id = int(sys.argv[1])
        num_gpus = int(sys.argv[2])
        main('/hpc/projects/capillary-flow/frog/demo_data.mat', 
             plot=False, marcus=True, gpu_id=gpu_id, 
             num_gpus=num_gpus, write=True)
            
    except Exception as e:
        print(f"Error on GPU {gpu_id}: {str(e)}")
        sys.exit(1)