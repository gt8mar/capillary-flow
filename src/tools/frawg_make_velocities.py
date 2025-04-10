"""
Filename: make_velocities.py
-------------------------------------------------
This file uses canny edge detection to call average velocities from
kymographs. 

By: Marcus Forst
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import os, platform
import seaborn as sns
import time
from src.tools.get_images import get_images
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from sklearn.linear_model import Lasso

FPS = 220
PIX_UM = 0.8
CANNY_THRESH_1 = 20
CANNY_THRESH_2 = 50

# TODO: Long term: zero speed capillaries handling
# TODO: clean up results folder saving stuff

def get_condition(filename):
    start_index = filename.find('WkSl')
    end_index = filename.find('Frog')
    start_index += len('WkSl')
    return filename[start_index:end_index].strip()

def average_array(array):
    """
    This returns an averaged array with half length of the input 1d array
    :param array: 1d numpy array length 2n
    :return: 1d array length n
    """
    if np.mod(len(array), 2) == 0:
        return (array[::2] + array[1::2]) // 2
    else:
        return (array[:-1:2] + array[1::2]) // 2
def select_outliers(data, lower_percentile = 10, upper_percentile = 90):
    """
    This function removes outliers by percentile and returns
    both the clipped data and the outlier points. 
    Args:
        data (list of list): List of lists with data to be plotted.
        lower_percentile (int): lower percentile to clip
        upper_percentile (int): upper percentile to clip
    """
    data_clipped = []
    outlier_points = []
    for column in data:
        q1 = np.percentile(column, lower_percentile)
        q3 = np.percentile(column, upper_percentile)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data_clipped.append(column[(column >= lower_bound) & (column <= upper_bound)]) 
        outlier_points.append(column[(column < lower_bound) | (column > upper_bound)])
    return data_clipped, outlier_points
def plot_box_swarm(data, x_labels, y_axis_label,  plot_title, figure_name, 
                   verbose = True, write = False, remove_outliers = False):
    """Plot box-plot and swarm plot for data list.
 
    Args:
        data (list of list): List of lists with data to be plotted.
        y_axis_label (str): Y- axis label.
        x_labels (list of str): List with labels of x-axis.
        plot_title (str): Plot title.
        figure_name (str): Path to output figure.
        verbose (bool): If True, show plot.
        write (bool): If True, write plot to file.
        remove_outliers (bool): If True, remove outliers from data.

    Returns:
        int: 0 if successful.
         
    """
    if remove_outliers:
        # Remove outliers using the clip function
        data_clipped, outliers = select_outliers(data)
        # initialize plot information:
        sns.set(color_codes=True)
        plt.figure(1, figsize=(9, 6))
        plt.title(plot_title)

        # Create box and swarm plot with clipped data
        ax = sns.boxplot(data=data_clipped)
        sns.swarmplot(data=data_clipped, color=".25")
        sns.swarmplot(data = outliers, color = "red")
        if verbose: plt.show()
    else:
        # initialize plot information:
        sns.set(color_codes=True)
        plt.figure(1, figsize=(9, 6))
        plt.title(plot_title)
    
        # plot data on swarmplot and boxplot
        ax = sns.boxplot(data=data)
        sns.swarmplot(data=data, color=".25")
        
        # y-axis label
        ax.set(ylabel=y_axis_label)
    
        # write labels with number of elements
        ax.set_xticks(np.arange(len(data)), labels = x_labels)
        ax.legend()
        
        if write:
            # write figure file with quality 400 dpi
            plt.savefig(figure_name, bbox_inches='tight', dpi=400)
            if verbose: plt.show()
            else: plt.close()
        if verbose: plt.show()
    return 0
def find_slopes(image, filename, output_folder=None, method = 'ridge', verbose = False, write = False, 
                plot_title = "Kymograph", remove_outliers = False):
    edges = cv2.Canny(image, CANNY_THRESH_1, CANNY_THRESH_2)
    # To save edge images:
    # Create a 1x3 grid for subplots
    fig = plt.figure(figsize= (16,4)) #figsize = (5,20)
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=1, rowspan=2)
    ax2 = plt.subplot2grid((2,4), (0, 1), colspan=1, rowspan=2)
    ax3 = plt.subplot2grid((1,4), (0, 2), colspan=2, rowspan=1)
    ax1.imshow(edges)
    ax1.set_title("Canny Edge Detection")
        
    # Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # remove contours with less than 10 points
    contours = [contour for contour in contours if len(contour) > 20]

    # Iterate through the contours
    slopes = []
    lengths = []
    for contour in contours:
        if method == 'ridge':
            # Fit a line to the contour using least squares regression
            [vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Compute the start and end points of the line
            lefty = int((-x*vy/vx) + y)
            righty = int(((image.shape[1]-x)*vy/vx)+y)
            print(f"lefty is {lefty}")
            print(f"righty is {righty}")
            # Draw the line on the original image
            # cv2.line(image, (image.shape[1]-1,righty), (0,lefty), (0,255,0), 2)
        if method == 'lasso':
            # Extract the x and y coordinates of the contour points
            x, y = contour[:, 0, 0], contour[:, 0, 1]
            
            # Fit a Lasso regression model to the contour
            lasso = Lasso(alpha=0.1, tol = 0.0001)
            X = x.reshape(-1, 1) # Reshape the x array into a 2D array
            lasso.fit(X, y)
            
            # Compute the start and end points of the line
            start_x, end_x = 0, image.shape[1]-1
            start_y, end_y = lasso.predict([[start_x]]), lasso.predict([[end_x]])
            slope = (end_y-start_y)/(end_x-start_x)

            # Remove lines with 0 slope
            if np.absolute(slope) <= 0.75:
                # cv2.line(image, (int((end_x-start_x)//4), int((end_y-start_y)//4)), (int(3*(end_x-start_x)//4), int(3*(end_y-start_y)//4)), (0,255,0), 2)
                continue
            """else:
                # Draw the line on the original image
                # cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0,255,0), 2)
                pass"""
            # weight the slope by the length of the contour
            length = len(contour)
            lengths.append(length)

            # Add line to list of lines
            slopes.append(slope[0])

    average_slope = np.mean(np.array(slopes, dtype = float))
    if average_slope > 0:
        slopes, lengths = zip(*[(s, c) for s, c in zip(slopes, lengths) if s > 0])
    if average_slope < 0:
        slopes, lengths = zip(*[(s, c) for s, c in zip(slopes, lengths) if s < 0])

    # Plot the image with the lines
    if remove_outliers:
        slopes_clipped, outliers = select_outliers(slopes)
        if len(slopes_clipped) == 0:
            average_slope = 0
            weighted_average_slope = 0
        else:
            average_slope = np.absolute(np.mean(np.array(slopes_clipped, dtype = float)))
            weighted_average_slope = np.absolute(np.average(np.array(slopes_clipped, dtype = float), weights = np.array(lengths, dtype = float)))
    else:
        if len(slopes) == 0:
            average_slope = 0
            weighted_average_slope = 0
        else:
            weighted_average_slope = np.average(np.array(slopes, dtype = float), weights = np.array(lengths, dtype = float))
            average_slope = np.mean(np.array(slopes, dtype = float))
    
    # Display the original image with lines drawn on it
    if weighted_average_slope == 0:
        cv2.line(image, (int(image.shape[1]/2), 0), (int(image.shape[1]/2), image.shape[0]-1), (255,255,0), 2)
    else:
        cv2.line(image, (int(image.shape[1]/2), 0), (int((image.shape[0]-1)/average_slope) + int(image.shape[1]/2), image.shape[0]-1), (255,255,0), 2)
        cv2.line(image, (int(image.shape[1]/2), 0), (int((image.shape[0]-1)/weighted_average_slope) + int(image.shape[1]/2), image.shape[0]-1), (0,255,0), 2)
    ax2.imshow(image, cmap='gray')
    ax2.set_title("Line Fitting")
    ax3.hist(slopes, bins = 100)
    ax3.set_title("Slope Histogram")
    plot_title = f"Average Slope: {weighted_average_slope:.3f}"
    plt.suptitle(filename + "\n" + plot_title)
    plt.tight_layout()

    if platform.system() != 'Windows':
        results_folder = '/hpc/projects/capillary-flow/results/velocities'
    else:
        results_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results\\velocities'
    
    if write: 
        plt.savefig(os.path.join(output_folder, str(filename) + ".png"), bbox_inches='tight', dpi=400)
        #plt.savefig(os.path.join(results_folder, str(filename) + ".png"), bbox_inches='tight', dpi=400)
        
    if verbose:
        plt.show()  
    else:
        plt.close()
    return weighted_average_slope

def main(path, fps_type, fps, verbose = False, write = False, write_data = True,
         test = False):
    """
    This function takes in a path to a folder containing kymographs and outputs
    a csv file with the average velocities for each capillary. It plots the
    velocities vs. pressure for each capillary and all capillaries on the same
    graph.

    Args:
        path (str): path to the location folder containing kymographs Ex: '/hpc/projects/capillary-flow/frog/240729/Frog4/Left'
        fps_type (str): 'individual' or 'constant'
        fps (int or None): frames per second of the video or flag to use calibration data
        verbose (bool): If True, show plots
        write (bool): If True, write plots to file
        test (bool): If True, use test data

    Returns:
        0 if successful
    """
    # Set up paths
    input_folder = os.path.join(path, 'kymographs')
    os.makedirs(os.path.join(path, 'velocities'), exist_ok=True)
    output_folder = os.path.join(path, 'velocities')

    if not os.path.exists(input_folder):
        print(f"Kymographs folder does not exist for {path}")
        return 1

    # Read in the kymographs
    images = get_images(input_folder, "tiff")

    # Create a dataframe to store the results
    df = pd.DataFrame(columns = ['Filename', 'Date', 'Frog', 'Side', 'Condition', 'Capillary', 'Velocity (um/s)'])

    if platform.system() != 'Windows':
        date = path.split("/")[-3]
        frog = path.split("/")[-2]
        side = path.split("/")[-1]
    else:
        date = path.split("\\")[-3]
        frog = path.split("\\")[-2]
        side = path.split("\\")[-1]
    missing_log = []
    for image in images:
        #if fps is not a number, print letter 
        if fps_type == 'individual':
            # see if 'fps' or 'fp' is in image.
            if 'fps' in image:
                fps = image.split("fps")[1].split('Rankle')[0] #hardcode to get the fps between Frog4fps and Lankle
                condition = image.split("fps")[1].split('_kymograph')[0] #hardcode to get the condition between Frog2fps and _kymograph
            elif 'fp' in image:
                fps = image.split("fp")[1].split('Rankle')[0] #hardcode to get the fps between Frog4fps and Lankle
                condition = image.split("fp")[1].split('_kymograph')[0] #hardcode to get the condition between Frog2fp and _kymograph
            else:
                print(f"FPS not found for {image}")
                continue
            fps = int(fps)
            filename = image.replace('kymograph', 'velocities').replace('.tiff', '').replace('_0','_')    
        else:
            fps = fps
            condition = get_condition(image) 
            video = image.split('_')[0]
            # filename = f'{date}_{video}_{capillary_name}'
            filename = image.replace('kymograph', 'velocities').replace('.tiff', '')    

   
      
        kymo_raw = cv2.imread(os.path.join(input_folder, image), cv2.IMREAD_GRAYSCALE)

        #fps = FPS
        
        # Get the capillary name for the video
        capillary_name = image.replace('.tiff', '').split('_')[-1].replace('0', '')
        kymo_blur = gaussian_filter(kymo_raw, sigma = 2)
        
        if write:
            weighted_average_slope = find_slopes(kymo_blur, filename, output_folder, method = 'lasso', verbose = False, write=True)
        else:
            weighted_average_slope = find_slopes(kymo_blur, filename, output_folder, method = 'lasso', verbose = verbose, write=False)
        # transform slope from pixels/frames into um/s:
        um_slope = np.absolute(weighted_average_slope) *fps/PIX_UM
        # add row to dataframe
        new_data = pd.DataFrame([[filename,date, frog, side, condition, capillary_name, um_slope]], columns = df.columns)
        df = pd.concat([df, new_data], ignore_index=True)

    # Write the dataframe to a file
    if write_data:
        df.to_csv(os.path.join(output_folder, filename + '.csv'), index=False)    
        #df.to_csv(os.path.join(results_folder, f"{file_prefix}_velocity_data.csv"), index=False)    

    # print(df)
    
    """
    --------------------------------- Plot the data---------------------------------------------------
    """
    """# Group the data by 'Capillary'
    grouped_df = df.groupby('Capillary')
    # Get the unique capillary names
    capillaries = df['Capillary'].unique()

    # Create subplots
    num_plots = len(capillaries)
    num_rows = (num_plots + 3) // 4  # Round up to the nearest integer

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(10, 2 * num_rows), sharey=True, sharex=True)

    # Flatten the 2x2 subplot array to make it easier to iterate over
    axes = axes.flatten()

    # Plot each capillary's data in separate subplots
    for i, capillary in enumerate(capillaries):
        capillary_data = grouped_df.get_group(capillary)
        ax = axes[i]
        ax.plot(capillary_data['Pressure'], capillary_data['Weighted Average Slope'], marker='o', linestyle='-')
        # Label all points which decrease in pressure with a red dot
        ax.plot(capillary_data.loc[capillary_data['Pressure'].diff() < 0, 'Pressure'],
                capillary_data.loc[capillary_data['Pressure'].diff() < 0, 'Weighted Average Slope'],
                marker='o', linestyle='-', color='red')
        ax.set_xlabel('Pressure (psi)')
        ax.set_ylabel('Velocity (um/s)')
        ax.set_title(f'Capillary {capillary}')
        ax.grid(True)

    # If there are unused subplots, remove them
    for i in range(num_plots, num_rows * 2):
        fig.delaxes(axes[i])

    # Adjust spacing between subplots
    plt.tight_layout()

    if write:
        plt.savefig(os.path.join(output_folder, f"{part}_{location}_velocity_vs_pressure_per_cap.png"), bbox_inches='tight', dpi=400)
        if platform != 'Windows':
            plt.savefig(os.path.join(results_folder, f"{part}_{location}_velocity_vs_pressure_per_cap.png"), bbox_inches='tight', dpi=400)
    if verbose:
        plt.show()
    else:
        plt.close()
    
    
    #--------------------------------- Plot the data on the same graph ---------------------------------------------------
    
    
    fig, ax = plt.subplots()
    for name, group in grouped_df:
        ax.plot(group['Pressure'], group['Weighted Average Slope'], marker='o', linestyle='', ms=12, label=name)
    
    ax.set_xlabel('Pressure (psi)')
    ax.set_ylabel('Velocity (um/s)')
    ax.set_title('Velocity vs. Pressure for each Capillary')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()

    if write:
        plt.savefig(os.path.join(output_folder, f"{part}_{location}_velocity_vs_pressure.png"), bbox_inches='tight', dpi=400)
        if platform != 'Windows':
            plt.savefig(os.path.join(results_folder, f"{part}_{location}_velocity_vs_pressure.png"), bbox_inches='tight', dpi=400)

    if verbose:
        plt.show()
    else:
        plt.close()


    # plot_box_swarm(data, ["0.2 psi", "0.4 psi", "0.6 psi", "0.8 psi"], 
    #                "velocity (um/s)", "Participant_4 cap_4", "figure 1")
    # plot_box_swarm(data_slice, ["0.2 psi", "0.4 psi", "0.6 psi", "0.8 psi"],
    #                  "velocity (um/s)", "slice", "figure 2")
    return 0"""



"""
-----------------------------------------------------------------------------
"""
# This provided line is required at the end of a Python file
# to call the main() function.
if __name__ == "__main__":
    ticks = time.time()
    # umbrella_folder = 'J:\\frog\\data'
    umbrella_folder = '/hpc/projects/capillary-flow/frog/'
    for date in os.listdir(umbrella_folder):
        if not date.startswith('240729'):
            continue
        if date == 'archive': 
            continue
        if date.endswith('alb'):
            continue
        for frog in os.listdir(os.path.join(umbrella_folder, date)):
            if frog.startswith('STD'):
                continue
            if not frog.startswith('Frog2'):
                continue   
            for side in os.listdir(os.path.join(umbrella_folder, date, frog)):
                if side.startswith('STD'):
                    continue
                # if not side.startswith('Left'): # only process the left side for now
                #     continue
                if side == 'archive':
                    continue
                print('Processing: ' + date + ' ' + frog + ' ' + side)
                path = os.path.join(umbrella_folder, date, frog, side)

                # if date in ['240213', '240214', '240229']:
                #     fps = 100
                # elif date == '240530':
                #     fps = 220
                # else:
                #     fps = 130
                main(path, fps_type='individual', fps = None, write = True, write_data=True, verbose= False, test = False)
    #main(path = 'E:\\frawg\\gabbyanalysis', write = True, write_data=True, verbose= False, test = False)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))
