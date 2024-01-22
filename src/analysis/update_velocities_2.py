import os, platform
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from src.analysis.make_velocities import find_slopes
from src.analysis.update_velocities import plot_velocities

def update_velocities2(csv_path):
    if platform.system() == 'Windows':
        if 'gt8mar' in os.getcwd():
            results_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results'
        else:
            results_folder = 'C:\\Users\\gt8ma\\capillary-flow\\results'
    else:
        results_folder = '/hpc/projects/capillary-flow/results'
# Import copy velocity file
# Read the csv file
    df = pd.read_csv(csv_path)

    Maxes = []
    # Iterate through each row
    for i in range(len(df)):
        # If the velocity is 0
        if df['Correct'][i] == 't':
            if df['Max'][i] == 't':
                Maxes.append([i, df['Velocity'][i]])
    print(Maxes)

    # max is the average of the maxes:
    max = np.mean([Max[1] for Max in Maxes])
    print(f'max = {max}')

    # make new column for corrected velocities
    df['Corrected Velocity'] = df['Velocity']

    for i in range(len(df)):
        # Check each velocity row to see if it is true or false
        # if true, continue
        # if false, check if zero or max
        if (df['Correct'][i] == 'f' and df['Zero'][i]=='t'):
            # If zero, set to zero
            df['Corrected Velocity'][i] = 0
        elif(df['Correct'][i] == 'f' and df['Notes'][i]=='too slow'):
            ## elif too slow, load in kymograph
            filename = f"set01_{df['Participant'][i]}_{df['Date'][i]}_{df['Location'][i]}_{df['Video'][i]}_kymograph_{df['Capillary'][i]}.tiff"
            filepath = os.path.join(results_folder, 'kymographs', filename)
            kymograph = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            kymo_blur = gaussian_filter(kymograph, sigma = 2)
            # print(filename)
            slopes = find_slopes(kymo_blur, filename, verbose=False, too_slow = True, write = False)
        elif(df['Correct'][i] == 'f' and df['Max'][i]=='t'):
            ## elif max, set to max (set to 750? 1000? avg of maxes?), if greater than 1000 set to 1k?
            df['Corrected Velocity'][i] = max


            
            # if df['Max'][i] == 't':
            # if df['Zero'][i] == 't':
            #     df['Corrected Velocity'][i] = 0

    








# elif false and "too slow" run find_slopes(too_slow = true)

# elif false and "too fast" run find_slopes(too_fast = true)
            
# plot:
    print(df.head())

    # plot the corrected velocities organized by capillary:
    # plot_velocities(df, write = False, verbose = True)

    return 0

if __name__ == '__main__':
    # Usage example
    if platform.system() == 'Windows':
        if 'gt8mar' in os.getcwd():
            velocities_folder = 'C:\\Users\\gt8mar\\capillary-flow\\results\\velocities\\velocities'
        else:
            velocities_folder = 'C:\\Users\\gt8ma\\capillary-flow\\results\\velocities\\velocities'
        for csv_file in os.listdir(velocities_folder):
            if csv_file.endswith('Copy.csv'):
                csv_file_path = os.path.join(velocities_folder, csv_file)
                update_velocities2(csv_file_path)
    else:
        csv_file_path = '/hpc/projects/capillary-flow/results/velocities'
        update_velocities2(csv_file_path)

