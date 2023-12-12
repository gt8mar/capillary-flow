import os
import time
import matplotlib as plt
import numpy as np
import pandas as pd

def overlay_velocities(path):
    background_dir = os.path.join(path, "vids")
    centerlines_dir = os.path.join(path, "centerlines")
    velocities_dir = os.path.join(path, "velocities")
    velocities_csv = os.path.join(path, "velocities", [file for file in os.listdir(velocities_dir) if file.endswith(".csv")][0])

    df = pd.read_csv(velocities_csv)
    
    

if __name__ == "__main__":
    ticks = time.time()
    overlay_velocities(path = 'E:\\frawg\\23-9-28')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))