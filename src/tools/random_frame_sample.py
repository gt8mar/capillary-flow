import time
import os
import numpy as np
import shutil

def main():
    moco_fp = 'G:\\Marcus\\data\\part14\\230428\\vid03\\moco'
    random_nums = np.random.randint(0, len(os.listdir(moco_fp)), size=20)
    counter = 0
    for frame in os.listdir(moco_fp):
        if counter in random_nums:
            shutil.copy(os.path.join(moco_fp, frame), os.path.join('C:\\Users\\Luke\\Documents\\capillary-flow\\random_frames\\part14_230428_vid03', frame))
        counter += 1

if __name__ == "__main__":
    ticks = time.time()
    main()
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))