import time
import os
from src.tools.align_segmented import align_segmented
from src.tools import group_caps
from src.tools.naming_overlay import make_overlays
from src.tools.rename_individual_caps import rename

if __name__ == "__main__":
    ticks = time.time()
    umbrella_folder = 'E:\\Marcus\\data'
    for participant in os.listdir(umbrella_folder):
        if participant in ['part35', 'part36']: 
            for date in os.listdir(os.path.join(umbrella_folder, participant)):
                for location in os.listdir(os.path.join(umbrella_folder, participant, date)):
                    if 'Temp' in location:
                        continue
                    print('Processing: ' + participant + ' ' + date + ' ' + location)
                    path = os.path.join(umbrella_folder, participant, date, location)
                    #align_segmented(path)
                    #group_caps.main(path)
                    #make_overlays(path)
                    rename(path)
                    make_overlays(path, rename=True)
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))