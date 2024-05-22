import os
import time
import numpy as np
from PIL import Image

def main(input_dir, output_dir):
    vidlist = os.listdir(input_dir)
    for vid in vidlist:
        vid_dir = os.path.join(input_dir, vid)
        frames = [f for f in os.listdir(vid_dir) if f.endswith('.tiff')]

        output_vid_dir = os.path.join(output_dir, vid)
        os.makedirs(output_vid_dir, exist_ok=True)

        for frame in frames:
            with Image.open(os.path.join(vid_dir, frame)) as img:
                #equalize image histogram
                img = img.convert('L')
                min, max = img.getextrema()
                equalized_img = Image.eval(img, lambda x: (x - min) * 255 / (max - min))
                #save
                equalized_img.save(os.path.join(output_vid_dir, frame))
    
if __name__ == '__main__':
    ticks = time.time()
    main(input_dir = "E:\\frawg\\24-2-13 WkSl", output_dir = "E:\\frawg\\gabbyanalysis\\vids")
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))    