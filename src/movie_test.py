# importing matplot lib
import matplotlib.pyplot as plt
import numpy as np
 
# importing movie py libraries
from moviepy.editor import VideoClip, VideoFileClip, clips_array
from moviepy.video.io.bindings import mplfig_to_npimage
 
path = 'C:\\Users\\Luke\\Documents\\Marcus\\Data\\220513\\pointer2_slice_stable.avi'
# numpy array
x = np.linspace(-2, 2, 200)
 
# duration of the video
duration = 4
 
# matplot subplot
fig, ax = plt.subplots()
 
# method to get frames
def make_frame(t):
     
    # clear
    ax.clear()
     
    # plotting line
    ax.plot(x, np.sinc(x**2) + np.sin(x + 2 * np.pi / duration * t), lw = 3)
    ax.set_ylim(-1.5, 2.5)
     
    # returning numpy image
    return mplfig_to_npimage(fig)
 
# creating animation
animation = VideoClip(make_frame, duration = duration)
capillary000 = VideoFileClip(path)
# capillary000.preview(fps = 60)
f = animation.subclip(0, 2)
s = animation.subclip(0,2)
s2 = capillary000.subclip(0,2)
d = clips_array([[f,s2]])
 
# displaying animation with auto play and looping
# animation.preview(fps = 20)
d.preview(fps = 20)