import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('C:\\Users\\gt8mar\\Desktop\\data\\Image__2024-12-11__19-20-31.tiff', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('C:\\Users\\gt8mar\\Desktop\\data\\calibration\\240522\\Image__2024-05-22__09-27-19.tiff', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('C:\\Users\\gt8mar\\Desktop\\data\\Image__2024-12-11__19-18-24.tiff', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('C:\\Users\\gt8mar\\Desktop\\data\\calibration\\Image__2022-04-26__22-02-43calib.tiff', cv2.IMREAD_GRAYSCALE)



row_avg = np.median(image, axis=1)
col_avg = np.median(image, axis=0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(len(row_avg)), row_avg)
plt.title('Row Average')
plt.ylabel('Pixel Intensity')
plt.xlabel('Row Number')
plt.ylim([0, 255])

plt.subplot(1, 2, 2)
plt.plot(range(len(col_avg)), col_avg)
plt.title('Column Average')
plt.ylabel('Pixel Intensity')
plt.xlabel('Column Number')
plt.ylim([0, 255])

plt.show()
