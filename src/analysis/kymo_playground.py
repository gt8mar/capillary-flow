"""
Filename: kymo_playground.py
------------------------------------------------------
This program looks at the signal of intensity vs time in a kymograph.

By: Marcus Forst
"""

# load in tiff file:

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt
import imageio
import os



# kymograph_path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\kymographs\\set_01_part09_230414_vid37_blood_flow_00.tiff'
# kymograph_path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\kymographs\\set_01_part11_230427_vid16_blood_flow_03.tiff'
kymograph_path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\kymographs\\set_01_part14_230428_vid28_blood_flow_03.tiff'


def wavelet_denoise(signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(signal)))  # Adjust threshold as needed
    coeffs[1:] = (pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:])
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal


def plot_kymo():
    kymograph = cv2.imread(kymograph_path, cv2.IMREAD_GRAYSCALE)
    kymograph_slice = kymograph[::5,:]
    kymograph_slice_total = np.zeros(kymograph_slice.shape)
    for i in range(2):
        print(i)
        kymograph_slice_total += kymograph[i::5,:]
    kymograph_slice_avg = kymograph_slice_total/3
    fig, ax = plt.subplots(2,2)
    ax[0][0].imshow(kymograph_slice)
    ax[1][0].imshow(kymograph_slice_avg)
    freq_1 = kymograph[kymograph.shape[0]//3-20,:]
    freq_2 = kymograph[kymograph.shape[0]//3,:]
    # Denoise Signal A and Signal B using wavelet denoising
    wavelet_level = 0  # Adjust the level based on the signal characteristics
    signal_a_denoised = wavelet_denoise(freq_1, level=wavelet_level)
    signal_b_denoised = wavelet_denoise(freq_2, level=wavelet_level)
    ax[0][1].plot(freq_1)
    ax[1][1].plot(freq_2)
    plt.show()
    # imageio.imwrite('C:\\Users\\gt8mar\\capillary-flow\\results\\kymographs\\set_01_part09_230414_vid37_blood_flow_00_slice.tiff', kymograph_slice)
    return 0

def plot_4_freq():
    kymograph = cv2.imread(kymograph_path, cv2.IMREAD_GRAYSCALE)
    freq_1 = kymograph[0,:]
    freq_2 = kymograph[kymograph.shape[0]//2,:]
    fig, ax = plt.subplots(2,1)
    ax[0].plot(freq_1)
    ax[1].plot(freq_2)
    plt.show()
    return 0


if __name__ == "__main__":
    plot_kymo()
    # plot_2_freq()