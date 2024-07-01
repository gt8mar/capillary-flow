import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def remove_horizontal_banding(image_path, filter_size=100, rotation=0):
    # Open the grayscale JPEG image and rotate it
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    # Compute the average intensity along each row
    row_means = np.mean(img, axis=1)

    # Create a smoothed version of the row means
    smoothed_means = ndimage.gaussian_filter1d(row_means, filter_size)

    # Subtract the smoothed means from each column of the image
    correction = np.tile(smoothed_means[:, np.newaxis], (1, img.shape[1]))
    corrected_image = img - correction

    # Normalize the image to the original intensity range
    corrected_image -= np.min(corrected_image)
    corrected_image /= np.max(corrected_image)
    corrected_image *= 255

    #  Convert back to PIL Image
    corrected_image = Image.fromarray(corrected_image.astype(np.uint8))

    return corrected_image, img

    # Function to plot line on an axis
def plot_line_on_image(ax, image):
    ax.imshow(image, cmap='gray', origin='upper')
    if avg_angle is not None:
        height, width = image.shape
        center_x, center_y = width // 2, height // 2
        length = max(width, height) // 2
        radian_angle = np.radians(avg_angle)
        plot_angle = np.pi - radian_angle
        x1 = int(center_x - length * np.cos(plot_angle))
        y1 = int(center_y - length * np.sin(plot_angle))
        x2 = int(center_x + length * np.cos(plot_angle))
        y2 = int(center_y + length * np.sin(plot_angle))
        ax.plot([x1, x2], [y1, y2], color='red', linewidth=2)
    ax.axis('off')

def detect_streaks_and_plot(image_path, filter_size=100):
    # Get both the corrected and original images
    corrected_image, original_image = remove_horizontal_banding(image_path, filter_size=filter_size)

    # plot the images next to each other
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title("Original Image")
    ax1.axis('off')
    ax2.imshow(corrected_image, cmap='gray')
    ax2.set_title("Corrected Image")
    ax2.axis('off')
    plt.show()

    # Convert to numpy array if it's not already
    img = np.array(corrected_image)

    # Apply edge detection
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # Apply Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Process the detected lines
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta)
        if 20 < angle < 160:  # Filter out near-horizontal and near-vertical lines
            angles.append(angle)

    # Calculate the average angle
    avg_angle = np.mean(angles) if angles else None
 
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot original image
    plot_line_on_image(ax1, original_image)
    ax1.set_title("Original Image")

    # Plot corrected image
    plot_line_on_image(ax2, img)
    ax2.set_title("Corrected Image")

    # Set the main title
    if avg_angle is not None:
        fig.suptitle(f"Detected Streak Angle: {avg_angle:.2f}°", fontsize=16)
    else:
        fig.suptitle("No significant diagonal streaks detected", fontsize=16)

    plt.tight_layout()
    plt.show()

    return avg_angle, original_image, img


# Use the function
folder_path = 'C:\\Users\\gt8mar\\capillary-flow\\results\\kymographs\\tricky_kymographs'
for filename in os.listdir(folder_path):
    if filename.endswith('.tiff'):
        filename = 'set01_part09_230414_loc01_vid14_kymograph_03a.tiff'
        image_path = os.path.join(folder_path, filename)
    else:
        continue
    

    print(f"Processing image: {image_path}")
    avg_angle, original_img, corrected_img = detect_streaks_and_plot(image_path)

    print(f"Average angle of diagonal streaks: {avg_angle:.2f} degrees")
