import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage.transform import radon
from sklearn.linear_model import Lasso



def remove_horizontal_banding(image_path, filter_size=10, rotation=0):
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
def plot_line_on_image(ax, image, avg_angle):
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

def radon_transform_angle(image):
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)
    angles = np.deg2rad(theta)
    
    # Find the angle with the highest variance
    variance = np.var(sinogram, axis=0)
    radon_angle = angles[np.argmax(variance)]
    
    return np.degrees(radon_angle)

def detect_streaks_and_plot(image_path, filter_size=1, min_angles = 5, blur = True):
    # Get both the corrected and original images
    corrected_image, original_image = remove_horizontal_banding(image_path, filter_size=filter_size)

    # Convert to numpy array if it's not already
    img_raw = np.array(corrected_image)
    if blur:
        img = ndimage.gaussian_filter(img_raw, sigma = 2)
        # plt.imshow(img, cmap='gray')
        # plt.show()
    # Define a list of Canny thresholds to try
    canny_thresholds = [(50, 150), (30, 100), (70, 200), (20, 80), (100, 250), (20, 50)]
    angles = []
    weights = []

    for lower, upper in canny_thresholds:
        angles = []
        weights = []
        # Apply edge detection
        edges = cv2.Canny(img, lower, upper)
        
        # Apply Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is None:
            continue

        # Process the detected lines
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if angle < 0:
                angle += 180
            if 5 < angle < 175:  # Filter out near-horizontal and near-vertical lines
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angles.append(angle)
                weights.append(length)
            

        # If we have enough angles, break the loop
        if len(angles) >= min_angles:
            break
    print(angles)
    # Calculate the weighted average angle
    if angles:
        avg_angle = np.average(angles, weights=weights)
    else:
        avg_angle = None

    return avg_angle, original_image, img

def find_slopes_angles(image_path, filter_size=1, min_angles=5, blur=True):
    """
    This function processes an image to detect streaks and calculate the weighted average angle
    of detected lines using the Lasso regression method.
    
    Args:
        image_path (str): Path to the image file.
        filter_size (int): Size of the filter for removing horizontal banding.
        min_angles (int): Minimum number of angles required to consider detection.
        blur (bool): Whether to apply Gaussian blur to the image.

    Returns:
        avg_angle (float): Weighted average angle of detected lines.
        original_image (ndarray): Original image.
        img (ndarray): Processed image.
    """
    # Get both the corrected and original images
    corrected_image, original_image = remove_horizontal_banding(image_path, filter_size=filter_size)

    # Convert to numpy array if it's not already
    img_raw = np.array(corrected_image)
    img = ndimage.gaussian_filter(img_raw, sigma=2) if blur else img_raw

    # Define a list of Canny thresholds to try
    canny_thresholds = [(50, 150), (30, 100), (70, 200), (20, 80), (100, 250), (20, 50)]
    angles = []
    weights = []

    for lower, upper in canny_thresholds:
        # Apply edge detection
        edges = cv2.Canny(img, lower, upper)
        
        # Find contours of the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = [contour for contour in contours if len(contour) > 20]

        # Process the detected contours
        for contour in contours:
            # Extract the x and y coordinates of the contour points
            x, y = contour[:, 0, 0], contour[:, 0, 1]
            
            # Fit a Lasso regression model to the contour
            lasso = Lasso(alpha=0.1, tol=0.0001)
            X = x.reshape(-1, 1)
            lasso.fit(X, y)
            
            # Compute the start and end points of the line
            start_x, end_x = 0, img.shape[1] - 1
            start_y, end_y = lasso.predict([[start_x]]), lasso.predict([[end_x]])
            slope = (end_y - start_y) / (end_x - start_x)
            angle = np.degrees(np.arctan(slope))

            if 5 < angle < 175:  # Filter out near-horizontal and near-vertical lines
                length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                angles.append(angle)
                weights.append(length)

        # If we have enough angles, break the loop
        if len(angles) >= min_angles:
            break

    # Calculate the weighted average angle
    if angles:
        avg_angle = np.average(angles, weights=weights)
    else:
        avg_angle = None

    return avg_angle, original_image, img

def create_line_mask(size, angle_degrees, line_spacing):
    angle_rad = np.radians(angle_degrees)
    perpendicular = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    y, x = np.indices(size)
    coords = np.column_stack((x.ravel(), y.ravel()))
    proj = coords.dot(perpendicular)
    mask = ((proj // line_spacing) % 2).reshape(size).astype(float)
    return mask

def find_best_angle(image, angle_range, angle_step, line_spacing):
    best_angle = None
    lowest_rms = float('inf')
    
    for angle in np.arange(angle_range[0], angle_range[1], angle_step):
        mask = create_line_mask(image.shape, angle, line_spacing)
        rms = np.sqrt(np.mean((image - mask)**2))
        
        if rms < lowest_rms:
            lowest_rms = rms
            best_angle = angle
    
    return best_angle, lowest_rms

# Function to plot line
def plot_line(img, ax, angle, color, label):
    if angle is None:
        return
    elif angle==0:
        #plot horizontal line on middle row
        ax.axhline(y=img.shape[0]//2, color=color, linewidth=2, label=f"{label}: {angle:.2f}°")
    else:
        height, width = img.shape
        center_x, center_y = width // 2, height // 2
        length = max(width, height) // 2
        radian_angle = np.radians(angle)
        x1 = int(center_x - length * np.cos(radian_angle))
        y1 = int(center_y - length * np.sin(radian_angle))
        x2 = int(center_x + length * np.cos(radian_angle))
        y2 = int(center_y + length * np.sin(radian_angle))
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, label=f"{label}: {angle:.2f}°")
    return 0

def detect_streaks_and_plot_combined(image_path, filter_size=1):
    # Get both the corrected and original images
    corrected_image, original_image = remove_horizontal_banding(image_path, filter_size=filter_size)

    # Convert to numpy array if it's not already
    img = np.array(corrected_image)

    # Normalize image to [0, 1] range
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_gaussian = ndimage.gaussian_filter(img_norm, sigma=2)

    # Steve method
    steve_angle, _ = find_best_angle(img_gaussian, angle_range=(0, 180), angle_step=0.5, line_spacing=3)

    # Normal method
    normal_angle, _, _ = detect_streaks_and_plot(image_path, filter_size=filter_size)

    # find slopes method
    slopes_angle, _, _ = find_slopes_angles(image_path, filter_size=filter_size)

    # Radon transform method
    radon_angle = radon_transform_angle(img_norm)

    # Plot the results
    fig, ax = plt.subplots(figsize=(7,7))

    # Plot corrected image
    ax.imshow(img, cmap='gray')
    ax.set_title("Corrected Image with Detected Angles")

    # Plot lines for each method
    plot_line(img, ax, steve_angle, 'red', 'Steve Method')
    plot_line(img, ax, normal_angle, 'blue', 'Normal Method')
    # plot_line(img, ax, normal_angle*(-1), 'purple', 'normal flip')
    plot_line(img, ax, slopes_angle, 'green', 'Slopes Method')

    ax.legend(loc='upper right')
    ax.axis('off')

    plt.tight_layout()
    plt.show()

    if steve_angle is None:
        steve_angle = 0.101
    if normal_angle is None:
        normal_angle = 0.101
    if radon_angle is None:
        radon_angle = 0.101
    if slopes_angle is None:
        slopes_angle = 0.101

    return steve_angle, normal_angle, radon_angle, img



# Function to display the image
def display_image(img):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Use the function
    folder_path = 'C:\\Users\\ejerison\\capillary-flow\\tests\\tricky_kymographs'
    for filename in os.listdir(folder_path):
        if filename.endswith('.tiff'):
            # filename = 'set01_part09_230414_loc01_vid14_kymograph_03a.tiff'
            image_path = os.path.join(folder_path, filename)
        else:
            continue
        
        steve_angle, normal_angle, radon_angle, corrected_img = detect_streaks_and_plot_combined(image_path)
        print(f"Processing image: {image_path}")
        print(f"Steve Method angle: {steve_angle:.2f} degrees")
        print(f"Normal Method angle: {normal_angle:.2f} degrees")
        print(f"Radon Transform angle: {radon_angle:.2f} degrees")
        print("\n")