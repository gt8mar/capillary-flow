import os
import pandas as pd
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2

def get_fps(participant, date, location, video):
    # Construct the filename based on the participant and date
    filename = f'{participant}_{int(date)}.xlsx'
    metadata_folder = 'C:\\Users\\gt8mar\\capillary-flow\\metadata'
    
    # Load the dataframe from the Excel file
    df = pd.read_excel(os.path.join(metadata_folder,filename))
    
    # Query the dataframe for the specific row based on participant, date, location, and video
    row = df[(df['Participant'] == participant) & 
             (df['Date'] == date) &  
             (df['Video'] == video)]
    
    # Return the FPS value if the row exists, otherwise return None
    if not row.empty:
        return row['FPS'].values[0]
    else:
        return None
    

# Function to generate filename from DataFrame row
def generate_filename(row):
    return f"set01_{row['Participant']}_{row['Date']}_{row['Location']}_{row['Video']}_kymograph_{row['Capillary']}.tiff"


# Function to load and display the image
def load_image(df, idx, invert_slope=False):
    global inverted_slope
    inverted_slope = invert_slope
    img_path = df.loc[idx, 'Image_Path']
    # Load the image using OpenCV
    image = cv2.imread(img_path)
    
    # Transform slope from um/s into pixels/frame:
    fps = get_fps(df.loc[idx, 'Participant'], df.loc[idx, 'Date'], df.loc[idx, 'Location'], df.loc[idx, 'Video'])
    PIX_UM = 2.44
    um_slope = df.loc[idx, 'Velocity']
    average_slope = (um_slope * PIX_UM)/fps 

    # Invert the slope if toggled
    if inverted_slope:
        average_slope = -average_slope 

    # Calculate the end coordinates of the line using the slope
    end_x = int((image.shape[0]-1) / average_slope) + int(image.shape[1] / 2)
    end_y = image.shape[0]-1

    # Draw the line on the image
    cv2.line(image, (int(image.shape[1]/2), 0), (end_x, end_y), (255, 255, 0), 2)

    # Convert the OpenCV image (which is numpy array) back to a PIL image
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = img.resize((500, 500), Image.Resampling.LANCZOS)  # Resize for better viewing
    photo = ImageTk.PhotoImage(img)
    panel.config(image=photo)
    panel.image = photo
    metadata_label.config(text=f"Index: {idx}\nVelocity: {df.loc[idx, 'Velocity']}\nClassification: {df.loc[idx, 'Classification']}")

def toggle_slope(df):
    global index, inverted_slope
    load_image(df, index, not inverted_slope)

# Function to save classification
def save_classification(df, classification):
    global index
    df.at[index, 'Classification'] = classification
    index += 1
    if index < len(df):
        load_image(df, index)
    else:
        print("No more images to classify.")
        root.destroy()  # Close the window if no more images
        df.to_csv('C:\\Users\\gt8mar\\capillary-flow\\classified_kymos.csv', index=False)  # Save the classifications to a new CSV

def main():
      # Directory containing kymographs
    image_dir = 'C:\\Users\\gt8mar\\capillary-flow\\results\\kymographs\\tricky_kymographs'
    # Load CSV with existing velocity data
    csv_path = 'C:\\Users\\gt8mar\\capillary-flow\\tricky_kymos.csv'
    df = pd.read_csv(csv_path)

    # List all TIFF files in the directory
    tiff_files = {file for file in os.listdir(image_dir) if file.lower().endswith(('.tif', '.tiff'))}

    # Add full image paths to the DataFrame and filter by existing files
    df['Image_Path'] = df.apply(lambda row: os.path.join(image_dir, generate_filename(row)), axis=1)
    df = df[df['Image_Path'].apply(lambda path: os.path.basename(path) in tiff_files)]

    # Ensure columns exist in DataFrame
    df['Classification'] = None

    # Initialize the image index
    inverted_slope = False

    global root, index, panel, metadata_label
    index = 0
    # Setup the GUI
    root = tk.Tk()
    root.title("Kymograph Classification")

    # Event bindings for classification keys
    root.bind('<c>', lambda event: save_classification(df, 'Correct'))
    root.bind('<f>', lambda event: save_classification(df, 'Too Fast'))
    root.bind('<s>', lambda event: save_classification(df, 'Too Slow'))
    root.bind('<z>', lambda event: save_classification(df, 'Zero'))
    root.bind('<u>', lambda event: save_classification(df, 'Unsure'))
    root.bind('<p>', lambda event: toggle_slope(df))

    # Image panel
    panel = tk.Label(root)
    panel.pack(fill="both", expand="yes")

    # Metadata display
    metadata_label = tk.Label(root, text="", wraplength=400)
    metadata_label.pack()

    # Load the first image if the DataFrame is not empty
    if not df.empty:
        load_image(df, index)
    else:
        print("No valid images found in the directory.")

    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()