import os
import pandas as pd
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2

class KymographClassifier:
    def __init__(self, image_dir, metadata_dir, csv_path):
        self.image_dir = image_dir
        self.metadata_dir = metadata_dir
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.prepare_data()
        self.index = 0
        self.inverted_slope = False
        self.high_velocities = [10, 750, 1000, 1500, 2000, 3000] 
        self.current_velocity_index = 0 
        self.setup_gui()

    def prepare_data(self):
        # List all TIFF files in the directory
        tiff_files = {file for file in os.listdir(self.image_dir) if file.lower().endswith(('.tif', '.tiff'))}
        # Add full image paths to the DataFrame and filter by existing files
        self.df['Image_Path'] = self.df.apply(lambda row: self.generate_filename(row), axis=1)
        self.df = self.df[self.df['Image_Path'].apply(lambda path: os.path.basename(path) in tiff_files)]
        self.df['Classification'] = None

    def generate_filename(self, row):
        return f"set01_{row['Participant']}_{row['Date']}_{row['Location']}_{row['Video']}_kymograph_{row['Capillary']}.tiff"

    def get_fps(self, participant, date, location, video):
        filename = f'{participant}_{int(date)}.xlsx'
        df = pd.read_excel(os.path.join(self.metadata_dir, filename))
        row = df[(df['Participant'] == participant) & (df['Date'] == date) & (df['Video'] == video)]
        return row['FPS'].values[0] if not row.empty else None

    def load_image(self):
        img_path = self.df.loc[self.index, 'Image_Path']
        image = cv2.imread(os.path.join(self.image_dir, img_path))
        fps = self.get_fps(self.df.loc[self.index, 'Participant'], self.df.loc[self.index, 'Date'], self.df.loc[self.index, 'Location'], self.df.loc[self.index, 'Video'])
        um_slope = self.df.loc[self.index, 'Velocity']
        if self.current_velocity_index == 0:
            average_slope = (um_slope * 2.44) / fps
        else:
            average_slope = (self.high_velocities[self.current_velocity_index] * 2.44) / fps
        if self.inverted_slope:
            average_slope = -average_slope
        end_x = int((image.shape[0]-1) / average_slope) + int(image.shape[1] / 2)
        cv2.line(image, (int(image.shape[1]/2), 0), (end_x, image.shape[0]-1), (255, 255, 0), 2)
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = img.resize((500, 500), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.panel.config(image=photo)
        self.panel.image = photo
        self.metadata_label.config(text=f"Index: {self.index}\nVelocity: {self.df.loc[self.index, 'Velocity']}\nClassification: {self.df.loc[self.index, 'Classification']}")

    def toggle_slope(self):
        self.inverted_slope = not self.inverted_slope
        self.load_image()

    def save_classification(self, classification):
        self.df.at[self.index, 'Classification'] = classification
        self.index += 1
        self.current_velocity_index = 0
        if self.index < len(self.df):
            self.load_image()
        else:
            print("No more images to classify.")
            self.root.destroy()
            self.df.to_csv(self.csv_path, index=False)
    
    def next_image(self):
        if self.index + 1 < len(self.df):
            self.index += 1
            self.current_velocity_index = 0
            self.load_image()
        else:
            print("No more images to classify.")
            self.root.destroy()
            self.df.to_csv(self.csv_path, index=False)

    def previous_image(self):
        if self.index > 0:
            self.index -= 1
            self.current_velocity_index = 0
            self.load_image()
    
    def cycle_velocities(self):
        self.current_velocity_index = (self.current_velocity_index + 1) % len(self.high_velocities)
        self.load_image()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Kymograph Classification")
        self.root.bind('<c>', lambda event: self.save_classification('Correct'))
        self.root.bind('<f>', lambda event: self.save_classification('Too Fast'))
        self.root.bind('<s>', lambda event: self.cycle_velocities())  # Bind 's' to cycle through velocities
        self.root.bind('<z>', lambda event: self.save_classification('Zero'))
        self.root.bind('<u>', lambda event: self.save_classification('Unsure'))
        self.root.bind('<p>', lambda event: self.toggle_slope())
        self.root.bind('<b>', lambda event: self.previous_image())
        self.root.bind('<n>', lambda event: self.next_image())
        self.panel = tk.Label(self.root)
        self.panel.pack(fill="both", expand="yes")
        self.metadata_label = tk.Label(self.root, text="", wraplength=400)
        self.metadata_label.pack()
        if not self.df.empty:
            self.load_image()
        else:
            print("No valid images found in the directory.")
        self.root.mainloop()

if __name__ == "__main__":
    classifier = KymographClassifier('C:\\Users\\gt8mar\\capillary-flow\\results\\kymographs\\tricky_kymographs', 'C:\\Users\\gt8mar\\capillary-flow\\metadata', 'C:\\Users\\gt8mar\\capillary-flow\\tricky_kymos.csv')
