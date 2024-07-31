import os
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2

class KymographClassifier:
    def __init__(self, image_dir, metadata_dir, csv_path, output_csv_path):
        self.image_dir = image_dir
        self.metadata_dir = metadata_dir
        self.csv_path = csv_path
        self.output_csv_path = output_csv_path
        self.df = pd.read_csv(csv_path)
        self.prepare_data()
        self.index = 0
        self.inverted_slope = False
        self.high_velocities = [10, 420, 500, 600, 750, 1000, 1500, 2000, 3000, 4000]  # Original velocities
        self.additional_velocities = [10, 20, 35, 50, 75, 110, 160, 220, 290, 360]  # Additional velocities with Shift key
        self.original_velocity = None
        self.current_velocity_index = 0
        self.use_additional_velocities = False  # Toggle between high and additional velocities
        self.initial_classification_complete = False
        self.initial_classification = None
        self.initialize_output_csv()
        self.setup_gui()

    def prepare_data(self):
        tiff_files = {file for file in os.listdir(self.image_dir) if file.lower().endswith(('.tif', '.tiff'))}
        self.df['Image_Path'] = self.df.apply(lambda row: self.generate_filename(row), axis=1)
        self.df = self.df[self.df['Image_Path'].apply(lambda path: os.path.basename(path) in tiff_files)]
        self.df['Initial_Classification'] = None
        self.df['Classified_Velocity'] = None
        self.df['Second_Classification'] = None

    def generate_filename(self, row):
        return f"set01_{row['Participant']}_{row['Date']}_{row['Location']}_{row['Video']}_kymograph_{row['Capillary']}.tiff"

    def get_fps(self, participant, date, location, video):
        filename = f'{participant}_{int(date)}.xlsx'
        df = pd.read_excel(os.path.join(self.metadata_dir, filename))
        row = df[(df['Participant'] == participant) & (df['Date'] == date) & (df['Video'] == video)]
        return row['FPS'].values[0] if not row.empty else None

    def load_image(self):
        if self.index is None or self.index >= len(self.df) or self.index < 0:
            self.metadata_label.config(text="No more images to classify.")
            self.panel.config(image='')
            self.panel.image = None
            return

        img_path = self.df.loc[self.index, 'Image_Path']
        image = cv2.imread(os.path.join(self.image_dir, img_path))
        fps = self.get_fps(self.df.loc[self.index, 'Participant'], self.df.loc[self.index, 'Date'], self.df.loc[self.index, 'Location'], self.df.loc[self.index, 'Video'])
        um_slope = self.df.loc[self.index, 'Velocity']
        if self.current_velocity_index == 0:
            average_slope = (um_slope * 2.44) / fps
            self.original_velocity = um_slope
        else:
            velocities = self.additional_velocities if self.use_additional_velocities else self.high_velocities
            average_slope = (velocities[self.current_velocity_index] * 2.44) / fps
        if self.inverted_slope:
            average_slope = -average_slope
        if average_slope != 0:
            end_x = int((image.shape[0]-1) / average_slope) + int(image.shape[1] / 2)
            cv2.line(image, (int(image.shape[1]/2), 0), (end_x, image.shape[0]-1), (255, 255, 0), 2)
        else:
            cv2.line(image, (0, int(image.shape[0]/2)), (image.shape[1]-1, int(image.shape[0]/2)), (255, 255, 0), 2)
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = img.resize((500, 500), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.panel.config(image=photo)
        self.panel.image = photo
        self.metadata_label.config(text=f"Index: {self.index}\nVelocity: {self.df.loc[self.index, 'Velocity']}\nInitial_Classification: {self.df.loc[self.index, 'Initial_Classification']}")
        velocities = self.additional_velocities if self.use_additional_velocities else self.high_velocities
        current_velocity = velocities[self.current_velocity_index] if self.current_velocity_index != 0 else um_slope
        self.metadata_label.config(text=f"Index: {self.index}\nVelocity: {um_slope}\nCurrent Set Velocity: {current_velocity}\nInitial_Classification: {self.df.loc[self.index, 'Initial_Classification']}")

    def toggle_slope(self):
        self.inverted_slope = not self.inverted_slope
        self.load_image()

    def toggle_velocity_set(self, event):
        self.use_additional_velocities = not self.use_additional_velocities
        print("Using additional velocities" if self.use_additional_velocities else "Using high velocities")
        self.load_image()
    
    def first_classification(self, classification):
        if classification in ['Too Fast', 'Too Slow']:
            self.df.at[self.index, 'Initial_Classification'] = classification
            self.initial_classification_complete = True
            self.current_velocity_index = 0
            self.load_image()
        else: 
            messagebox.showinfo("Try again", "Please classify the original velocity")

    def unclear_classification(self, classification):
        self.df.at[self.index, 'Initial_Classification'] = 'Unclear'
        self.df.at[self.index, 'Classified_Velocity'] = self.original_velocity
        self.df.at[self.index, 'Second_Classification'] = 'Unclear'
        self.update_output_csv()
        self.index = self.find_next_unclassified()
        self.initial_classification_complete = False
        self.current_velocity_index = 0
        if self.index is not None:
            self.load_image()
        else:
            print("No more images to classify.")
    
    def zero_classification(self, classification):
        self.df.at[self.index, 'Initial_Classification'] = 'Zero'
        self.df.at[self.index, 'Classified_Velocity'] = 0
        self.df.at[self.index, 'Second_Classification'] = 'Correct'
        self.update_output_csv()
        self.index = self.find_next_unclassified()
        self.initial_classification_complete = False
        self.current_velocity_index = 0
        if self.index is not None:
            self.load_image()
        else:
            print("No more images to classify.")
            

    # def save_classification(self, classification):
    #     if pd.isna(self.df.at[self.index, 'Initial_Classification']) or self.df.at[self.index, 'Initial_Classification'] == 'Correct':
    #         self.df.at[self.index, 'Initial_Classification'] = classification
    #     else:
    #         self.df.at[self.index, 'Second_Classification'] = classification
    #     if classification in ['Too Fast', 'Too Slow']:
    #         velocities = self.additional_velocities if self.use_additional_velocities else self.high_velocities
    #         self.df.at[self.index, 'Classified_Velocity'] = velocities[self.current_velocity_index - 1] if self.current_velocity_index != 0 else self.df.at[self.index, 'Velocity']
    #     elif classification == 'Zero':
    #         self.df.at[self.index, 'Classified_Velocity'] = 0
    #     elif classification == 'Unsure':
    #         self.df.at[self.index, 'Classified_Velocity'] = self.original_velocity
    #     elif classification == 'Correct':
    #         self.df.at[self.index, 'Classified_Velocity'] = self.original_velocity
    #     self.update_output_csv()

        
    def correct_classification(self):
        if self.initial_classification_complete:
            self.df.at[self.index, 'Second_Classification'] = 'Correct'
            velocities = self.additional_velocities if self.use_additional_velocities else self.high_velocities
            self.df.at[self.index, 'Classified_Velocity'] = velocities[self.current_velocity_index] if self.current_velocity_index != 0 else self.df.at[self.index, 'Velocity']
        else: 
            self.df.at[self.index, 'Initial_Classification'] = 'Correct'
            self.df.at[self.index, 'Classified_Velocity'] = self.original_velocity
            self.df.at[self.index, 'Second_Classification'] = 'Correct'
        self.update_output_csv()
        self.index = self.find_next_unclassified()
        self.current_velocity_index = 0
        self.initial_classification_complete = False
        if self.index is not None:
            self.load_image()
        else:
            print("No more images to classify.")
            self.root.destroy()
        
        

    def next_image(self):
        self.index = self.find_next_unclassified(start_index=self.index + 1)
        if self.index is not None:
            self.load_image()
        else:
            self.metadata_label.config(text="No more images to classify.")
            self.panel.config(image='')
            self.panel.image = None

    def previous_image(self):
        self.index = self.index -1
        if self.index is not None and self.index >= 0:
            self.load_image()
        elif self.index < 0:
            self.index = 0
            self.load_image()
        else:
            self.metadata_label.config(text="No more images to classify.")
            self.panel.config(image='')
            self.panel.image = None

    def set_velocity(self, event):
        if self.initial_classification_complete:
            key = event.keysym
            if key.isdigit():
                key = int(key)
                if key == 0:
                    self.current_velocity_index = 0
                elif 1 <= key <= 9:
                    self.current_velocity_index = key
                self.load_image()
        else:
            messagebox.showinfo("Try again", "Please classify the original velocity first.")

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Kymograph Classification")
        self.root.bind('<c>', lambda event: self.correct_classification())
        self.root.bind('<f>', lambda event: self.first_classification('Too Fast'))
        self.root.bind('<s>', lambda event: self.first_classification('Too Slow'))
        self.root.bind('<z>', lambda event: self.zero_classification('Zero'))
        self.root.bind('<u>', lambda event: self.unclear_classification('Unsure'))
        self.root.bind('<p>', lambda event: self.toggle_slope())
        self.root.bind('<b>', lambda event: self.previous_image())
        self.root.bind('<n>', lambda event: self.next_image())
        self.root.bind('<Shift_L>', self.toggle_velocity_set)
        self.root.bind('<Shift_R>', self.toggle_velocity_set)
        for i in range(10):
            self.root.bind(str(i), self.set_velocity)
            
        self.panel = tk.Label(self.root)
        self.panel.pack(fill="both", expand="yes")
        self.metadata_label = tk.Label(self.root, text="", wraplength=400)
        self.metadata_label.pack()
        self.index = self.find_next_unclassified()
        if self.index is not None:
            self.load_image()
        else:
            print("No valid images found in the directory.")
        self.root.mainloop()

    def initialize_output_csv(self):
        if not os.path.exists(self.output_csv_path):
            output_df = self.df[['Participant', 'Date', 'Location', 'Video', 'Image_Path', 'Velocity']]
            output_df['Classified_Velocity'] = None
            output_df['Initial_Classification'] = None
            output_df['Second_Classification'] = None
            output_df.to_csv(self.output_csv_path, index=False)
        else:
            self.df = pd.read_csv(self.output_csv_path)

    def update_output_csv(self):
        output_df = pd.read_csv(self.output_csv_path)
        current_row = self.df.iloc[self.index]
        output_df.loc[self.index] = [
            current_row['Participant'],
            current_row['Date'],
            current_row['Location'],
            current_row['Video'],
            current_row['Image_Path'],
            current_row['Velocity'],
            current_row['Classified_Velocity'],
            current_row['Initial_Classification'],
            current_row['Second_Classification']
        ]
        output_df.to_csv(self.output_csv_path, index=False)

    def find_next_unclassified(self, start_index=0):
        for idx in range(start_index, len(self.df)):
            if self.df.at[idx, 'Second_Classification'] != 'Correct':
                if self.df.at[idx,'Second_Classification'] != 'Unclear':
                    return idx
                else:
                    continue
        return None

    def find_previous_unclassified(self, start_index):
        for idx in range(start_index, -1, -1):
            if self.df.at[idx, 'Second_Classification'] != 'Correct':
                return idx
        return None

if __name__ == "__main__":
    classifier = KymographClassifier(
        'C:\\Users\\gt8mar\\capillary-flow\\results\\kymographs\\tricky_kymographs', 
        'C:\\Users\\gt8mar\\capillary-flow\\metadata', 
        'C:\\Users\\gt8mar\\capillary-flow\\tricky_kymos.csv',
        'C:\\Users\\gt8mar\\capillary-flow\\classified_kymos.csv'
    )
