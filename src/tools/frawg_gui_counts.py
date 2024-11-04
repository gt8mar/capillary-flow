import os
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import cv2

class KymographCounter:
    def __init__(self, image_dir, csv_path, output_csv_path):
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.output_csv_path = output_csv_path
        self.load_data()
        self.index = self.find_next_uncounted()
        self.current_count = 0
        self.setup_gui()

    def load_data(self):
        if os.path.exists(self.output_csv_path):
            self.df = pd.read_csv(self.output_csv_path)
            if 'Image_Path' not in self.df.columns:
                original_df = pd.read_csv(self.csv_path)
                self.df['Image_Path'] = original_df.apply(lambda row: self.generate_filename(row), axis=1)
        else:
            self.df = pd.read_csv(self.csv_path)
            self.prepare_data()
            self.initialize_output_csv()

    def prepare_data(self):
        tiff_files = {file for file in os.listdir(self.image_dir) if file.lower().endswith(('.tif', '.tiff'))}
        self.df['Image_Path'] = self.df.apply(lambda row: self.generate_filename(row), axis=1)
        self.df = self.df[self.df['Image_Path'].apply(lambda path: os.path.basename(path) in tiff_files)]
        if 'Counts' not in self.df.columns:
            self.df['Counts'] = None

    def generate_filename(self, row):
        condition = row['Condition']
        capillary = row['Capillary']
        return f'CalFrog4fps{condition}_kymograph_0{capillary}.tiff'

    def load_image(self):
        if self.index is None or self.index >= len(self.df) or self.index < 0:
            self.metadata_label.config(text="No more images to count.")
            self.canvas.delete("all")
            return
        
        img_path = self.df.loc[self.index, 'Image_Path']
        image = cv2.imread(os.path.join(self.image_dir, img_path))
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Resize image if it's larger than 800x600
        max_width, max_height = 1200, 600
        img_width, img_height = img.size
        if img_width > max_width or img_height > max_height:
            scale = min(max_width/img_width, max_height/img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(img)
        
        # Update canvas size and scrollregion
        self.canvas.config(scrollregion=(0, 0, img.width, img.height))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        self.metadata_label.config(text=f"Index: {self.index}\nCurrent Count: {self.current_count}")

    def save_count(self, event):
        if self.current_count > 0:
            self.df.at[self.index, 'Counts'] = self.current_count
            self.update_output_csv()
            self.index = self.find_next_uncounted()
            self.current_count = 0
            if self.index is not None:
                self.load_image()
            else:
                print("No more images to count.")
                self.root.destroy()
        else:
            messagebox.showinfo("Invalid Count", "Please enter a count greater than 0 before saving.")

    def set_count(self, event):
        key = event.keysym
        if key.isdigit():
            self.current_count = self.current_count * 10 + int(key)
            self.metadata_label.config(text=f"Index: {self.index}\nCurrent Count: {self.current_count}")

    def clear_count(self, event):
        self.current_count = 0
        self.metadata_label.config(text=f"Index: {self.index}\nCurrent Count: {self.current_count}")

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Kymograph Counter")
        self.root.bind('<c>', self.save_count)
        self.root.bind('<BackSpace>', self.clear_count)
        for i in range(10):
            self.root.bind(str(i), self.set_count)
        
        # Create a frame for the canvas and scrollbars
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas and scrollbars
        self.canvas = tk.Canvas(frame, width=1200, height=600)
        h_scrollbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        # Configure canvas
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        self.metadata_label = tk.Label(self.root, text="", wraplength=400)
        self.metadata_label.pack()

        if self.index is not None:
            self.load_image()
        else:
            print("No valid images found in the directory.")
        self.root.mainloop()

    def initialize_output_csv(self):
        output_df = self.df[['Date', 'Frog', 'Side', 'Condition', 'Capillary', 'Velocity (um/s)', 'Image_Path', 'Counts']]
        output_df.to_csv(self.output_csv_path, index=False)

    def update_output_csv(self):
        self.df.to_csv(self.output_csv_path, index=False)

    def find_next_uncounted(self, start_index=0):
        for idx in range(start_index, len(self.df)):
            if pd.isna(self.df.at[idx, 'Counts']):
                return idx
        return None

# average_slope = (um_slope * 0.8) / fps

if __name__ == "__main__":
    counter = KymographCounter(
        'D:\\frog\\kymographs', 
        'D:\\frog\\velocities\\CalFrog4fps220Lankle_velocities_u.csv',
        'D:\\frog\\counted_kymos_CalFrog4.csv'
    )