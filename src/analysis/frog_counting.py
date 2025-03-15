import glob, os
import numpy as np
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, convolve
from scipy.ndimage import gaussian_filter1d
from matplotlib.font_manager import FontProperties
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure as mpl_fig

FPS = 130 #113.9 #227.8 #169.3
PIX_UM = 0.8 #2.44 #1.74
source_sans = FontProperties(fname='C:\\Users\\ejerison\\Downloads\\Source_Sans_3\\static\\SourceSans3-Regular.ttf')


def setup_plotting_style():
    """Set up consistent plotting style according to coding standards."""
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 5,
        'lines.linewidth': 0.5,
        'figure.figsize': (2.4, 2.0)
    })



def overlay_x_values_on_kymograph(image, x_values, color='r', linewidth=1):
    """
    Overlays vertical lines at specified x-values on a rotated kymograph image.
    
    Parameters
    ----------
    image : 2D numpy array
        The rotated kymograph image array (rows = y, columns = x).
    x_values : array-like
        The x (column) coordinates where RBC lines have been detected.
    color : str, optional
        Color of the overlaid lines. Default is 'r' (red).
    linewidth : float, optional
        Thickness of the overlaid lines. Default is 1.
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap='gray', aspect='auto')
    
    # Overlay vertical lines at each detected RBC position
    for x in x_values:
        plt.axhline(y=x, color=color, linewidth=linewidth)
    
    plt.title("Rotated Kymograph with Detected RBC Positions")
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.show()

def analyze_kymograph(path_to_kymograph, counts_df, prominence_threshold=0.05):
    """
    Analyze a kymograph to detect RBCs.
    Returns the detected RBC count, rotated kymograph, profile, and peaks.
    """
    # -----------------------------
    # Step 1: Load the Kymograph, velocity, and estimated counts
    # -----------------------------
    # The kymograph is assumed to be a 2D grayscale image (rows=frames, cols=position)
    kymograph = io.imread(path_to_kymograph)  # or png, etc.
    # slice into bottom half of the kymograph
    kymograph = kymograph[kymograph.shape[0]//2:, :]
    kymograph = kymograph.astype(float)  # ensure floating point
    # get other data from counts_df based on the filename
    counts_row = counts_df[counts_df['Image_Path'] == os.path.basename(path_to_kymograph)]
    image_name = os.path.basename(path_to_kymograph)
    rbc_velocity_um_s = counts_row['Classified_Velocity'].values[0]
    rbc_count_est = counts_row['Counts'].values[0]

    # -----------------------------
    # Step 2: Determine Rotation Angle
    # -----------------------------
    # Calculate the average velocity of RBCs (in pixels/frame)
    rbc_velocity = rbc_velocity_um_s / (PIX_UM * FPS)

    # Angle in radians:
    theta_radians = np.arctan(rbc_velocity)
    # Convert to degrees:
    theta_degrees = np.degrees(theta_radians)

    # Rotate image so RBC lines are vertical:
    # If RBCs tilt towards the right, you likely need a negative angle to correct.
    rotated = transform.rotate(kymograph, -theta_degrees, resize=True)

    # After rotation, RBC traces should be more vertical.

    # -----------------------------
    # Step 3: Average Pixels Along Each Row
    # -----------------------------
    # Now that RBC lines are vertical, we can compress each row into a single intensity value
    # by averaging across columns. This gives a 1D intensity profile.
    profile = rotated.mean(axis=1)  # average each row

    # -----------------------------
    # Step 5: Threshold or Peak Detection with Adaptive Parameters
    # -----------------------------
    # If RBCs appear as darker lines (lower intensity), invert the profile:
    inverted = -profile

    # Calculate expected distance between peaks based on estimated count
    expected_distance = rotated.shape[0] / max(1, rbc_count_est)  # Avoid division by zero

    # Initial peak detection with adaptive parameters
    peaks, properties = find_peaks(inverted, 
                                  prominence=prominence_threshold,
                                  distance=max(1, expected_distance * 0.7),  # Minimum expected spacing
                                  height=None)
    
    initial_count = len(peaks)
    
    # Final RBC count
    rbc_count = len(peaks)

    return rbc_count, rotated, profile, peaks, inverted, rbc_count_est

class RBCCounterGUI:
    def __init__(self, kymograph_dir, counts_df_path):
        self.kymograph_dir = kymograph_dir
        self.counts_df = pd.read_csv(counts_df_path)
        self.current_index = 0
        self.kymograph_files = self.get_kymograph_files()
        self.manual_count_mode = False
        self.manual_count_value = ""
        self.prominence_threshold = 0.05
        
        # Create output dataframe with additional columns
        self.output_df = self.counts_df.copy()
        if 'Measured_Counts' not in self.output_df.columns:
            self.output_df['Measured_Counts'] = None
        if 'Final_Counts' not in self.output_df.columns:
            self.output_df['Final_Counts'] = None
        if 'Adjustment_Type' not in self.output_df.columns:
            self.output_df['Adjustment_Type'] = None
        
        # Save path for output
        output_filename = counts_df_path.replace('counts', 'final_counts')
        if output_filename == counts_df_path:
            output_filename = os.path.splitext(counts_df_path)[0] + '_final' + os.path.splitext(counts_df_path)[1]
        self.output_path = output_filename
        
        self.setup_gui()
        self.analyze_current_kymograph()
        
    def get_kymograph_files(self):
        """Get list of kymograph files that match entries in the counts dataframe"""
        all_files = glob.glob(os.path.join(self.kymograph_dir, '*.tiff'))
        all_files.extend(glob.glob(os.path.join(self.kymograph_dir, '*.tif')))
        
        # Filter to only include files in the dataframe
        valid_files = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            if filename in self.counts_df['Image_Path'].values:
                valid_files.append(file_path)
        
        return valid_files
    
    def setup_gui(self):
        """Set up the GUI interface"""
        self.root = tk.Tk()
        self.root.title("RBC Counter GUI")
        self.root.geometry("1200x800")
        
        # Create frames
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Info labels
        self.file_label = tk.Label(self.info_frame, text="File: ", font=("Arial", 12))
        self.file_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.count_label = tk.Label(self.info_frame, text="Counts: ", font=("Arial", 12))
        self.count_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.status_label = tk.Label(self.info_frame, text="Status: Ready", font=("Arial", 12))
        self.status_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Manual count entry
        self.manual_entry_label = tk.Label(self.info_frame, text="Manual count: ", font=("Arial", 12))
        self.manual_entry_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.manual_entry_value = tk.Label(self.info_frame, text="", font=("Arial", 12))
        self.manual_entry_value.grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Create matplotlib figures
        self.setup_plots()
        
        # Key bindings
        self.root.bind('c', self.mark_as_correct)
        self.root.bind('n', self.next_kymograph)
        self.root.bind('b', self.previous_kymograph)
        self.root.bind('m', self.mark_as_too_many)
        self.root.bind('f', self.mark_as_too_few)
        
        # Number key bindings for manual count
        for i in range(10):
            self.root.bind(str(i), self.add_digit)
        
        # Backspace for manual count
        self.root.bind('<BackSpace>', self.remove_digit)
        
        # Instructions
        instructions = """
        Keyboard Controls:
        c - Mark as correct
        n - Next kymograph
        b - Previous kymograph
        m - Too many (increase prominence)
        f - Too few (enter manual count)
        0-9 - Enter digits for manual count
        Backspace - Remove last digit
        """
        
        self.instructions_label = tk.Label(self.control_frame, text=instructions, font=("Arial", 10), justify=tk.LEFT)
        self.instructions_label.pack(side=tk.LEFT, padx=10, pady=10)
    
    def setup_plots(self):
        """Set up the matplotlib plots"""
        # Create figure for profile plot
        self.profile_fig = mpl_fig.Figure(figsize=(6, 3), dpi=100)
        self.profile_ax = self.profile_fig.add_subplot(111)
        self.profile_canvas = FigureCanvasTkAgg(self.profile_fig, master=self.plot_frame)
        self.profile_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create figure for kymograph
        self.kymo_fig = mpl_fig.Figure(figsize=(6, 3), dpi=100)
        self.kymo_ax = self.kymo_fig.add_subplot(111)
        self.kymo_canvas = FigureCanvasTkAgg(self.kymo_fig, master=self.plot_frame)
        self.kymo_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def analyze_current_kymograph(self):
        """Analyze the current kymograph and update the display"""
        if not self.kymograph_files or self.current_index >= len(self.kymograph_files):
            self.status_label.config(text="No more kymographs to analyze")
            return
        
        current_file = self.kymograph_files[self.current_index]
        filename = os.path.basename(current_file)
        
        # Update file label
        self.file_label.config(text=f"File: {filename} ({self.current_index + 1}/{len(self.kymograph_files)})")
        
        # Analyze the kymograph
        rbc_count, rotated, profile, peaks, inverted, rbc_count_est = analyze_kymograph(
            current_file, self.counts_df, self.prominence_threshold
        )
        
        self.current_rotated = rotated
        self.current_profile = profile
        self.current_peaks = peaks
        self.current_inverted = inverted
        self.current_count = rbc_count
        self.current_est_count = rbc_count_est
        
        # Update count label
        self.count_label.config(text=f"Estimated: {rbc_count_est}, Measured: {rbc_count}")
        
        # Update plots
        self.update_plots()
        
        # Update output dataframe
        idx = self.counts_df[self.counts_df['Image_Path'] == filename].index[0]
        self.output_df.at[idx, 'Measured_Counts'] = rbc_count
        if pd.isna(self.output_df.at[idx, 'Final_Counts']):
            self.output_df.at[idx, 'Final_Counts'] = rbc_count
        
        # Save the output dataframe
        self.output_df.to_csv(self.output_path, index=False)
    
    def update_plots(self):
        """Update the matplotlib plots"""
        # Clear previous plots
        self.profile_ax.clear()
        self.kymo_ax.clear()
        
        # Plot profile with peaks
        self.profile_ax.plot(self.current_profile, label='Intensity Profile')
        self.profile_ax.plot(self.current_peaks, self.current_profile[self.current_peaks], 'rx', label='Detected RBCs')
        self.profile_ax.set_title(f'Vertical Intensity Profile (Detected: {self.current_count}, Estimated: {self.current_est_count})')
        self.profile_ax.set_xlabel('Row Index')
        self.profile_ax.set_ylabel('Intensity')
        self.profile_ax.legend()
        
        # Plot kymograph with overlaid peaks
        self.kymo_ax.imshow(self.current_rotated, cmap='gray', aspect='auto')
        for x in self.current_peaks:
            self.kymo_ax.axhline(y=x, color='r', linewidth=1)
        self.kymo_ax.set_title("Rotated Kymograph with Detected RBC Positions")
        self.kymo_ax.set_xlabel("X Position (pixels)")
        self.kymo_ax.set_ylabel("Y Position (pixels)")
        
        # Draw the canvases
        self.profile_fig.tight_layout()
        self.kymo_fig.tight_layout()
        self.profile_canvas.draw()
        self.kymo_canvas.draw()
    
    def mark_as_correct(self, event=None):
        """Mark the current kymograph count as correct"""
        if self.manual_count_mode:
            # If in manual count mode, use the manual count value
            if self.manual_count_value:
                try:
                    manual_count = int(self.manual_count_value)
                    filename = os.path.basename(self.kymograph_files[self.current_index])
                    idx = self.counts_df[self.counts_df['Image_Path'] == filename].index[0]
                    self.output_df.at[idx, 'Final_Counts'] = manual_count
                    self.output_df.at[idx, 'Adjustment_Type'] = 'Manual'
                    self.status_label.config(text=f"Saved manual count: {manual_count}")
                    
                    # Reset manual count mode
                    self.manual_count_mode = False
                    self.manual_count_value = ""
                    self.manual_entry_value.config(text="")
                    
                    # Move to next kymograph
                    self.next_kymograph()
                except ValueError:
                    self.status_label.config(text="Invalid manual count value")
            else:
                self.status_label.config(text="Please enter a manual count value")
        else:
            # Mark the current count as correct
            filename = os.path.basename(self.kymograph_files[self.current_index])
            idx = self.counts_df[self.counts_df['Image_Path'] == filename].index[0]
            self.output_df.at[idx, 'Final_Counts'] = self.current_count
            self.output_df.at[idx, 'Adjustment_Type'] = 'Correct'
            self.status_label.config(text="Marked as correct")
            
            # Save the output dataframe
            self.output_df.to_csv(self.output_path, index=False)
            
            # Move to next kymograph
            self.next_kymograph()
    
    def next_kymograph(self, event=None):
        """Move to the next kymograph"""
        if self.manual_count_mode:
            self.status_label.config(text="Please complete manual count first")
            return
            
        if self.current_index < len(self.kymograph_files) - 1:
            self.current_index += 1
            self.prominence_threshold = 0.05  # Reset prominence threshold
            self.analyze_current_kymograph()
        else:
            self.status_label.config(text="No more kymographs to analyze")
    
    def previous_kymograph(self, event=None):
        """Move to the previous kymograph"""
        if self.manual_count_mode:
            self.status_label.config(text="Please complete manual count first")
            return
            
        if self.current_index > 0:
            self.current_index -= 1
            self.prominence_threshold = 0.05  # Reset prominence threshold
            self.analyze_current_kymograph()
        else:
            self.status_label.config(text="Already at the first kymograph")
    
    def mark_as_too_many(self, event=None):
        """Mark as too many RBCs detected and increase prominence threshold"""
        if self.manual_count_mode:
            self.status_label.config(text="Please complete manual count first")
            return
            
        # Increase prominence threshold
        self.prominence_threshold += 0.02
        self.status_label.config(text=f"Increased prominence to {self.prominence_threshold:.2f}")
        
        # Re-analyze with new threshold
        self.analyze_current_kymograph()
        
        # Update adjustment type
        filename = os.path.basename(self.kymograph_files[self.current_index])
        idx = self.counts_df[self.counts_df['Image_Path'] == filename].index[0]
        self.output_df.at[idx, 'Adjustment_Type'] = f'Increased prominence to {self.prominence_threshold:.2f}'
    
    def mark_as_too_few(self, event=None):
        """Enter manual count mode for too few RBCs"""
        if not self.manual_count_mode:
            self.manual_count_mode = True
            self.manual_count_value = ""
            self.manual_entry_value.config(text="")
            self.status_label.config(text="Enter manual count and press 'c' to confirm")
    
    def add_digit(self, event=None):
        """Add a digit to the manual count value"""
        if self.manual_count_mode:
            self.manual_count_value += event.char
            self.manual_entry_value.config(text=self.manual_count_value)
    
    def remove_digit(self, event=None):
        """Remove the last digit from the manual count value"""
        if self.manual_count_mode and self.manual_count_value:
            self.manual_count_value = self.manual_count_value[:-1]
            self.manual_entry_value.config(text=self.manual_count_value)

def main(path_to_kymograph, counts_df):
    # This function is kept for backward compatibility
    rbc_count, rotated, profile, peaks, inverted, rbc_count_est = analyze_kymograph(path_to_kymograph, counts_df)
    
    # -----------------------------
    # Step 7: Visual Inspection
    # -----------------------------
    setup_plotting_style()

    plt.figure(figsize=(2.4,2))
    plt.plot(profile, label='Smoothed Profile')
    plt.plot(peaks, profile[peaks], 'rx', label='Detected RBCs')
    plt.xlabel('Row Index', fontproperties=source_sans)
    plt.ylabel('Intensity', fontproperties=source_sans)
    plt.title(f'Vertical Intensity Profile After Rotation (Detected: {rbc_count}, Estimated: {rbc_count_est})', fontproperties=source_sans)
    plt.legend(prop=source_sans)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Step 8: Overlay Detected RBCs on Kymograph
    # -----------------------------
    overlay_x_values_on_kymograph(rotated, peaks)
    
    return rbc_count

def run_gui(kymograph_dir, counts_df_path):
    """Run the GUI application"""
    app = RBCCounterGUI(kymograph_dir, counts_df_path)
    app.root.mainloop()

if __name__ == '__main__':
    # For backward compatibility
    # counts_df = pd.read_csv('D:\\frog\\counted_kymos_CalFrog4.csv')
    # for path in glob.glob('D:\\frog\\kymographs\\*.tiff'):
    #     main(path, counts_df)
    
    # Run the GUI instead
    run_gui('D:\\frog\\kymographs', 'D:\\frog\\counted_kymos_CalFrog4.csv')