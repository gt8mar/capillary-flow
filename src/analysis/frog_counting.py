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
        plt.axhline(y=x, color=color, linewidth=linewidth, linestyle='--', alpha=0.5)
    
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
    counts_row = counts_df[counts_df['Filename'] == os.path.basename(path_to_kymograph)]
    image_name = os.path.basename(path_to_kymograph)
    rbc_velocity_um_s = counts_row['Velocity (um/s)'].values[0]
    rbc_count_est = counts_row['Estimated_Counts'].values[0]

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

    # Calculate a dynamic height threshold based on the intensity distribution
    # This will disallow peaks that are too close to the background level
    background_level = np.percentile(inverted, 25)  # Approximate background level (25th percentile)
    peak_level = np.percentile(inverted, 95)       # Approximate peak level (95th percentile)
    
    # Set minimum height as a percentage above the background level
    # Adjust this percentage (0.2 = 20%) based on your specific data characteristics
    height_threshold = background_level + 0.2 * (peak_level - background_level)
    
    # Initial peak detection with adaptive parameters and height threshold
    peaks, properties = find_peaks(inverted, 
                                  prominence=prominence_threshold,
                                  distance=max(0.5, expected_distance * 0.7),  # Minimum expected spacing
                                  height=height_threshold)  # Dynamic height threshold
    
    initial_count = len(peaks)
    
    # Final RBC count
    rbc_count = len(peaks)

    return rbc_count, rotated, profile, peaks, inverted, rbc_count_est, height_threshold  # Return height threshold for debugging

class RBCCounterGUI:
    def __init__(self, kymograph_dir, counts_df_path):
        self.kymograph_dir = kymograph_dir
        
        # Check if output file already exists
        # Get base filename without extension
        base_name = os.path.basename(counts_df_path)
        dir_name = os.path.dirname(counts_df_path)
        name_without_ext, extension = os.path.splitext(base_name)
        
        # Create output filename in the same folder but with 'final_' prefix
        output_filename = os.path.join(dir_name, f"final_{name_without_ext}{extension}")
        
        self.output_path = output_filename
        
        # If output file exists, load it; otherwise, create from the counts file
        if os.path.exists(self.output_path):
            self.output_df = pd.read_csv(self.output_path)
            self.counts_df = pd.read_csv(counts_df_path)
        else:
            self.counts_df = pd.read_csv(counts_df_path)
            self.output_df = self.counts_df.copy()
            # Initialize additional columns if they don't exist
            if 'Measured_Counts' not in self.output_df.columns:
                self.output_df['Measured_Counts'] = None
            if 'Final_Counts' not in self.output_df.columns:
                self.output_df['Final_Counts'] = None
            if 'Adjustment_Type' not in self.output_df.columns:
                self.output_df['Adjustment_Type'] = None
            if 'Classified_Velocity' not in self.output_df.columns:
                self.output_df['Classified_Velocity'] = None
            if 'Modified_Estimated_Counts' not in self.output_df.columns:
                self.output_df['Modified_Estimated_Counts'] = None
        
        # Get all kymograph files
        self.kymograph_files = self.get_kymograph_files()
        
        # Find the first kymograph that doesn't have a final count
        self.current_index = self.find_first_unprocessed_kymograph()
        
        self.manual_count_mode = False
        self.manual_count_value = ""
        self.prominence_threshold = 0.05
        self.height_threshold = None  # Store current height threshold
        
        # Add velocity adjustment mode variables
        self.velocity_adjustment_mode = False
        self.high_velocities = [10, 420, 500, 600, 750, 1000, 1500, 2000, 3000, 4000]
        self.additional_velocities = [10, 20, 35, 50, 75, 110, 160, 220, 290, 360]
        self.use_additional_velocities = False
        self.current_velocity_index = 0
        self.original_velocity = None
        self.inverted_velocity = False
        
        # Add estimated counts adjustment mode variables
        self.estimated_counts_mode = False
        self.estimated_count_values = [5, 10, 20, 30, 40, 50, 75, 100, 150, 200]
        self.current_est_count_index = 0
        self.original_est_count = None
        
        self.setup_gui()
        self.analyze_current_kymograph()
    
    def find_first_unprocessed_kymograph(self):
        """Find the index of the first kymograph without a final count"""
        for i, file_path in enumerate(self.kymograph_files):
            filename = os.path.basename(file_path)
            # Check if this file exists in the output_df and has no Final_Counts
            if filename in self.output_df['Filename'].values:
                idx = self.output_df[self.output_df['Filename'] == filename].index[0]
                if pd.isna(self.output_df.at[idx, 'Adjustment_Type']):
                    return i
        
        # If all kymographs have been processed, start at the beginning
        return 0
    
    def get_kymograph_files(self):
        """Get list of kymograph files that match entries in the counts dataframe"""
        all_files = glob.glob(os.path.join(self.kymograph_dir, '*.tiff'))
        all_files.extend(glob.glob(os.path.join(self.kymograph_dir, '*.tif')))
        
        # Filter to only include files in the dataframe
        valid_files = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            if filename in self.counts_df['Filename'].values:
                valid_files.append(file_path)
        
        return valid_files
    
    def setup_gui(self):
        """Set up the GUI interface"""
        self.root = tk.Tk()
        self.root.title("RBC Counter GUI")
        self.root.geometry("1200x800")
        
        # Add mode indicator at the top
        self.mode_frame = tk.Frame(self.root, bg="#e0e0e0")
        self.mode_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.mode_label = tk.Label(self.mode_frame, text="COUNTING MODE", font=("Arial", 14, "bold"), bg="#e0e0e0")
        self.mode_label.pack(pady=5)
        
        # Create frames with improved layout
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create left and right info columns
        self.info_left_frame = tk.Frame(self.info_frame)
        self.info_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.info_right_frame = tk.Frame(self.info_frame)
        self.info_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Left column info labels
        self.file_label = tk.Label(self.info_left_frame, text="File: ", font=("Arial", 12))
        self.file_label.pack(anchor=tk.W, pady=2)
        
        self.count_label = tk.Label(self.info_left_frame, text="Counts: ", font=("Arial", 12))
        self.count_label.pack(anchor=tk.W, pady=2)
        
        self.threshold_label = tk.Label(self.info_left_frame, text="Thresholds: ", font=("Arial", 12))
        self.threshold_label.pack(anchor=tk.W, pady=2)
        
        # Right column info labels
        self.velocity_label = tk.Label(self.info_right_frame, text="Velocity: ", font=("Arial", 12))
        self.velocity_label.pack(anchor=tk.W, pady=2)
        
        self.original_est_count_label = tk.Label(self.info_right_frame, text="Original Est. Count: ", font=("Arial", 12))
        self.original_est_count_label.pack(anchor=tk.W, pady=2)
        
        self.est_count_label = tk.Label(self.info_right_frame, text="New Est. Count: ", font=("Arial", 12))
        self.est_count_label.pack(anchor=tk.W, pady=2)
        
        # Manual count entry in right frame
        self.manual_entry_frame = tk.Frame(self.info_right_frame)
        self.manual_entry_frame.pack(anchor=tk.W, pady=2, fill=tk.X)
        
        self.manual_entry_label = tk.Label(self.manual_entry_frame, text="Manual count: ", font=("Arial", 12))
        self.manual_entry_label.pack(side=tk.LEFT)
        
        self.manual_entry_value = tk.Label(self.manual_entry_frame, text="", font=("Arial", 12))
        self.manual_entry_value.pack(side=tk.LEFT)
        
        # Add a Help button to open instructions window
        self.help_button = tk.Button(self.info_right_frame, text="Show Instructions", 
                                   command=self.show_instructions_window)
        self.help_button.pack(anchor=tk.E, pady=10)
        
        # Status label (not visible in GUI but kept for function calls)
        self.status_label = tk.Label(self.root)
        
        # Create matplotlib figures with improved sizing
        self.setup_plots()
        
        # Key bindings for counting mode
        self.root.bind('c', self.mark_as_correct)
        self.root.bind('n', self.next_kymograph)
        self.root.bind('b', self.previous_kymograph)
        self.root.bind('m', self.mark_as_manual)  # Now behaves like 'f'
        self.root.bind('f', self.mark_as_too_few)
        self.root.bind('r', self.refresh_analysis)  # Added refresh key
        
        # Key bindings for velocity adjustment mode
        self.root.bind('v', self.toggle_velocity_mode)
        self.root.bind('p', self.toggle_velocity_sign)
        self.root.bind('z', self.handle_velocity_key)
        self.root.bind('s', self.handle_velocity_key)
        self.root.bind('<Shift_L>', self.toggle_velocity_set)
        self.root.bind('<Shift_R>', self.toggle_velocity_set)
        
        # Key binding for estimated counts mode
        self.root.bind('e', self.toggle_estimated_counts_mode)
        
        # Number key bindings for manual count, velocity selection, or estimated count selection
        for i in range(10):
            self.root.bind(str(i), self.handle_number_key)
        
        # Backspace for manual count
        self.root.bind('<BackSpace>', self.remove_digit)
        
        # Create instructions but don't add them to the main window
        self.instructions_text = """
        Counting Mode:
        c - Mark as correct
        n - Next kymograph
        b - Previous kymograph
        m - Too many (enter manual count)
        f - Too few (enter manual count)
        r - Refresh analysis with current settings
        0-9 - Enter digits for manual count
        Backspace - Remove last digit
        v - Toggle velocity adjustment mode
        e - Toggle estimated counts mode
        
        Velocity Mode (press v to toggle):
        z - Set velocity to zero
        f - Too fast (adjust velocity)
        s - Too slow (adjust velocity)
        v - Return to counting mode
        p - Toggle velocity sign
        0-9 - Select velocity
        Shift - Toggle velocity sets
        r - Refresh analysis
        
        Estimated Counts Mode (press e to toggle):
        0-9 - Select estimated counts value
        e - Return to counting mode
        r - Refresh analysis
        """
        
        # Initial instructions window state
        self.instructions_window = None
    
    def show_instructions_window(self):
        """Opens a new window with the instructions"""
        # Close existing window if open
        if self.instructions_window is not None and self.instructions_window.winfo_exists():
            self.instructions_window.focus_force()  # Just bring to front if already open
            return
        
        # Create new window
        self.instructions_window = tk.Toplevel(self.root)
        self.instructions_window.title("RBC Counter Instructions")
        self.instructions_window.geometry("500x500")
        
        # Add instructions text
        instructions_label = tk.Label(
            self.instructions_window, 
            text=self.instructions_text, 
            font=("Arial", 12),
            justify=tk.LEFT,
            padx=20, 
            pady=20
        )
        instructions_label.pack(fill=tk.BOTH, expand=True)
        
        # Add a close button
        close_button = tk.Button(
            self.instructions_window, 
            text="Close", 
            command=self.instructions_window.destroy,
            font=("Arial", 12)
        )
        close_button.pack(pady=10)
    
    def setup_plots(self):
        """Set up the matplotlib plots with improved sizing"""
        # Create figure for profile plot - taller than before
        self.profile_fig = mpl_fig.Figure(figsize=(8, 6), dpi=100)
        self.profile_ax = self.profile_fig.add_subplot(111)
        self.profile_canvas = FigureCanvasTkAgg(self.profile_fig, master=self.plot_frame)
        self.profile_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create figure for kymograph - taller than before
        self.kymo_fig = mpl_fig.Figure(figsize=(8, 6), dpi=100)
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
        
        # Get current velocity from dataframe or use previously set velocity
        if 'Classified_Velocity' in self.output_df.columns and not pd.isna(self.output_df.loc[self.output_df['Filename'] == filename, 'Classified_Velocity'].values[0]):
            current_velocity = self.output_df.loc[self.output_df['Filename'] == filename, 'Classified_Velocity'].values[0]
        else:
            current_velocity = self.counts_df.loc[self.counts_df['Filename'] == filename, 'Velocity (um/s)'].values[0]
        
        self.original_velocity = current_velocity
        
        # Store the original estimated count from the input file
        original_est_count = self.counts_df.loc[self.counts_df['Filename'] == filename, 'Estimated_Counts'].values[0]
        
        # Get current estimated counts from dataframe or use previously set value
        if 'Modified_Estimated_Counts' in self.output_df.columns and not pd.isna(self.output_df.loc[self.output_df['Filename'] == filename, 'Modified_Estimated_Counts'].values[0]):
            current_est_count = self.output_df.loc[self.output_df['Filename'] == filename, 'Modified_Estimated_Counts'].values[0]
        else:
            current_est_count = original_est_count
        
        self.original_est_count = original_est_count
        
        # Update velocity label
        self.velocity_label.config(text=f"Velocity: {current_velocity} um/s")
        
        # Update estimated count labels - now showing both original and new
        self.original_est_count_label.config(text=f"Original Est. Count: {original_est_count}")
        self.est_count_label.config(text=f"New Est. Count: {current_est_count}")
        
        # Create a temporary dataframe for analysis
        temp_df = self.counts_df.copy()
        idx = temp_df[temp_df['Filename'] == filename].index[0]
        temp_df.at[idx, 'Velocity (um/s)'] = current_velocity
        temp_df.at[idx, 'Estimated_Counts'] = current_est_count
        
        # Analyze the kymograph with potentially modified values
        rbc_count, rotated, profile, peaks, inverted, rbc_count_est, height_threshold = analyze_kymograph(
            current_file, temp_df, self.prominence_threshold
        )
        
        self.current_rotated = rotated
        self.current_profile = profile
        self.current_peaks = peaks
        self.current_inverted = inverted
        self.current_count = rbc_count
        self.current_est_count = rbc_count_est
        self.height_threshold = height_threshold
        
        # Update count label
        self.count_label.config(text=f"Estimated: {rbc_count_est}, Measured: {rbc_count}")
        
        # Update threshold label
        self.threshold_label.config(text=f"Prominence: {self.prominence_threshold:.3f}, Height: {self.height_threshold:.3f}")
        
        # Update plots
        self.update_plots()
        
        # Update output dataframe
        idx = self.counts_df[self.counts_df['Filename'] == filename].index[0]
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
        self.profile_ax.plot(-self.current_profile, label='Intensity Profile')
        self.profile_ax.plot(self.current_peaks, -self.current_profile[self.current_peaks], 'rx', label='Detected RBCs')
        
        # Add height threshold line to the plot
        if self.height_threshold is not None:
            self.profile_ax.axhline(y=self.height_threshold, color='g', linestyle='--', 
                                   label=f'Height Threshold ({self.height_threshold:.3f})')
        
        self.profile_ax.set_title(f'Vertical Intensity Profile After Rotation (Detected: {self.current_count}, Estimated: {self.current_est_count})')
        self.profile_ax.set_xlabel('Row Index')
        self.profile_ax.set_ylabel('Intensity')
        self.profile_ax.legend()
        
        # Plot kymograph with overlaid peaks
        self.kymo_ax.imshow(self.current_rotated, cmap='gray', aspect='auto')
        for x in self.current_peaks:
            self.kymo_ax.axhline(y=x, color='r', linewidth=1, linestyle='--', alpha=0.5)
        self.kymo_ax.set_title("Rotated Kymograph with Detected RBC Positions")
        self.kymo_ax.set_xlabel("X Position (pixels)")
        self.kymo_ax.set_ylabel("Y Position (pixels)")
        
        # Draw the canvases
        self.profile_fig.tight_layout()
        self.kymo_fig.tight_layout()
        self.profile_canvas.draw()
        self.kymo_canvas.draw()
    
    def mark_as_correct(self, event=None):
        """Mark the current kymograph count as correct or confirm velocity in velocity mode"""
        if self.velocity_adjustment_mode:
            # We no longer use this path - removed functionality that was previously here
            # Just toggle back to counting mode
            self.velocity_adjustment_mode = False
            self.status_label.config(text="Counting Mode - Use c/m/f keys")
            self.mode_label.config(text="COUNTING MODE", bg="#e0e0e0")  # Default gray background
        elif self.estimated_counts_mode:
            # Toggle back to counting mode
            self.estimated_counts_mode = False
            self.status_label.config(text="Counting Mode - Use c/m/f keys")
            self.mode_label.config(text="COUNTING MODE", bg="#e0e0e0")  # Default gray background
        elif self.manual_count_mode:
            # If in manual count mode, use the manual count value
            if self.manual_count_value:
                try:
                    manual_count = int(self.manual_count_value)
                    filename = os.path.basename(self.kymograph_files[self.current_index])
                    idx = self.counts_df[self.counts_df['Filename'] == filename].index[0]
                    self.output_df.at[idx, 'Final_Counts'] = manual_count
                    self.output_df.at[idx, 'Adjustment_Type'] = 'Manual'
                    self.status_label.config(text=f"Saved manual count: {manual_count}")
                    
                    # Reset manual count mode
                    self.manual_count_mode = False
                    self.manual_count_value = ""
                    self.manual_entry_value.config(text="")
                    self.mode_label.config(text="COUNTING MODE", bg="#e0e0e0")  # Default gray background
                    
                    # Save the output dataframe
                    self.output_df.to_csv(self.output_path, index=False)
                    
                    # Move to next kymograph
                    self.next_kymograph()
                except ValueError:
                    self.status_label.config(text="Invalid manual count value")
            else:
                self.status_label.config(text="Please enter a manual count value")
        else:
            # Mark the current count as correct
            filename = os.path.basename(self.kymograph_files[self.current_index])
            idx = self.counts_df[self.counts_df['Filename'] == filename].index[0]
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
    
    def mark_as_too_few(self, event=None):
        """Enter manual count mode for too few RBCs"""
        if not self.manual_count_mode:
            self.manual_count_mode = True
            self.manual_count_value = ""
            self.manual_entry_value.config(text="")
            self.status_label.config(text="Enter manual count and press 'c' to confirm")
            self.mode_label.config(text="MANUAL COUNT MODE", bg="#e0e0ff")  # Light blue background
    
    def mark_as_manual(self, event=None):
        """Enter manual count mode for too many RBCs - behaves like too_few"""
        self.mark_as_too_few(event)
        self.status_label.config(text="Enter manual count and press 'c' to confirm (too many detected)")
    
    def handle_number_key(self, event=None):
        """Handle number key press in manual count, velocity, or estimated counts mode"""
        if self.manual_count_mode:
            self.add_digit(event)
        elif self.velocity_adjustment_mode:
            self.set_velocity(event)
        elif self.estimated_counts_mode:
            self.set_estimated_count(event)
    
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
    
    def toggle_velocity_mode(self, event=None):
        """Toggle between velocity adjustment mode and counting mode"""
        # Exit estimated counts mode if active
        if self.estimated_counts_mode:
            self.estimated_counts_mode = False
            
        self.velocity_adjustment_mode = not self.velocity_adjustment_mode
        self.manual_count_mode = False  # Exit manual count mode if active
        self.manual_count_value = ""
        self.manual_entry_value.config(text="")
        
        if self.velocity_adjustment_mode:
            self.status_label.config(text="Velocity Adjustment Mode - Use z/f/s/v keys")
            self.mode_label.config(text="VELOCITY ADJUSTMENT MODE", bg="#ffe0e0")  # Light red background
            # Initialize velocity values
            filename = os.path.basename(self.kymograph_files[self.current_index])
            if 'Classified_Velocity' in self.output_df.columns and not pd.isna(self.output_df.loc[self.output_df['Filename'] == filename, 'Classified_Velocity'].values[0]):
                self.original_velocity = self.output_df.loc[self.output_df['Filename'] == filename, 'Classified_Velocity'].values[0]
            else:
                self.original_velocity = self.counts_df.loc[self.counts_df['Filename'] == filename, 'Velocity (um/s)'].values[0]
            
            self.current_velocity_index = 0
            self.inverted_velocity = self.original_velocity < 0
        else:
            self.status_label.config(text="Counting Mode - Use c/m/f keys")
            self.mode_label.config(text="COUNTING MODE", bg="#e0e0e0")  # Default gray background
            
    def toggle_velocity_sign(self, event=None):
        """Toggle velocity sign while in velocity adjustment mode"""
        if self.velocity_adjustment_mode:
            self.inverted_velocity = not self.inverted_velocity
            
            # Get current velocity value
            if self.current_velocity_index == 0:
                current_velocity = self.original_velocity
            else:
                velocities = self.additional_velocities if self.use_additional_velocities else self.high_velocities
                current_velocity = velocities[self.current_velocity_index]
            
            # Invert sign
            if self.inverted_velocity:
                current_velocity = -abs(current_velocity)
            else:
                current_velocity = abs(current_velocity)
            
            # Update status
            self.status_label.config(text=f"Toggled velocity sign: {current_velocity} um/s")
            
            # Update the classified velocity
            filename = os.path.basename(self.kymograph_files[self.current_index])
            idx = self.counts_df[self.counts_df['Filename'] == filename].index[0]
            self.output_df.at[idx, 'Classified_Velocity'] = current_velocity
            
            # Update velocity label
            self.velocity_label.config(text=f"Velocity: {current_velocity} um/s")
            
            # Reanalyze kymograph with new velocity
            self.reanalyze_with_new_velocity(current_velocity)
            
    def toggle_estimated_counts_mode(self, event=None):
        """Toggle between estimated counts adjustment mode and counting mode"""
        # Exit velocity mode if active
        if self.velocity_adjustment_mode:
            self.velocity_adjustment_mode = False
            
        self.estimated_counts_mode = not self.estimated_counts_mode
        self.manual_count_mode = False  # Exit manual count mode if active
        self.manual_count_value = ""
        self.manual_entry_value.config(text="")
        
        if self.estimated_counts_mode:
            self.status_label.config(text="Estimated Counts Mode - Use number keys to select count, e to return")
            self.mode_label.config(text="ESTIMATED COUNTS MODE", bg="#e0ffe0")  # Light green background
            # Initialize estimated count values
            filename = os.path.basename(self.kymograph_files[self.current_index])
            if 'Modified_Estimated_Counts' in self.output_df.columns and not pd.isna(self.output_df.loc[self.output_df['Filename'] == filename, 'Modified_Estimated_Counts'].values[0]):
                self.original_est_count = self.output_df.loc[self.output_df['Filename'] == filename, 'Modified_Estimated_Counts'].values[0]
            else:
                self.original_est_count = self.counts_df.loc[self.counts_df['Filename'] == filename, 'Estimated_Counts'].values[0]
            
            self.current_est_count_index = 0
        else:
            self.status_label.config(text="Counting Mode - Use c/m/f keys")
            self.mode_label.config(text="COUNTING MODE", bg="#e0e0e0")  # Default gray background
    
    def set_estimated_count(self, event=None):
        """Set estimated count based on number key press"""
        if self.estimated_counts_mode and event and event.char.isdigit():
            idx = int(event.char)
            if idx == 0:
                # Use original estimated count from input file
                current_est_count = self.counts_df.loc[self.counts_df['Filename'] == os.path.basename(self.kymograph_files[self.current_index]), 'Estimated_Counts'].values[0]
                self.current_est_count_index = 0
            else:
                # Use selected estimated count
                self.current_est_count_index = idx
                current_est_count = self.estimated_count_values[idx]
            
            # Update the modified estimated count in output dataframe
            filename = os.path.basename(self.kymograph_files[self.current_index])
            df_idx = self.counts_df[self.counts_df['Filename'] == filename].index[0]
            self.output_df.at[df_idx, 'Modified_Estimated_Counts'] = current_est_count
            
            # Update estimated count label (for new est count only)
            self.est_count_label.config(text=f"New Est. Count: {current_est_count}")
            self.status_label.config(text=f"Estimated count set to {current_est_count} - press e to return to counting mode")
            
            # Reanalyze kymograph with new estimated count and save changes
            self.reanalyze_with_new_estimated_count(current_est_count)
            self.output_df.to_csv(self.output_path, index=False)
            
    def reanalyze_with_new_estimated_count(self, new_est_count):
        """Reanalyze kymograph with the new estimated count"""
        # Temporarily update the estimated count in the dataframe
        filename = os.path.basename(self.kymograph_files[self.current_index])
        idx = self.counts_df[self.counts_df['Filename'] == filename].index[0]
        old_est_count = self.counts_df.at[idx, 'Estimated_Counts']
        self.counts_df.at[idx, 'Estimated_Counts'] = new_est_count
        
        # Get current velocity (use classified if available)
        if 'Classified_Velocity' in self.output_df.columns and not pd.isna(self.output_df.loc[self.output_df['Filename'] == filename, 'Classified_Velocity'].values[0]):
            current_velocity = self.output_df.loc[self.output_df['Filename'] == filename, 'Classified_Velocity'].values[0]
            # Also update velocity temporarily
            old_velocity = self.counts_df.at[idx, 'Velocity (um/s)']
            self.counts_df.at[idx, 'Velocity (um/s)'] = current_velocity
        else:
            current_velocity = None
            old_velocity = None
        
        # Reanalyze
        try:
            rbc_count, rotated, profile, peaks, inverted, rbc_count_est, height_threshold = analyze_kymograph(
                self.kymograph_files[self.current_index], self.counts_df, self.prominence_threshold
            )
            
            # Update stored values
            self.current_rotated = rotated
            self.current_profile = profile
            self.current_peaks = peaks
            self.current_inverted = inverted
            self.current_count = rbc_count
            self.current_est_count = rbc_count_est
            self.height_threshold = height_threshold
            
            # Update count label
            self.count_label.config(text=f"Estimated: {rbc_count_est}, Measured: {rbc_count}")
            
            # Update plots
            self.update_plots()
            
            # Update output df
            self.output_df.at[idx, 'Measured_Counts'] = rbc_count
            if pd.isna(self.output_df.at[idx, 'Final_Counts']):
                self.output_df.at[idx, 'Final_Counts'] = rbc_count
        finally:
            # Restore original estimated count in counts_df (we keep the new one in output_df)
            self.counts_df.at[idx, 'Estimated_Counts'] = old_est_count
            # Restore original velocity if it was changed
            if old_velocity is not None:
                self.counts_df.at[idx, 'Velocity (um/s)'] = old_velocity
    
    def handle_velocity_key(self, event):
        """Handle velocity-related key presses in velocity adjustment mode"""
        if not self.velocity_adjustment_mode:
            return
            
        key = event.char.lower()
        filename = os.path.basename(self.kymograph_files[self.current_index])
        idx = self.counts_df[self.counts_df['Filename'] == filename].index[0]
        
        if key == 'z':  # Zero velocity
            self.output_df.at[idx, 'Classified_Velocity'] = 0
            self.status_label.config(text="Velocity set to zero")
            self.velocity_label.config(text="Velocity: 0 um/s")
            self.reanalyze_with_new_velocity(0)
            
            # Save the output dataframe
            self.output_df.to_csv(self.output_path, index=False)
            
        elif key == 'f':  # Too fast
            self.status_label.config(text="Too fast - Select velocity using number keys")
            
        elif key == 's':  # Too slow
            self.status_label.config(text="Too slow - Select velocity using number keys")
    
    def set_velocity(self, event=None):
        """Set velocity based on number key press"""
        if self.velocity_adjustment_mode and event and event.char.isdigit():
            idx = int(event.char)
            if idx == 0:
                # Use original velocity
                current_velocity = self.original_velocity
                self.current_velocity_index = 0
            else:
                # Use selected velocity
                self.current_velocity_index = idx
                velocities = self.additional_velocities if self.use_additional_velocities else self.high_velocities
                current_velocity = velocities[idx]
                
                if self.inverted_velocity:
                    current_velocity = -current_velocity
            
            # Update the classified velocity in output dataframe
            filename = os.path.basename(self.kymograph_files[self.current_index])
            df_idx = self.counts_df[self.counts_df['Filename'] == filename].index[0]
            self.output_df.at[df_idx, 'Classified_Velocity'] = current_velocity
            
            # Update velocity label
            self.velocity_label.config(text=f"Velocity: {current_velocity} um/s")
            self.status_label.config(text=f"Velocity set to {current_velocity} um/s - press v to return to counting mode")
            
            # Reanalyze kymograph with new velocity and save changes
            self.reanalyze_with_new_velocity(current_velocity)
            self.output_df.to_csv(self.output_path, index=False)

    def reanalyze_with_new_velocity(self, new_velocity):
        """Reanalyze kymograph with the new velocity"""
        # Temporarily update the velocity in the dataframe
        filename = os.path.basename(self.kymograph_files[self.current_index])
        idx = self.counts_df[self.counts_df['Filename'] == filename].index[0]
        old_velocity = self.counts_df.at[idx, 'Velocity (um/s)']
        self.counts_df.at[idx, 'Velocity (um/s)'] = new_velocity
        
        # Get current estimated count if available
        if 'Modified_Estimated_Counts' in self.output_df.columns and not pd.isna(self.output_df.loc[self.output_df['Filename'] == filename, 'Modified_Estimated_Counts'].values[0]):
            current_est_count = self.output_df.loc[self.output_df['Filename'] == filename, 'Modified_Estimated_Counts'].values[0]
            # Also update estimated count temporarily
            old_est_count = self.counts_df.at[idx, 'Estimated_Counts']
            self.counts_df.at[idx, 'Estimated_Counts'] = current_est_count
        else:
            current_est_count = None
            old_est_count = None
        
        # Reanalyze
        try:
            rbc_count, rotated, profile, peaks, inverted, rbc_count_est, height_threshold = analyze_kymograph(
                self.kymograph_files[self.current_index], self.counts_df, self.prominence_threshold
            )
            
            # Update stored values
            self.current_rotated = rotated
            self.current_profile = profile
            self.current_peaks = peaks
            self.current_inverted = inverted
            self.current_count = rbc_count
            self.current_est_count = rbc_count_est
            self.height_threshold = height_threshold
            
            # Update count label
            self.count_label.config(text=f"Estimated: {rbc_count_est}, Measured: {rbc_count}")
            
            # Update threshold label
            self.threshold_label.config(text=f"Prominence: {self.prominence_threshold:.3f}, Height: {self.height_threshold:.3f}")
            
            # Update plots
            self.update_plots()
            
            # Update output df
            self.output_df.at[idx, 'Measured_Counts'] = rbc_count
            if pd.isna(self.output_df.at[idx, 'Final_Counts']):
                self.output_df.at[idx, 'Final_Counts'] = rbc_count
        finally:
            # Restore original velocity in counts_df
            self.counts_df.at[idx, 'Velocity (um/s)'] = old_velocity
            # Restore original est count if it was changed
            if old_est_count is not None:
                self.counts_df.at[idx, 'Estimated_Counts'] = old_est_count

    def toggle_velocity_set(self, event=None):
        """Toggle between high and additional velocity sets"""
        if self.velocity_adjustment_mode:
            self.use_additional_velocities = not self.use_additional_velocities
            self.status_label.config(text=f"Using {'additional' if self.use_additional_velocities else 'high'} velocities")
            
            # If a velocity is selected, update to the corresponding value in the new set
            if self.current_velocity_index > 0:
                velocities = self.additional_velocities if self.use_additional_velocities else self.high_velocities
                current_velocity = velocities[self.current_velocity_index]
                
                if self.inverted_velocity:
                    current_velocity = -current_velocity
                
                # Update the classified velocity
                filename = os.path.basename(self.kymograph_files[self.current_index])
                idx = self.counts_df[self.counts_df['Filename'] == filename].index[0]
                self.output_df.at[idx, 'Classified_Velocity'] = current_velocity
                
                # Update velocity label
                self.velocity_label.config(text=f"Velocity: {current_velocity} um/s")
                
                # Reanalyze kymograph with new velocity
                self.reanalyze_with_new_velocity(current_velocity)

    def refresh_analysis(self, event=None):
        """Refresh the analysis with current velocity and estimated count values"""
        filename = os.path.basename(self.kymograph_files[self.current_index])
        idx = self.counts_df[self.counts_df['Filename'] == filename].index[0]
        
        # Get current velocity
        if 'Classified_Velocity' in self.output_df.columns and not pd.isna(self.output_df.loc[self.output_df['Filename'] == filename, 'Classified_Velocity'].values[0]):
            current_velocity = self.output_df.loc[self.output_df['Filename'] == filename, 'Classified_Velocity'].values[0]
        else:
            current_velocity = self.counts_df.loc[self.counts_df['Filename'] == filename, 'Velocity (um/s)'].values[0]
        
        # Get current estimated count
        if 'Modified_Estimated_Counts' in self.output_df.columns and not pd.isna(self.output_df.loc[self.output_df['Filename'] == filename, 'Modified_Estimated_Counts'].values[0]):
            current_est_count = self.output_df.loc[self.output_df['Filename'] == filename, 'Modified_Estimated_Counts'].values[0]
        else:
            current_est_count = self.counts_df.loc[self.counts_df['Filename'] == filename, 'Estimated_Counts'].values[0]
        
        # Create temporary df for analysis
        temp_df = self.counts_df.copy()
        temp_df.at[idx, 'Velocity (um/s)'] = current_velocity
        temp_df.at[idx, 'Estimated_Counts'] = current_est_count
        
        # Reanalyze with current settings
        try:
            rbc_count, rotated, profile, peaks, inverted, rbc_count_est, height_threshold = analyze_kymograph(
                self.kymograph_files[self.current_index], temp_df, self.prominence_threshold
            )
            
            # Update stored values
            self.current_rotated = rotated
            self.current_profile = profile
            self.current_peaks = peaks
            self.current_inverted = inverted
            self.current_count = rbc_count
            self.current_est_count = rbc_count_est
            self.height_threshold = height_threshold
            
            # Update count label
            self.count_label.config(text=f"Estimated: {rbc_count_est}, Measured: {rbc_count}")
            
            # Update threshold label
            self.threshold_label.config(text=f"Prominence: {self.prominence_threshold:.3f}, Height: {self.height_threshold:.3f}")
            
            # Update plots
            self.update_plots()
            
            # Update measured counts in output df
            self.output_df.at[idx, 'Measured_Counts'] = rbc_count
            
            # Status update
            self.status_label.config(text="Analysis refreshed with current settings")
            
            # Save the output dataframe
            self.output_df.to_csv(self.output_path, index=False)
            
        except Exception as e:
            self.status_label.config(text=f"Error refreshing analysis: {str(e)}")

def main(path_to_kymograph, counts_df):
    # This function is kept for backward compatibility
    rbc_count, rotated, profile, peaks, inverted, rbc_count_est, height_threshold = analyze_kymograph(path_to_kymograph, counts_df)
    
    # -----------------------------
    # Step 7: Visual Inspection
    # -----------------------------
    setup_plotting_style()

    plt.figure(figsize=(2.4,2))
    plt.plot(profile, label='Smoothed Profile')
    plt.plot(peaks, profile[peaks], 'rx', label='Detected RBCs')
    plt.axhline(y=-height_threshold, color='g', linestyle='--', label=f'Height Threshold')
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
    run_gui('H:\\240729\\Frog2\\Right\\kymographs', 'H:\\240729\\Frog2\\Right\\counts\\predictions.csv')