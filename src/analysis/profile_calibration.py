import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.font_manager import FontProperties
import seaborn as sns
from datetime import datetime

# Import paths from config
from src.config import PATHS, load_source_sans

# Standard plot configuration
sns.set_style("whitegrid")
source_sans = load_source_sans()

plt.rcParams.update({
    'pdf.fonttype': 42,  # For editable text in PDFs
    'ps.fonttype': 42,   # For editable text in PostScript
    'font.size': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5,
    'lines.linewidth': 0.5
})

# Define input and output paths
image_path = 'C:\\Users\\gt8mar\\Desktop\\data\\Image__2024-12-11__19-20-31.tiff'
# image_path = 'C:\\Users\\gt8mar\\Desktop\\data\\calibration\\240522\\Image__2024-05-22__09-27-19.tiff'
# image_path = 'C:\\Users\\gt8mar\\Desktop\\data\\Image__2024-12-11__19-18-24.tiff'
# image_path = 'C:\\Users\\gt8mar\\Desktop\\data\\calibration\\Image__2022-04-26__22-02-43calib.tiff'

# Create results directory if it doesn't exist
results_dir = os.path.join(PATHS['cap_flow'], 'results', 'calibration')
os.makedirs(results_dir, exist_ok=True)

# Extract filename from path for the output filename
image_filename = os.path.basename(image_path)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"profile_calibration_{timestamp}.png"
output_path = os.path.join(results_dir, output_filename)

# Load the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Calculate row and column profiles
row_avg = np.median(image, axis=1)
col_avg = np.median(image, axis=0)

# Create figure with standard dimensions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.8, 2.0))

# Row average plot
ax1.plot(range(len(row_avg)), row_avg)
ax1.set_title('Row Average', fontproperties=source_sans if source_sans else None)
ax1.set_ylabel('Pixel Intensity', fontproperties=source_sans if source_sans else None)
ax1.set_xlabel('Row Number', fontproperties=source_sans if source_sans else None)
ax1.set_ylim([200, 255])

# Column average plot
ax2.plot(range(len(col_avg)), col_avg)
ax2.set_title('Column Average', fontproperties=source_sans if source_sans else None)
ax2.set_ylabel('Pixel Intensity', fontproperties=source_sans if source_sans else None)
ax2.set_xlabel('Column Number', fontproperties=source_sans if source_sans else None)
ax2.set_ylim([200, 255])
# Apply font to tick labels if source_sans is available
if source_sans:
    for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick.set_fontproperties(source_sans)
    for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
        tick.set_fontproperties(source_sans)

fig.tight_layout()

# Save the figure
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Display the figure
plt.show()
