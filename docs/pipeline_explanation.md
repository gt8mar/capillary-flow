# Capillary Flow Analysis Pipeline

This document outlines the complete data processing pipeline for capillary flow analysis, with each section describing a key script in the pipeline, its purpose, and how it contributes to the overall workflow.

## Table of Contents

1. [Image Preprocessing](#image-preprocessing)
   - [Capillary Contrast Enhancement](#capillary-contrast-enhancement)
   - [Background Generation](#background-generation)
2. [Capillary Identification](#capillary-identification)
   - [Capillary Naming](#capillary-naming)
   - [Capillary Renaming](#capillary-renaming)
3. [Centerline and Flow Analysis](#centerline-and-flow-analysis)
   - [Centerline Detection](#centerline-detection)
   - [Kymograph Generation](#kymograph-generation)
   - [Velocity Calculation](#velocity-calculation)
   - [Velocity Validation](#velocity-validation)

## Image Preprocessing

### Capillary Contrast Enhancement

**Script**: `src/capillary_contrast.py`

#### Purpose
This script automatically enhances the contrast of capillary images by applying histogram-based contrast adjustment. It processes a series of images without requiring manual checking, improving the visibility of capillaries in the microscope images.

#### Key Functions

1. **calculate_histogram_cutoffs(histogram, total_pixels, saturated_percentage)**
   - Calculates optimal cutoff values for contrast stretching based on a specified saturation percentage
   - Finds the points in the histogram that exclude the specified percentage of pixels from both ends

2. **apply_contrast(image, lower_cutoff, upper_cutoff, hist_size=256)**
   - Applies the contrast adjustment to an image using a lookup table (LUT)
   - Maps intensity values below the lower cutoff to 0 (black)
   - Maps intensity values above the upper cutoff to 255 (white)
   - Linearly scales values in between for maximum contrast

3. **capillary_contrast(input_folder, output_folder, saturated_percentage=0.85, plot=False)**
   - Main function that processes all images in the input folder
   - Uses the first image to calculate histogram cutoffs, then applies them to all images
   - Saves the processed images to the output folder
   - Optionally displays a before/after comparison plot

#### How to Use

```python
# Process all images in the specified folder
input_folder = "path/to/moco/images"
output_folder = "path/to/output/folder"
capillary_contrast(input_folder, output_folder, saturated_percentage=0.85)

# To view before/after comparison
capillary_contrast(input_folder, output_folder, plot=True)
```

#### Image Processing Approach

The contrast enhancement works by:
1. Computing the histogram of pixel intensities in the first image
2. Finding cutoff points that exclude a small percentage (default 0.85%) of the darkest and brightest pixels
3. Stretching the remaining intensity range to use the full 0-255 range
4. Applying the same transformation to all images in the sequence for consistency

#### Figure Output in Pipeline
This script is typically the first step in the capillary analysis pipeline. It enhances the visibility of capillaries in the microscope images, making subsequent segmentation and analysis more accurate. The contrast-enhanced images aren't directly used as figures in the paper but serve as improved inputs for later processing steps.

### Background Generation

**Script**: `src/write_background_file.py`

#### Purpose
This script generates background images from stabilized video sequences of capillaries. By averaging or taking the median of all frames in a video, it creates a static background image that represents the non-moving structures. This background is essential for later segmentation and analysis steps.

#### Key Functions

1. **main(path, method="mean", make_video=True, color=False, verbose=False, plot=False)**
   - Creates a background image from a series of stabilized frames
   - Can use either "mean" or "median" methods for background generation
   - Optionally produces a video of the stabilized frames
   - Generates both background and standard deviation images

#### Processing Approach

The background generation works through these steps:
1. **Input Selection**: Identifies the appropriate stabilized image folder (moco, mocoslice, or mocosplit)
2. **Frame Loading**: Reads all frames from the stabilized video
3. **Frame Cropping**: Uses shift values from stabilization metadata to crop frames to a consistent size
4. **Background Calculation**: Computes either the mean or median value for each pixel across all frames
5. **Standard Deviation Calculation**: Computes pixel-wise standard deviation across frames to identify areas with movement
6. **Output**: Saves the background image and contrast-enhanced standard deviation image

#### Example Usage

```python
# Generate background using median method
path = 'path/to/video/folder'
write_background_file.main(path, method="median", make_video=False)

# Generate background and create a video of the stabilized frames
write_background_file.main(path, method="mean", make_video=True, color=False)
```

#### Output Files

1. **Background Image**: A TIFF file representing the static background (named `prefix_video_background.tiff`)
2. **Standard Deviation Image**: A contrast-enhanced visualization of pixel variation over time (named `prefix_video_stdev.tiff`)
3. **Stabilized Video** (optional): An AVI file showing the stabilized frames

#### Figure Output in Pipeline
The background images serve as essential inputs for capillary segmentation. The standard deviation images can be used as figures in the paper to demonstrate regions of blood flow activity. Areas with high standard deviation correspond to regions with movement (active blood flow), while low standard deviation indicates static tissue.

## Capillary Identification

### Capillary Naming

**Script**: `scripts/cap_name_pipeline2.py` and `src/name_capillaries.py`

#### Purpose
This script orchestrates the capillary naming process across multiple participant datasets. It uses the `name_capillaries` module to identify and label individual capillaries in segmented images, creating a consistent naming scheme for further analysis.

#### Key Functions in `cap_name_pipeline2.py`

1. **main()**
   - Processes data for a participant specified via command-line argument
   - Finds the earliest date directory for the participant
   - Loops through all location folders
   - Runs the `name_capillaries.main()` function on each location

#### Key Functions in `src/name_capillaries.py`

1. **uncrop_segmented(video_path, input_seg_img)**
   - Reverses the cropping that was applied during motion correction
   - Uses shift data from the stabilization algorithm's CSV output
   - Pads the segmented image to match the original frame dimensions
   - Returns the uncropped image and the gap values for alignment reference

2. **create_capillary_masks(binary_mask)**
   - Takes a binary segmentation mask containing multiple capillaries
   - Uses connected component labeling to identify individual capillaries
   - Returns a list of separate binary masks, one for each capillary

3. **create_overlay_with_label(frame_img, cap_mask, color, label)**
   - Overlays a colored capillary mask onto a background image
   - Applies transparency for better visualization
   - Finds the centroid of the capillary for label placement
   - Adds a readable text label with black outline and white fill

4. **main(location_path)**
   - Loads segmented images and corresponding background images
   - Uncrops segmented images to original video dimensions
   - Creates individual masks for each capillary using connected component analysis
   - Assigns a sequential number to each capillary
   - Creates CSV files mapping original filenames to capillary identifiers
   - Saves individual capillary masks as separate files
   - Generates color-coded overlay visualizations of all capillaries

#### Processing Pipeline Detail

1. **Segmentation Processing**:
   - Loads segmented binary masks from the hasty.ai segmentation output
   - Normalizes the segmentation masks to binary values (0 or 255)
   - Uncrops the masks to match original video dimensions using shift data

2. **Capillary Isolation**:
   - The `create_capillary_masks()` function uses scikit-image's `label()` function to identify connected regions
   - Each connected region (representing a single capillary) is extracted as a separate binary mask
   - Masks are numbered sequentially (00, 01, 02, etc.) based on the order they were found

3. **Overlay Creation**:
   - Generates a color-coded visualization where each capillary is shown in a different color
   - Uses matplotlib's tab20 colormap to ensure visually distinct colors
   - Overlays the colored masks onto the background image with transparency
   - Labels each capillary with its assigned number

4. **Data Organization**:
   - Creates a CSV mapping structure with "File Name" and "Capillary Name" columns
   - Initially, "Capillary Name" is left empty for manual assignment later
   - Saves individual capillary masks in dedicated directories for further processing
   - Stores CSV files both in the local data directory and in a centralized results location

#### Example Output Files

1. **Individual Capillary Masks**: 
   - Named according to the pattern: `prefix_video_seg_cap_XX.png`
   - Stored in `segmented/hasty/individual_caps_original/`

2. **Naming CSV**:
   - Contains mapping between auto-generated filenames and capillary identifiers
   - Named according to the pattern: `participant_date_location_cap_names.csv`
   - Stored both locally and in the results directory

3. **Overlay Visualizations**:
   - Color-coded visualizations of all detected capillaries with labels
   - Named according to the pattern: `prefix_video_overlay.png`
   - Stored in `segmented/hasty/overlays/`

#### Figure Output in Pipeline
This naming process establishes a consistent identification system for tracking specific capillaries across multiple videos and conditions. The overlay images serve as visual references, enabling researchers to identify specific capillaries in the original videos and correlate them with their assigned identifiers in the dataset.

### Capillary Renaming

**Script**: `scripts/cap_rename_pipeline2.py` and `src/rename_capillaries.py`

#### Purpose
These scripts handle the renaming of capillaries after manual review and annotation. Once capillaries have been initially identified and mapped, researchers manually update the naming CSV files to assign consistent identifiers to the same capillaries across different videos. The renaming process then applies these identifiers and generates updated visualizations.

#### Key Functions in `cap_rename_pipeline2.py`

1. **main()**
   - Processes data for a participant specified via command-line argument
   - Finds the earliest date directory for the participant
   - Loops through all location folders
   - Runs the `rename_capillaries()` and `create_renamed_overlays()` functions

#### Key Functions in `src/rename_capillaries.py`

1. **rename_capillaries(location_path)**
   - Reads the manually updated CSV files with capillary name assignments
   - Iterates through each capillary entry in the CSV
   - Creates copies of capillary mask files with updated naming
   - Preserves original filenames for capillaries without manual name assignments
   - Handles edge cases like missing files and ensures zero-padding for consistent naming

2. **create_renamed_overlays(location_path)**
   - Groups renamed capillary files by their base name for processing
   - Loads the background image for each video
   - Creates a new overlay visualization with the updated capillary names
   - Uses the same color scheme and layout as the original overlays for consistency
   - Saves the updated visualizations in the renamed_overlays directory

3. **create_overlay_with_label(frame_img, cap_mask, color, label)**
   - Same function used in the initial naming process
   - Creates a transparent, colored overlay with the capillary's assigned identifier
   - Ensures visibility by using black outline and white fill for text

#### Manual Annotation Process

Between the naming and renaming steps, researchers perform a critical manual annotation:

1. Researchers examine the initial overlay visualizations
2. They identify the same capillaries across multiple videos
3. They update the CSV files, assigning consistent numbers to the same capillaries
4. For example, a capillary initially labeled "00" might be renamed to "05" to match its identifier in other videos

#### Renaming Implementation Detail

1. **CSV Processing**:
   - The renaming process reads the manually updated CSV files
   - Each row maps an original capillary filename to its new identifier
   - If a new identifier is provided, it's used to create a new filename
   - If no new identifier is provided, the original filename is preserved

2. **File Management**:
   - New copies of the capillary mask files are created with updated names
   - Original files are preserved to maintain the full processing history
   - The new files are stored in the `renamed_individual_caps_original` directory

3. **Visualization Update**:
   - New overlays are created showing the renamed capillaries
   - These use the same visual styling as the original overlays but with updated labels
   - The overlays help verify that the renaming process was performed correctly

#### Output Files

1. **Renamed Capillary Masks**:
   - Named according to the pattern: `prefix_video_seg_cap_XX.png` (where XX is the manually assigned identifier)
   - Stored in `segmented/hasty/renamed_individual_caps_original/`

2. **Renamed Overlay Visualizations**:
   - Updated color-coded visualizations showing the manually assigned capillary names
   - Named according to the pattern: `prefix_video_renamed_overlay.png`
   - Stored in `segmented/hasty/renamed_overlays/`

#### Figure Output in Pipeline
The renamed overlay visualizations serve as key reference figures for all subsequent analyses. They allow researchers to:
1. Visually verify that the same capillaries have been consistently identified across multiple videos
2. Reference specific capillaries by their assigned identifiers in discussions and publications
3. Track changes in specific capillaries under different physiological conditions
4. Ensure that comparative analyses (e.g., velocity changes) are performed on the same capillaries

These consistent identifiers are essential for the validity of longitudinal and comparative analyses across different videos and conditions.

## Centerline and Flow Analysis

### Centerline Detection

**Script**: `src/find_centerline.py`

#### Purpose
This script identifies the centerlines (skeletons) of segmented capillaries and calculates their radii. These centerlines are used for creating kymographs and analyzing blood flow velocities.

#### Key Functions

1. **find_junctions(skel)** and **find_endpoints(skel)**
   - Identify branching points and endpoints in the skeleton
   - Essential for analyzing the topology of capillary networks

2. **make_skeletons(binary_image, plot=False)**
   - Uses the FilFinder package to find and prune skeletons from binary images
   - Calculates the distance transform to determine radii along the skeleton
   - Returns the skeleton, pruned skeleton, and radii values

3. **sort_continuous(array_2D, verbose=False)**
   - Orders skeleton points to form a continuous path
   - Essential for creating ordered centerlines for kymograph generation

4. **main(path, verbose=False, write=False, plot=False, hasty=True)**
   - Main function that processes segmented capillary images
   - Finds centerlines for each capillary
   - Calculates radii along the centerlines
   - Saves centerline coordinates and radii information

#### Analysis Approach

The centerline detection works through these steps:
1. **Image Loading**: Reads segmented binary images of capillaries
2. **Component Isolation**: Separates individual capillaries using connected component analysis
3. **Skeleton Creation**: Applies medial axis transforms to find the centerline of each capillary
4. **Skeleton Pruning**: Removes branches to create a single path through each capillary
5. **Point Ordering**: Sorts centerline points to create a continuous path
6. **Radius Calculation**: Uses distance transform to calculate radius at each point

#### Example Output

![Centerline Detection Example](methods_plots/centerline_skeletons.png)

*Figure: Example output showing skeleton extraction. Left: original skeleton with possible branches; Middle: pruned skeleton showing main centerline; Right: original capillary segment.*

#### Figure Output in Pipeline
This script generates skeleton visualizations that can be included in method figures to demonstrate the centerline extraction process. The extracted centerlines and their radii are essential for subsequent analyses, including kymograph generation and diameter statistics.

### Kymograph Generation

**Script**: `src/make_kymograph.py`

#### Purpose
This script creates kymographs (space-time plots) along capillary centerlines. Kymographs visualize blood flow by tracking intensity changes along the centerline over time, allowing for velocity analysis.

#### Key Functions

1. **create_circular_kernel(radius)** and **compute_average_surrounding_pixels(image_stack, radius=4, circle=True)**
   - Create circular averaging kernels for noise reduction
   - Apply spatial averaging to improve signal quality

2. **build_centerline_vs_time_kernal(image, centerline_coords, long=True)**
   - Core function that constructs the kymograph
   - Extracts pixel intensities along the centerline for each frame
   - Applies smoothing to reduce noise

3. **main(path, write=True, variable_radii=False, verbose=False, plot=False, test=False)**
   - Orchestrates the kymograph generation process
   - Loads centerline coordinates and video frames
   - Builds kymographs for each capillary
   - Normalizes and enhances kymograph contrast
   - Saves the kymographs as TIFF images

#### Analysis Approach

The kymograph generation works through these steps:
1. **Data Loading**: Reads centerline coordinates and stabilized video frames
2. **Spatial Filtering**: Applies circular averaging around centerline points
3. **Intensity Extraction**: Records intensity values along the centerline for each frame
4. **Stacking**: Arranges intensities into a 2D image (space vs. time)
5. **Contrast Enhancement**: Applies intensity rescaling for better visualization
6. **Output**: Saves the kymograph as a TIFF image

#### Example Output

![Kymograph Example](methods_plots/kymograph_example.png)

*Figure: Example kymograph showing diagonal patterns that represent blood flow. The x-axis represents time, while the y-axis represents position along the capillary centerline. The slope of the diagonal patterns corresponds to the velocity of blood cells.*

#### Figure Output in Pipeline
Kymographs are key analytical figures that visualize blood flow. In the paper, they demonstrate the velocity patterns in capillaries under different conditions. The slope of the diagonal patterns in kymographs directly corresponds to blood cell velocity.

### Velocity Calculation

**Script**: `src/analysis/make_velocities.py`

#### Purpose
This script analyzes kymographs to extract blood flow velocities in capillaries. It uses edge detection and line-finding algorithms to identify the slope patterns in kymographs, which correspond to blood cell velocities.

#### Key Functions

1. **remove_horizontal_banding(image_path, filter_size=10)**
   - Corrects for horizontal banding artifacts in kymographs
   - Improves the accuracy of velocity detection

2. **find_slopes_hough(image, filename, min_angles=5, output_folder=None, plot=False, write=False)**
   - Uses Hough transform to detect lines in kymographs
   - Filters lines based on angle constraints
   - Calculates a weighted average slope based on line lengths

3. **find_slopes(image, filename, output_folder=None, method='lasso', verbose=False, write=False)**
   - Alternative method using the Lasso algorithm for line detection
   - Suitable for more complex kymographs

4. **main(path, verbose=False, write=False, write_data=True, test=False)**
   - Main function that processes all kymographs in a directory
   - Detects velocity for each capillary
   - Converts pixel slopes to physical velocities (μm/s)
   - Saves velocity data and generates plots

#### Velocity Calculation Approach

The velocity analysis works through these steps:
1. **Preprocessing**: Applies Gaussian filtering and removes banding artifacts
2. **Edge Detection**: Uses Canny edge detection to find patterns in kymographs
3. **Line Detection**: Applies Hough transform to identify lines representing blood flow
4. **Slope Calculation**: Computes the slope of detected lines
5. **Conversion**: Transforms pixel slopes to physical velocities using calibration factors:
   ```
   velocity (μm/s) = |slope (pixels/frame)| × FPS (frames/s) ÷ PIX_UM (pixels/μm)
   ```
6. **Visualization**: Generates plots showing the detected velocities for each capillary

#### Example Output

![Velocity Detection Example](methods_plots/velocity_detection.png)

*Figure: Example kymograph with detected flow lines overlaid (yellow). The slope of these lines is used to calculate blood flow velocity.*

#### Figure Output in Pipeline
This script generates figures showing detected velocity lines on kymographs and plots of velocities across different conditions. In the paper, these figures demonstrate how blood flow changes with pressure and other physiological factors.

### Velocity Validation

**Script**: `scripts/gui_kymos.py`

#### Purpose
This script provides a graphical user interface for manual validation of velocities from kymographs, which are time-space images used to measure blood flow velocities in capillaries. The tool allows users to visually assess the accuracy of automatically detected velocities and adjust them if necessary.

#### Key Functions

1. **KymographClassifier class**
   - Loads kymograph images and associated velocity data
   - Displays kymographs with overlaid velocity reference lines
   - Provides interface for classifying and adjusting velocities

2. **Classification Workflow**
   - Initial classification of each kymograph as "Correct," "Too Fast," "Too Slow," "Zero," or "Unclear"
   - For "Too Fast" or "Too Slow" classifications, selection of an alternative velocity from predefined sets
   - Navigation through unclassified kymographs with progress tracking

3. **Velocity Adjustment System**
   - Two velocity sets accessible via Shift key toggle:
     - High velocities (10, 420, 500, 600, 750, 1000, 1500, 2000, 3000, 4000 μm/s)
     - Additional velocities (10, 20, 35, 50, 75, 110, 160, 220, 290, 360 μm/s)
   - Dynamic overlay of velocity reference lines based on selected velocity
   - Slope inversion option for handling bidirectional flows

4. **Data Management**
   - Real-time saving of classification results to CSV
   - Loading of relevant metadata (e.g., FPS) for accurate velocity calculations
   - Tracking of classification decisions for quality control

#### How to Use

1. **Input**: 
   - Directory containing kymograph TIFF images
   - Metadata directory with acquisition parameters
   - CSV file with initial velocity measurements
   - Output path for classification results

2. **Example Usage**:
   ```python
   classifier = KymographClassifier(
       'path/to/kymographs',
       'path/to/metadata',
       'path/to/velocity_measurements.csv',
       'path/to/output_classifications.csv'
   )
   ```

3. **Keyboard Controls**:
   - `c`: Mark kymograph as correct or accept current velocity
   - `f`: Mark original velocity as too fast
   - `s`: Mark original velocity as too slow
   - `z`: Mark as zero flow
   - `u`: Mark as unclear
   - `p`: Toggle slope direction
   - `n`: Next kymograph
   - `b`: Previous kymograph
   - `Shift`: Toggle between velocity sets
   - `0-9`: Select velocity from current set

4. **Output**:
   - CSV file with original velocities, classifications, and adjusted velocities

#### Figure Output in Paper
This tool generates the dataset used for analyzing the accuracy of automated velocity measurements. The classifications and adjusted velocities are used in figures comparing automated and manual measurements, demonstrating the reliability of the velocity detection algorithms and identifying systematic biases or limitations.

![GUI Kymograph Example](methods_plots/gui_kymo_interface.png)

*Figure: Screenshot of the GUI interface showing a kymograph with velocity overlay line. The user can classify the velocity and adjust it if needed.*

## Complete Pipeline Workflow

1. **Preprocessing**:
   - Apply contrast enhancement to improve capillary visibility
   - Generate background images for segmentation

2. **Segmentation** (external step using hasty.ai):
   - Upload background images to hasty.ai
   - Create segmentation model for capillaries
   - Download segmented masks

3. **Capillary Identification**:
   - Run naming pipeline to identify individual capillaries
   - Manually review and update naming CSV files
   - Run renaming pipeline to apply consistent identifiers

4. **Flow Analysis**:
   - Detect centerlines of renamed capillaries
   - Generate kymographs along centerlines
   - Analyze kymographs to extract velocities
   - Validate velocities using GUI tool

5. **Statistical Analysis**:
   - Compute summary statistics
   - Generate visualizations and figures
   - Perform comparative analyses 