# Capillary Flow Analysis Pipeline

This document outlines the complete data processing pipeline for capillary flow analysis, with each section describing a key script in the pipeline and providing placeholder for pseudocode.

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

**Purpose**: Enhances contrast in capillary images to improve visibility and subsequent segmentation.

**Pseudocode**: 
*Note: This is a simplified version of the algorithm. The actual function names are listed in the "Key functions" section below.*
```python
def capillary_contrast(input_folder, output_folder):
    # Get all images from input folder
    images = load_images(input_folder)
    
    # Use first image to calculate intensity cutoffs
    first_image = images[0]
    histogram = calculate_histogram(first_image)
    lower_cutoff, upper_cutoff = find_cutoffs(histogram, saturation_percent=0.85)
    
    # Process all images using same cutoffs
    for image in images:
        # Map pixels: values below lower_cutoff → 0, above upper_cutoff → 255
        # Scale values in between to full range (0-255)
        processed = stretch_contrast(image, lower_cutoff, upper_cutoff)
        save_image(processed, output_folder)
```

**Key functions**:
- `calculate_histogram_cutoffs`: Determines optimal intensity cutoffs for contrast enhancement
- `apply_contrast`: Applies contrast stretching using lookup tables
- `capillary_contrast`: Main function that processes all images in a directory

**Inputs**:
- Directory of stabilized capillary images

**Outputs**:
- Directory of contrast-enhanced images

---

### Background Generation

**Script**: `src/write_background_file.py`

**Purpose**: Creates a static background image by averaging or taking the median of all frames in a video.

**Pseudocode**:
*Note: This is a simplified version of the algorithm. The actual function names are listed in the "Key functions" section below.*
```python
def main(path, method="mean"):   
    # Get cropping boundaries from stabilization data
    shifts = read_shifts(path)
    crop_bounds = get_crop_bounds(shifts)
    
    # Load and crop all images
    images = load_and_crop_images(input_folder, crop_bounds)
    
    # Generate background image
    if method == "mean":
        background = average_all_frames(images)
    else:  # method == "median"
        background = median_all_frames(images)
    
    # Calculate pixel-wise standard deviation across frames
    stdev_image = calculate_std_deviation(images)
    
    # Save results
    save_images(background, stdev_image, path)
```

**Key functions**:
*Note: The actual source file primarily uses a single main function with inline processing rather than separate helper functions.*
- `main`: Loads frames, calculates background, and generates standard deviation images
- `parse_path`: Extracts metadata from file path
- `get_images`: Gets sorted list of image files from a directory
- `pic2vid`: Creates a video from image frames (optional)

**Inputs**:
- Directory of stabilized video frames

**Outputs**:
- Background image (TIFF)
- Standard deviation image (TIFF)

## Capillary Identification

### Capillary Naming

**Script**: `scripts/cap_name_pipeline2.py` & `src/name_capillaries.py`

**Purpose**: Identifies individual capillaries in segmented images and assigns initial identifiers.

**Pseudocode**:
*Note: This is a simplified version of the algorithm. The actual function names are listed in the "Key functions" section below.*
```python
def main(location_path):
    # Get segmented images and backgrounds
    segmented_images = get_segmented_images(location_path)
    
    # Process each segmented image
    for image in segmented_images:
        # Parse filename to get metadata
        participant, date, location, video = parse_filename(image)
        
        # Load segmented image and background
        segmented = load_image(image)
        background = load_image(get_background_filename(image))
        
        # Uncrop the segmented image using motion correction shifts
        uncropped_segmented = uncrop_segmented(video_path, segmented)
        
        # Identify individual capillaries using connected component analysis
        capillary_masks = create_capillary_masks(uncropped_segmented)
        
        # Process each identified capillary
        for i, mask in enumerate(capillary_masks):
            # Assign initial name (sequential number)
            capillary_name = format_number_as_string(i)
            
            # Save individual capillary mask
            save_capillary_mask(mask, image, capillary_name)
            
            # Record capillary name in dataframe
            add_to_dataframe(capillary_names, image, capillary_name)
            
        # Create overlay visualization with capillary labels
        create_visual_overlay(background, capillary_masks, capillary_names)
    
    # Save capillary naming information to CSV
    save_csv(capillary_names, location_path)
```

**Key functions**:
- `uncrop_segmented`: Reverses cropping from motion correction
- `create_capillary_masks`: Isolates individual capillaries using connected component analysis
- `create_overlay_with_label`: Creates visualizations with color-coded labels
- `main`: Orchestrates the naming process
- `parse_filename`: Extracts metadata from image filenames

**Inputs**:
- Segmented images (hasty.ai output)
- Background images

**Outputs**:
- Individual capillary masks
- Naming CSV files
- Overlay visualizations

---

### Capillary Renaming

**Script**: `scripts/cap_rename_pipeline2.py` & `src/rename_capillaries.py`

**Purpose**: Applies consistent naming to capillaries across different videos after manual review.

**Pseudocode**:
*Note: This is a simplified version of the algorithm. The actual function names are listed in the "Key functions" section below.*
```python
def main(location_path):
    # Get the manually edited CSV file with capillary names
    naming_csv = load_naming_csv(location_path)
    
    # Process each row in the naming CSV
    for row in naming_csv:
        # Get original capillary filename and new name
        original_filename = row['File Name']
        new_name = row['Capillary Name']
            
        # Load original mask
        mask = load_mask(original_masks_path, original_filename)
        
        # Generate new filename with proper capillary name
        new_filename = generate_renamed_filename(original_filename, new_name)
        
        # Save mask with new filename
        save_renamed_mask(mask, new_filename)
    
    # Create overlay visualizations with the renamed capillaries
    create_renamed_overlays(location_path, naming_csv)
```

**Key functions**:
- `rename_capillaries`: Reads CSV files and creates renamed copies of capillary masks
- `create_renamed_overlays`: Generates updated visualizations with the assigned names
- `create_overlay_with_label`: Creates transparent overlays with text labels

**Inputs**:
- Original capillary masks
- Manually updated naming CSV files
- Background images

**Outputs**:
- Renamed capillary masks
- Updated overlay visualizations

## Centerline and Flow Analysis

### Centerline Detection

**Script**: `src/find_centerline.py`

**Purpose**: Identifies the centerlines (skeletons) of segmented capillaries and calculates their radii.

**Pseudocode**:
*Note: This is a simplified version of the algorithm. The actual function names are listed in the "Key functions" section below.*
```python
def main(path):
    # Get all capillary mask files
    mask_files = get_segmented_files(path)
    
    # Process each mask
    for mask_file in mask_files:
        # Load mask and identify connected regions
        mask = load_mask(mask_file)
        capillary = find_connected_components(mask)
        
        # Generate and prune skeleton
        skeleton, pruned_skeleton, radii = make_skeletons(capillary)
        
        # Skip if capillary is too small
        if skeleton_size < MIN_CAP_LEN:
            continue
        
        # Sort skeleton points into continuous path
        sorted_points, point_order = sort_continuous(skeleton_points)
        
        # Combine coordinates with corresponding radii values
        centerline_data = combine_points_and_radii(sorted_points, radii[point_order])
        
        # Save centerline data
        save_centerline_data(centerline_data, output_path)
```

**Key functions**:
- `find_junctions`: Identifies branching points in skeletons
- `find_endpoints`: Identifies endpoints in skeletons
- `make_skeletons`: Creates and prunes skeletons from binary images
- `sort_continuous`: Orders skeleton points to form a continuous path
- `find_connected_components`: Identifies connected regions in binary masks
- `main`: Processes all capillary masks in a directory

**Inputs**:
- Individual capillary masks (renamed)

**Outputs**:
- Centerline coordinates (CSV)
- Centerline visualizations

---

### Kymograph Generation

**Script**: `src/make_kymograph.py`

**Purpose**: Creates space-time plots (kymographs) along capillary centerlines to visualize blood flow.

**Pseudocode**:
*Note: This is a simplified version of the algorithm. The actual function names are listed in the "Key functions" section below.*
```python
def main(location_path):
    # Get paths for centerline files and video frames
    centerline_files = get_centerline_files(location_path)
    video_frames_path = get_video_frames_path(location_path)
    
    # Process each centerline file
    for centerline_file in centerline_files:
        # Load centerline coordinates and radii
        centerline_data = load_centerline_data(centerline_file)
        
        # Parse filename to get metadata (participant, date, etc.)
        metadata = parse_filename(centerline_file)
        
        # Load all video frames
        frames = load_video_frames(video_frames_path, metadata)
        
        # Create a circular kernel for spatial averaging
        kernel = create_circular_kernel(radius=3)
        
        # Build kymograph by extracting intensities along centerline for each frame
        kymograph = empty_array(num_centerline_points, num_frames)
        
        # For each point on the centerline
        for i, point in enumerate(centerline_data):
            row, col, radius = point
            
            # For each frame
            for j, frame in enumerate(frames):
                # Calculate average intensity in circular region around centerline point
                avg_intensity = compute_average_surrounding_pixels(frame, row, col, kernel)
                
                # Store in kymograph array
                kymograph[i, j] = avg_intensity
        
        # Save kymograph as image
        save_kymograph(kymograph, output_path, metadata)
```

**Key functions**:
- `create_circular_kernel`: Creates kernels for spatial averaging
- `compute_average_surrounding_pixels`: Applies filtering to reduce noise
- `build_centerline_vs_time_kernal`: Extracts intensities along centerlines over time
- `main`: Processes all centerlines to create kymographs

**Inputs**:
- Centerline coordinates
- Stabilized video frames

**Outputs**:
- Kymograph images (TIFF)

---

### Velocity Calculation

**Script**: `src/analysis/make_velocities.py`

**Purpose**: Analyzes kymographs to extract blood flow velocities in capillaries.

**Pseudocode**:
*Note: This is a simplified version of the algorithm. The actual function names are listed in the "Key functions" section below.*
```python
def main(location_path):
    # Get all kymograph files
    kymograph_files = get_kymograph_files(location_path)
    
    # Create output directory for velocity data
    create_output_directory(location_path)
    
    # Process each kymograph
    for kymo_file in kymograph_files:
        # Load kymograph image
        kymograph = load_kymograph(kymo_file)
        
        # Remove horizontal banding artifacts
        processed_kymo = remove_horizontal_banding(kymograph)
        
        # Choose algorithm for line detection
        if use_hough_transform:
            # Detect flow lines using Hough transform
            slopes, intercepts, strengths = find_slopes_hough(processed_kymo)
        else:
            # Alternative: Use LASSO-based line detection
            slopes, intercepts, strengths = find_slopes(processed_kymo)
        
        # Convert slopes to velocities using pixel calibration and FPS
        velocities = convert_slopes_to_velocities(slopes, fps, pixel_size)
        
        # Create visualization of detected lines on kymograph
        visualization = create_velocity_visualization(kymograph, slopes, intercepts)
        
        # Save velocity data and visualization
        save_velocity_data(velocities, strengths, output_path)
        save_visualization(visualization, output_path)
```

**Key functions**:
- `remove_horizontal_banding`: Corrects kymograph artifacts
- `find_slopes_hough`: Uses Hough transform to detect flow lines
- `find_slopes`: Alternative method using Lasso for line detection
- `main`: Processes all kymographs to calculate velocities

**Inputs**:
- Kymograph images
- Metadata (FPS, pixel calibration)

**Outputs**:
- Velocity measurements (CSV)
- Velocity visualizations

---

### Velocity Validation

**Script**: `scripts/gui_kymos.py`

**Purpose**: Provides a GUI for manual validation and correction of automatically detected velocities.

**Pseudocode**:
*Note: This is a simplified version of the algorithm. The actual function names are listed in the "Key functions" section below.*
```python
def main():
    # Load velocity data and corresponding kymographs
    velocity_data = load_velocity_data(input_path)
    kymograph_files = get_kymograph_files(input_path)
    
    # Create GUI classifier interface
    classifier = KymographClassifier(velocity_data, kymograph_files)
    
    # For each kymograph
    for i, kymo_file in enumerate(kymograph_files):
        # Display kymograph with detected velocity lines
        display_kymograph(kymo_file, velocity_data[i])
        
        # Wait for user classification input
        classification = wait_for_user_input()
        
        # Handle different classification options
        if classification == "correct":
            # Mark velocity as valid
            mark_velocity_as_valid(velocity_data[i])
        elif classification == "unclear":
            # Mark velocity as unclear
            mark_velocity_as_unclear(velocity_data[i])
        elif classification == "wrong":
            # Allow user to correct velocity by drawing line
            corrected_velocity = get_user_drawn_line()
            update_velocity(velocity_data[i], corrected_velocity)
        elif classification == "skip":
            # Skip this kymograph
            continue
        
        # Update output data with classification
        update_output_data(velocity_data[i], classification)
    
    # Save updated velocity data with manual classifications
    save_validated_data(velocity_data, output_path)
    
    # Display classification statistics
    show_statistics(velocity_data)
```

**Key functions**:
- `KymographClassifier`: Main class that handles the validation interface
- `load_image`: Displays kymographs with velocity overlay
- `correct_classification`, `unclear_classification`, etc.: Classification methods
- `update_output_csv`: Saves validation results

**Inputs**:
- Kymograph images
- Initial velocity measurements
- Metadata (FPS)

**Outputs**:
- Validated velocity measurements (CSV)
- Classification statistics

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

5. **Statistical Analysis** (additional scripts):
   - Compute summary statistics
   - Generate visualizations and figures
   - Perform comparative analyses 