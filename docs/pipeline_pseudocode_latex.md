# Pipeline Pseudocode in LaTeX Format

## Image Preprocessing

\subsection{Capillary Contrast Enhancement}

\begin{lstlisting}
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
\end{lstlisting}

\subsection{Background Generation}

\begin{lstlisting}
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
\end{lstlisting}

## Capillary Identification

\subsection{Capillary Naming}

\begin{lstlisting}
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
\end{lstlisting}

\subsection{Capillary Renaming}

\begin{lstlisting}
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
\end{lstlisting}

## Centerline and Flow Analysis

\subsection{Centerline Detection}

\begin{lstlisting}
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
\end{lstlisting}

\subsection{Kymograph Generation}

\begin{lstlisting}
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
\end{lstlisting}

\subsection{Velocity Calculation}

\begin{lstlisting}
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
\end{lstlisting}

\subsection{Velocity Validation}

\begin{lstlisting}
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
\end{lstlisting} 