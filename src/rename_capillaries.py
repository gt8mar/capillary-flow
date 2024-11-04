import os
import pandas as pd
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.name_capillaries import uncrop_segmented
from src.tools.parse_filename import parse_filename

def rename_capillaries(location_path):
    """
    Reads CSV files with capillary names and copies all capillary images to a new folder,
    renaming those that have new names specified in the CSV.
    """
    csv_dir = os.path.join(location_path, 'segmented', 'hasty', 'individual_caps_original')
    source_dir = os.path.join(location_path, 'segmented', 'hasty', 'individual_caps_original')
    dest_dir = os.path.join(location_path, 'segmented', 'hasty', 'renamed_individual_caps_original')
    
    os.makedirs(dest_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_cap_names.csv')]
    
    for csv_file in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            original_filename = row['File Name']
            new_cap_name = str(row['Capillary Name']).strip()
            
            source_path = os.path.join(source_dir, original_filename)
            
            # If no new name, keep original number from filename
            if not new_cap_name or pd.isna(new_cap_name) or new_cap_name == '' or new_cap_name == 'nan':
                original_cap_num = original_filename.split('_cap_')[-1].replace('.png', '')
                new_filename = original_filename
            else:
                # Use new name with zero padding
                new_filename = original_filename.replace('.png', '')
                new_filename = new_filename[:-2] + str(int(float(new_cap_name))).zfill(2)+ '.png'
            
            dest_path = os.path.join(dest_dir, new_filename)
            
            if os.path.exists(source_path):
                try:
                    import shutil
                    shutil.copy2(source_path, dest_path)
                    print(f"Copied {original_filename} to {new_filename}")
                except Exception as e:
                    print(f"Error copying {original_filename}: {str(e)}")
            else:
                print(f"Source file not found: {original_filename}")

def create_renamed_overlays(location_path):
    """
    Creates overlays for renamed capillaries on their background images.
    """
    # Setup paths
    renamed_dir = os.path.join(location_path, 'segmented', 'hasty', 'renamed_individual_caps_original')
    background_dir = os.path.join(location_path, 'backgrounds')
    overlay_dir = os.path.join(location_path, 'segmented', 'hasty', 'renamed_overlays')
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Group files by their base name (everything before _cap_XX.png)
    from collections import defaultdict
    file_groups = defaultdict(list)
    for filename in os.listdir(renamed_dir):
        if '_cap_' in filename:
            base_name = filename.split('_cap_')[0]
            file_groups[base_name].append(filename)
    
    for base_name, cap_files in file_groups.items():
        participant, date, location, video, file_prefix = parse_filename(base_name)

        # Load background image
        background_name = f"{base_name}_background.tiff"
        background_name = background_name.replace('_seg', '')
        background_path = os.path.join(background_dir, background_name)

        if not os.path.exists(background_path):
            print(f"Background not found: {background_name}")
            continue
            
        background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
        background = uncrop_segmented(os.path.join(location_path, 'vids', f'vid{video[3:]}'), background)[0]
        background_bgr = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        result = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2BGRA)
        
        # Get number of capillaries for color map
        colors = plt.cm.get_cmap('tab20', len(cap_files))
        
        # Process each capillary
        for i, cap_file in enumerate(sorted(cap_files)):
            # Load capillary mask
            cap_path = os.path.join(renamed_dir, cap_file)
            mask = cv2.imread(cap_path, cv2.IMREAD_GRAYSCALE)
            
            # Get capillary number from filename
            cap_num = cap_file.split('_cap_')[-1].replace('.png', '')
            
            # Convert color to BGR format
            color = (np.array(colors(i)[:3]) * 255).astype(np.uint8)
            
            # Create overlay
            result = create_overlay_with_label(
                cv2.cvtColor(result, cv2.COLOR_BGRA2BGR),
                mask,
                color,
                cap_num
            )
            result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        
        # Save final overlay
        final_result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        overlay_name = f"{base_name}_renamed_overlay.png"
        cv2.imwrite(os.path.join(overlay_dir, overlay_name), final_result)
        print(f"Created overlay: {overlay_name}")

def create_overlay_with_label(frame_img, cap_mask, color, label):
    """
    Create an overlay of a capillary mask on a frame image with a label.
    """
    height, width = cap_mask.shape
    
    # Create colored mask
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    colored_mask[cap_mask > 0] = color
    
    # Create alpha channel
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[cap_mask > 0] = 128  # 50% transparency
    
    # Convert to BGRA
    frame_bgra = cv2.cvtColor(frame_img, cv2.COLOR_BGR2BGRA)
    overlay_bgra = np.dstack((colored_mask, alpha))
    
    # Blend images
    result = frame_bgra.copy()
    mask_region = (overlay_bgra[:, :, 3] > 0)
    result[mask_region] = cv2.addWeighted(
        frame_bgra[mask_region],
        0.5,
        overlay_bgra[mask_region],
        0.5,
        0
    )
    
    # Find centroid of the mask for label placement
    moments = cv2.moments(cap_mask)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = width // 2, height // 2
    
    # Add label
    cv2.putText(result, label, (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(result, label, (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
    
    return cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)