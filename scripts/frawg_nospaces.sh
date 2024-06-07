#!/bin/bash

# Navigate to the initial directory containing the date folders
cd /hpc/projects/capillary-flow/frog || exit

# Loop through date folders
for date_dir in */; do
    # Check for directories starting with 'Frog' within each date folder
    for frog_dir in "$date_dir"Frog*/; do
        # Ensure the directory exists before proceeding
        if [[ -d "$frog_dir" ]]; then
            # Check for 'Left' and 'Right' directories within each 'Frog' folder
            for side in Left Right; do
                if [[ -d "${frog_dir}${side}" ]]; then
                    # Go into 'Left' or 'Right' and rename subdirectories
                    (cd "${frog_dir}${side}" && for sub in *\ *; do
                        mv "$sub" "${sub// /_}"
                    done)
                fi
            done
        fi
    done
done
