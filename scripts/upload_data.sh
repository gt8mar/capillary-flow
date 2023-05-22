#!/bin/bash

source_folder="/mnt/d/Marcus/data"   # Replace with the path to your source folder
remote_folder="marcus.forst@login-01.czbiohub.org:/hpc/mydata/marcus.forst/data"   # Replace with the remote destination

# Include specific subfolders
included_subfolders=(
  "metadata"
  "moco"
)

# Specify the range of folder numbers
start_number=9
end_number=12

# Iterate through "part" folders
# for part_folder in "$source_folder"/part*; do
for ((folder_number=start_number; folder_number<=end_number; folder_number++))
do
  part_folder="part$(printf "%02d" $folder_number)"
  echo "Processing part folder: $part_folder"
  part_path="$source_folder/$part_folder"

  if [ -d "$part_folder" ]; then
    echo "Processing part folder: $part_folder"
    # Iterate through "date" folders within each "part" folder
    for date_folder in "$part_folder"/*/; do
      echo "Processing date folder: $date_folder"
      if [ -d "$date_folder" ]; then
        # Iterate through "vid" folders within each "date" folder
        for vid_folder in "$date_folder"/vid*; do
          echo "Processing vid folder: $vid_folder"
          if [ -d "$vid_folder" ]; then
            rsync_command="rsync -avz"

            # Add include patterns for the specified subfolders
            for subfolder in "${included_subfolders[@]}"; do
              rsync_command+=" --include=/$subfolder"
            done

            # Exclude loose files
            rsync_command+=" --exclude=/*"

            # Perform the synchronization
            eval "$rsync_command" "$vid_folder" "$remote_folder"
          fi
        done
      fi
    done
  fi
done