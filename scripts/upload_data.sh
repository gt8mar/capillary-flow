#!/bin/bash

source_folder="/path/to/source"   # Replace with the path to your source folder
remote_folder="marcus.forst@login-01.czbiohub.org:/hpc/mydata/marcus.forst/data"   # Replace with the remote destination

# Include specific subfolders
included_subfolders=(
  "metadata"
  "moco"
)

# Specify the range of folder numbers
start_number=1
end_number=10

for ((folder_number=start_number; folder_number<=end_number; folder_number++))
do
  source_folder="part$(printf "%02d" $folder_number)"   # Format the folder number with leading zeros if necessary

  rsync_command="rsync -avz"

  # Add include patterns for the specified subfolders
  for subfolder in "${included_subfolders[@]}"; do
    rsync_command+=" --include=/$subfolder"
  done

  # Exclude loose files
  rsync_command+=" --exclude=/*"

  # Perform the synchronization
  eval "$rsync_command" "$source_folder" "$remote_folder"
done