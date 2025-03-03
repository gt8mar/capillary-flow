#!/bin/bash

# Segment All Frogs
# This script runs the frog segmentation tool on all JPG images in the Downloads/whole-frog folder
# 
# Usage:
#   ./scripts/segment_all_frogs.sh

# Get the current hostname to determine paths
HOSTNAME=$(hostname)

# Set paths based on hostname
if [[ "$HOSTNAME" == "LAPTOP-I5KTBOR3" ]]; then
    CAPFLOW_PATH="C:\\Users\\gt8ma\\capillary-flow"
    DOWNLOAD_PATH="C:\\Users\\gt8ma\\Downloads\\whole-frog"
elif [[ "$HOSTNAME" == "Quake-Blood" ]]; then
    CAPFLOW_PATH="C:\\Users\\gt8mar\\capillary-flow"
    DOWNLOAD_PATH="C:\\Users\\gt8mar\\Downloads\\whole-frog"
else
    CAPFLOW_PATH="/hpc/projects/capillary-flow"
    DOWNLOAD_PATH="/home/downloads/whole-frog"
fi

# Create log directory if it doesn't exist
mkdir -p "$CAPFLOW_PATH/logs"
LOG_FILE="$CAPFLOW_PATH/logs/frog_segmentation_$(date +%Y%m%d_%H%M%S).log"

echo "Starting frog segmentation process at $(date)" | tee -a "$LOG_FILE"
echo "Looking for JPG files in $DOWNLOAD_PATH" | tee -a "$LOG_FILE"

# Count the number of JPG files
NUM_FILES=$(find "$DOWNLOAD_PATH" -name "*.JPG" | wc -l)
echo "Found $NUM_FILES JPG files to process" | tee -a "$LOG_FILE"

# Process each JPG file
COUNTER=0
for img in "$DOWNLOAD_PATH"/*.JPG; do
    if [ -f "$img" ]; then
        COUNTER=$((COUNTER+1))
        FILENAME=$(basename "$img")
        echo "[$COUNTER/$NUM_FILES] Processing $FILENAME..." | tee -a "$LOG_FILE"
        
        # Run the segmentation script
        python -m scripts.frog_segmentation "$img" 2>&1 | tee -a "$LOG_FILE"
        
        echo "Completed $FILENAME" | tee -a "$LOG_FILE"
        echo "----------------------------------------" | tee -a "$LOG_FILE"
    fi
done

echo "Segmentation process completed at $(date)" | tee -a "$LOG_FILE"
echo "Processed $COUNTER files" | tee -a "$LOG_FILE"
echo "Log file saved to $LOG_FILE" 