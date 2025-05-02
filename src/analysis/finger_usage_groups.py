"""
Filename: src/analysis/finger_usage_groups.py

This script analyzes participants based on which fingers they used in capillary velocity data.
It categorizes participants into three groups:
1. People who used just the ring finger
2. People who used both ring and pointer fingers
3. People who used other finger combinations

The script loads the same data as finger_stats.py and uses similar merging techniques.
"""

import os
import pandas as pd
import numpy as np
from src.config import PATHS

# Define constants
cap_flow_path = PATHS['cap_flow']

def main():
    """Main function for analyzing finger usage patterns."""
    print("\nAnalyzing participant finger usage patterns...")
    
    # Load data
    data_filepath = os.path.join(cap_flow_path, 'summary_df_nhp_video_stats.csv')
    df = pd.read_csv(data_filepath)
    
    # Standardize finger column names (same as in finger_stats.py)
    df['Finger'] = df['Finger'].str[1:]
    df['Finger'] = df['Finger'].str.lower()
    df['Finger'] = df['Finger'].str.capitalize()
    df['Finger'] = df['Finger'].replace('Mid', 'Middle')
    df['Finger'] = df['Finger'].replace('Index', 'Pointer')
    print(f"Fingers in dataset: {df['Finger'].unique()}")

    # Load finger stats data
    finger_stats_df = pd.read_csv(os.path.join(cap_flow_path, 'finger_stats.csv'))
    
    # Merge with velocity data
    merged_df = pd.merge(df, finger_stats_df, on='Participant', how='left')
    
    # Filter for control data (same as in finger_stats.py)
    controls_df = merged_df[merged_df['SET'] == 'set01']
    
    # Group by participant and find which fingers each participant used
    participant_fingers = controls_df.groupby('Participant')['Finger'].unique()
    
    # Create finger usage groups
    ring_only_participants = []
    ring_pointer_participants = []
    other_finger_participants = []
    
    for participant, fingers in participant_fingers.items():
        # Convert from numpy array to a set for easier comparison
        finger_set = set(fingers)
        
        if finger_set == {'Ring'}:
            ring_only_participants.append(participant)
        elif finger_set == {'Ring', 'Pointer'} or finger_set == {'Pointer', 'Ring'}:
            ring_pointer_participants.append(participant)
        else:
            other_finger_participants.append(participant)
    
    # Print results
    print("\nParticipant groups based on finger usage:")
    print(f"\nGroup 1: Participants who used ONLY the ring finger ({len(ring_only_participants)}):")
    for p in sorted(ring_only_participants):
        print(f"  - {p}")
    
    print(f"\nGroup 2: Participants who used BOTH ring and pointer fingers ({len(ring_pointer_participants)}):")
    for p in sorted(ring_pointer_participants):
        print(f"  - {p}")
    
    print(f"\nGroup 3: Participants who used other finger combinations ({len(other_finger_participants)}):")
    for p in sorted(other_finger_participants):
        part_fingers = participant_fingers[p]
        print(f"  - {p}: {', '.join(part_fingers)}")
    
    # Summary stats
    total_participants = len(ring_only_participants) + len(ring_pointer_participants) + len(other_finger_participants)
    print(f"\nTotal participants analyzed: {total_participants}")
    print(f"Ring only: {len(ring_only_participants)} ({len(ring_only_participants)/total_participants*100:.1f}%)")
    print(f"Ring and pointer: {len(ring_pointer_participants)} ({len(ring_pointer_participants)/total_participants*100:.1f}%)")
    print(f"Other combinations: {len(other_finger_participants)} ({len(other_finger_participants)/total_participants*100:.1f}%)")

if __name__ == "__main__":
    main() 