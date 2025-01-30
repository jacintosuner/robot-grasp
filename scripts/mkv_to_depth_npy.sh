#!/bin/bash

# Directory containing the demo directories
demo_dir="/home/jacinto/robot-grasp/data/demos/demos_20241230_173916"

# Iterate over each directory in the demo directory
for dir in "$demo_dir"/*; do
    if [ -d "$dir" ]; then
        video_file="$dir/video.mkv"
        if [ -f "$video_file" ]; then
            python3 utils/mkv_to_depth_npy.py --mkv_file_path "$video_file" --output_npy_path "$dir/depth.npy"
        else
            echo "No video.mkv file found in $dir"
        fi
    fi
done