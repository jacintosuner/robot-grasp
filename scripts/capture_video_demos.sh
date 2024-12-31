#!/bin/bash

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output_directory) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set default output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR=~/robot-grasp/data/demos_$timestamp/videos
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

for i in $(seq 1 15); do
    echo "Press any key to start capturing video $i..."
    read -n 1 -s
    for j in $(seq 4 -1 1); do
        echo "Starting in $j seconds..."
        sleep 1
    done
    VIDEO_DIR="$OUTPUT_DIR/$i"
    mkdir -p "$VIDEO_DIR"
    OUTPUT_PATH="$VIDEO_DIR/video.mkv"
    python utils/capture_video.py --duration 6 --output_path "$OUTPUT_PATH"
    echo "Video $i saved to $OUTPUT_PATH"
done

echo "All videos have been captured."