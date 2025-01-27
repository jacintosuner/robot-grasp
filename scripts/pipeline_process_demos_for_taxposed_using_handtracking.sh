#!/bin/bash

# Example usage: ./pipeline_process_demos_for_taxposed.sh --dir_path ~/robot-grasp/data/demos/demos_20241230_173916 --object_name mug

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dir_path) dir_path="$2"; shift ;;
        --object_name) object_name="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Activate miniconda
echo "Activating virtual environment..."
source ~/miniconda3/etc/profile.d/conda.sh

# # Analyze videos to find, frame by frame, when the hand is grasping an object, what type of object it is, etc.
conda activate hands23
cd ~/robot-grasp/third_party/hands23_detector
for video_dir in "$dir_path"/*; do
    if [ -d "$video_dir" ]; then
        echo "Processing video in directory: $video_dir"
        python demo.py --video_path "$video_dir/video.mkv" --save_dir "$video_dir"
        mv "$video_dir/result.json" "$video_dir/result_hands23.json"
    fi
done
conda deactivate

# Extract the clear scene, initial grasping and final grasping npy files
conda activate main_env
cd ~/robot-grasp/scripts/utils
for video_dir in "$dir_path"/*; do
    if [ -d "$video_dir" ]; then
        echo "Extracting frames in directory: $video_dir"
        python extract_key_frames.py --dir_path "$video_dir" --extract_frames initial initial_grasping final_grasping
    fi
done
conda deactivate


# Find the 3D hands in the grasping frames
conda activate wilor
for video_dir in "$dir_path"/*; do
    if [ -d "$video_dir" ]; then
        echo "Finding 3D hand in directory: $video_dir"
        cd ~/robot-grasp/third_party/WiLoR
        python demo_rgbdk.py --npy_folder "$video_dir" --out_folder "$video_dir" --save_mesh
        # rm initial_scene.obj
    fi
done
conda deactivate

# Preprocess all data to get taxpose training data
conda activate main_env
## Get the rgb image from the rgbdk image
cd ~/robot-grasp/scripts/utils
for video_dir in "$dir_path"/*; do
    if [ -d "$video_dir" ]; then
        echo "Preprocessing data in directory: $video_dir"
        python3 rgbdk_to_rgb.py --input_path "$video_dir/initial_scene.npy" --output_path $video_dir --output_file initial_scene.jpg
    fi
done
## Run Segmentation on the image to find where the object is
cd ~/robot-grasp/third_party/Grounded-SAM-2
for video_dir in "$dir_path"/*; do
    if [ -d "$video_dir" ]; then
        echo "Preprocessing data in directory: $video_dir"
        python3 grounded_sam2_local_demo.py --text_prompt "$object_name." --img_path $video_dir/initial_scene.jpg --output_dir $video_dir        
    fi
done


# Get taxpose data ready
## Find the rigid transform between the two hands F
## Move the object point cloud using the hand rigid transformation
## Save the data into a final npy file
cd ~/robot-grasp/scripts/utils
for video_dir in "$dir_path"/*; do
    if [ -d "$video_dir" ]; then
        echo "Preprocessing data in directory: $video_dir"
        python taxposed_train_data_processing.py --dir_path "$video_dir" --object_name "$object_name"
    fi
done


# Put all the processed taxpose data into a single folder
final_dir_name=$(basename "$dir_path")
mkdir -p ~/robot-grasp/data/taxpose_data_for_"$final_dir_name"
counter=0
for video_dir in "$dir_path"/*; do
    if [ -d "$video_dir" ]; then
        counter=$((counter + 1))
        cp "$video_dir"/"${counter}_teleport_obj_points.npz" ~/robot-grasp/data/taxpose_data_for_"$final_dir_name"
    fi
done