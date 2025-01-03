#!/bin/bash

# Comments:
# This script is the same as pipeline_process_demos_for_taxposed.sh but for a single demo and specific frame numbers for initial and final grasping.
# That means that hands23_detector is not used.

# Example usage: ./pipeline_process_demos_for_taxposed_custom.sh --dir_path ~/robot-grasp/data/demos/demos_20241230_173916/12 --object_name mug

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dir_path) dir_path="$2"; shift ;;
        --object_name) object_name="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if required parameters are set
if [ -z "$dir_path" ] || [ -z "$object_name" ]; then
    echo "Error: --dir_path and --object_name are required parameters."
    exit 1
fi

# Activate miniconda
echo "Activating virtual environment..."
source ~/miniconda3/etc/profile.d/conda.sh

# Extract the clear scene, initial grasping and final grasping npy files
conda activate main_env
cd ~/robot-grasp/scripts/utils
if [ -d "$dir_path" ]; then
    echo "Extracting frames in directory: $dir_path"
    python extract_key_frames.py --dir_path "$dir_path" --extract_frames initial initial_grasping final_grasping --frame_numbers 0 15 50
fi
conda deactivate

# Find the 3D hands in the grasping frames
conda activate wilor
if [ -d "$dir_path" ]; then
    echo "Finding 3D hand in directory: $dir_path"
    cd ~/robot-grasp/third_party/WiLoR
    python demo_rgbdk.py --npy_folder "$dir_path" --out_folder "$dir_path" --save_mesh
    # rm initial_scene.obj
fi
conda deactivate

# Preprocess all data to get taxpose training data
conda activate main_env
## Get the rgb image from the rgbdk image
cd ~/robot-grasp/scripts/utils
if [ -d "$dir_path" ]; then
    echo "Preprocessing data in directory: $dir_path"
    python3 rgbdk_to_rgb.py --input_path "$dir_path/initial_scene.npy" --output_path $dir_path --output_file initial_scene.jpg
fi
## Run Segmentation on the image to find where the object is
cd ~/robot-grasp/third_party/Grounded-SAM-2
if [ -d "$dir_path" ]; then
    echo "Preprocessing data in directory: $dir_path"
    python3 grounded_sam2_local_demo.py --text_prompt "$object_name." --img_path $dir_path/initial_scene.jpg --output_dir $dir_path        
fi

# Get taxpose data ready
## Find the rigid transform between the two hands F
## Move the object point cloud using the hand rigid transformation
## Save the data into a final npy file
cd ~/robot-grasp/scripts/utils
if [ -d "$dir_path" ]; then
    echo "Preprocessing data in directory: $dir_path"
    python taxposed_input_processing.py --dir_path "$dir_path" --object_name "$object_name"
fi

# Put all the processed taxpose data into a single folder
final_dir_name=$(basename "$dir_path")
mkdir -p ~/robot-grasp/data/taxpose_data_for_"$final_dir_name"
cp "$dir_path"/teleport_obj_points.npz ~/robot-grasp/data/taxpose_data_for_"$final_dir_name"
