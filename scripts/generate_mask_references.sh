#!/bin/bash

# Example usage: ./pipeline_mask_references.sh --base_dir 

timestamp=$(date +"%Y%m%d_%H%M%S")
default_output_dir=~/robot-grasp/data/mask_references/reference_$timestamp
output_dir=$default_output_dir

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --base_dir) output_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p $output_dir

source ~/miniconda3/etc/profile.d/conda.sh


# Capture the video
conda activate main_env
cd ~/robot-grasp/scripts/utils
python capture_video.py --output_path $output_dir/demo.mkv
conda deactivate

# Analyse video to find, frame by frame, when the hand is grasping an object, what type of object,...
conda activate hands23
cd ~/robot-grasp/third_party/hands23_detector
python demo.py --video_path $output_dir/demo.mkv --save_dir $output_dir
mv "$output_dir/result.json" "$output_dir/result_hands23.json"
conda deactivate


# Extract the clear scene npy file
# Extract the hand grasping npy file
conda activate main_env
cd ~/robot-grasp/scripts/utils
python extract_key_frames.py --dir_path $output_dir --extract_frames clear initial_grasping
# Find the 3D hand in the grasping frame
mkdir -p $output_dir/hand_frames
cp $output_dir/hand_grasping.npy $output_dir/hand_frames/initial_grasping_scene.npy
conda deactivate
conda activate wilor
cd ~/robot-grasp/third_party/WiLoR
python demo_rgbdk.py --npy_folder $output_dir/hand_frames --out_folder $output_dir/hand_frames --save_mesh

# Visualize the 3D hand in the grasping frame
conda deactivate
conda activate main_env
cd ~/robot-grasp/scripts/utils
python visualize_rgbdk_or_obj.py $output_dir/clear_scene.npy $output_dir/hand_frames/hand_grasping_0.obj
# python visualize_rgbdk_or_obj.py $output_dir/hand_frames/hand_grasping.npy $output_dir/hand_frames/hand_grasping_0.obj

# Generate the affordance reference
python generate_affordance_reference.py --clear_scene $output_dir/clear_scene.npy --hand $output_dir/hand_frames/hand_grasping_0.obj --output_dir $output_dir