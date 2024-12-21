#!/bin/bash

# Example usage: ./pipeline_mask_references.sh

timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir=~/robot-grasp/data/mask_references/reference_$timestamp
mkdir -p $output_dir

source ~/miniconda3/etc/profile.d/conda.sh

# OLD:
# conda activate main_env
# cd ~/robot-grasp/scripts/utils
# python capture_rgbdk.py --output_path $output_dir --name clear_scene
# echo Sleeping for 4 seconds
# sleep 4
# echo Done sleeping
# python capture_rgbdk.py --output_path $output_dir/hand_frames --name hand_grasping

# NEW:
# Capture the video or get an already recorded video
conda activate main_env
cd ~/robot-grasp/scripts/utils
python capture_video.py --output_path $output_dir/demo.mkv
conda deactivate

# Extract when the hand is grasping the object
conda activate hands23
cd ~/robot-grasp/third_party/hands23_detector
python demo.py --video_path $output_dir/demo.mkv --save_dir $output_dir
conda deactivate


# Extract the clear scene npy file
# Extract the hand grasping npy file
conda activate main_env
cd ~/robot-grasp/scripts/utils
python extract_clear_and_grasping_frames.py --dir_path $output_dir


# Find the 3D hand in the grasping frame
mkdir -p $output_dir/hand_frames
cp $output_dir/hand_grasping.npy $output_dir/hand_frames/hand_grasping.npy
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