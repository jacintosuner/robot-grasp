#!/bin/bash

# Example usage: ./pipeline_mask_references.sh

timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir=~/robot-grasp/data/mask_references/reference_$timestamp
mkdir -p $output_dir

source ~/miniconda3/etc/profile.d/conda.sh
conda activate main_env
cd ~/robot-grasp/scripts/utils
python capture_rgbdk.py --output_path $output_dir --name initial_scene
echo Sleeping for 4 seconds
sleep 4
echo Done sleeping
python capture_rgbdk.py --output_path $output_dir/hand_frames --name hand_grasping


mkdir -p $output_dir/hand_frames
conda deactivate
conda activate wilor
cd ~/robot-grasp/third_party/WiLoR
python demo_rgbdk.py --npy_folder $output_dir/hand_frames --out_folder $output_dir/hand_frames --save_mesh

conda deactivate
conda activate main_env
cd ~/robot-grasp/scripts/utils
python visualize_rgbdk_or_obj.py $output_dir/initial_scene.npy $output_dir/hand_frames/hand_grasping_0.obj
# python visualize_rgbdk_or_obj.py $output_dir/hand_frames/hand_grasping.npy $output_dir/hand_frames/hand_grasping_0.obj
python generate_affordance_reference.py --initial_scene $output_dir/initial_scene.npy --hand $output_dir/hand_frames/hand_grasping_0.obj --output_dir $output_dir