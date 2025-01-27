#!/bin/bash

# Example usage: ./generate_mask_references.sh
# Example usage: ./generate_mask_references.sh --video_demo ~/robot-grasp/data/mask_references/reference_20241122_153952/video.mkv

timestamp=$(date +"%Y%m%d_%H%M%S")
default_output_dir=~/robot-grasp/data/mask_references/reference_$timestamp
output_dir=$default_output_dir
video_demo=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --base_dir) output_dir="$2"; shift ;;
        --video_demo) video_demo="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p $output_dir

source ~/miniconda3/etc/profile.d/conda.sh

# Capture the video only if not provided via --video_demo
if [ -z "$video_demo" ]; then
    # Capture the video
    for i in {5..1}; do
        echo "Recording video in $i..."
        sleep 1
    done
    echo "Capturing video..."
    conda activate main_env
    cd ~/robot-grasp/scripts/utils
    python capture_video.py --output_path $output_dir/video.mkv --duration 7
    conda deactivate
else
    # Copy the provided video to output directory
    cp "$video_demo" "$output_dir/video.mkv"
fi

# Analyse video to find, frame by frame, when the hand is grasping an object, what type of object,...
echo "Analyzing video to find the frames where the hand is grasping an object..."
conda activate hands23
cd ~/robot-grasp/third_party/hands23_detector
python demo.py --video_path $output_dir/video.mkv --save_dir $output_dir
mv "$output_dir/result.json" "$output_dir/result_hands23.json"
conda deactivate


# Extract the clear scene npy file
# Extract the hand grasping npy file
echo "Extracting clear scene and hand grasping npy files..."
conda activate main_env
cd ~/robot-grasp/scripts/utils
python extract_key_frames.py --dir_path $output_dir --extract_frames initial initial_grasping # clear # final_grasping
# Find the 3D hand in the grasping frame
mkdir -p $output_dir/hand_frames
cp $output_dir/initial_grasping_scene.npy $output_dir/hand_frames/initial_grasping_scene.npy
conda deactivate
conda activate wilor
cd ~/robot-grasp/third_party/WiLoR
python demo_rgbdk.py --npy_folder $output_dir/hand_frames --out_folder $output_dir/hand_frames --save_mesh

# Visualize the 3D hand in the grasping frame
echo "Visualizing the 3D hand in the grasping frame..."
conda deactivate
conda activate main_env
cd ~/robot-grasp/scripts/utils
python visualize_rgbdk_or_obj_or_pcd.py $output_dir/initial_scene.npy $output_dir/hand_frames/initial_grasping_scene_0.obj
# python visualize_rgbdk_or_obj_or_pcd.py $output_dir/hand_frames/hand_grasping.npy $output_dir/hand_frames/initial_grasping_scene_0.obj

# Generate the affordance reference
echo "Generating the affordance reference..."
python generate_affordance_reference.py --clear_scene $output_dir/initial_scene.npy --hand $output_dir/hand_frames/initial_grasping_scene_0.obj --output_dir $output_dir