#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: No input path provided."
  exit 1
fi

rgbd_input_path=$1
mask_reference_path=$2
filename=$(basename -- "$rgbd_input_path")
output_directory=~/robot-grasp/data/pipeline_results_for_${filename%.*}

# Create folder with results in it
echo "Creating folder with pipeline inputs and results..."
mkdir -p $output_directory

cp "$rgbd_input_path" "$output_directory/rgbd.npy"


# Find affordance mask

source ../venv/bin/activate

## Prepare input for Affordance Mask
if [[ " $@ " =~ " --bgrd " ]]; then
    python3 utils/rgdb_to_rgb.py --input_path "$output_directory/rgbd.npy" --output_path $output_directory --bgrd
else
    python3 utils/rgdb_to_rgb.py --input_path "$output_directory/rgbd.npy" --output_path $output_directory
fi

## Run Affordance Mask
cd ../third_party/UCL-AffCorrs/demos
python3 show_part_annotation_correspondence.py --reference_dir_path $mask_reference_path --target_image_path $output_directory/rgb.jpg --output_dir_path $output_directory

# Prepare input for Contact Graspnet
cd ~/robot-grasp/scripts
python3 utils/contact_graspnet_input_preprocessing.py --data_dir $output_directory --bgrd


deactivate

# Run Contact Graspnet
source ~/miniconda3/etc/profile.d/conda.sh
conda activate contact_graspnet_env
cd ~/robot-grasp/third_party/contact_graspnet
python3 contact_graspnet/inference.py --np_path=$output_directory/contact_graspnet_input.npy --local_regions --filter_grasps


# Execute the grasp on the robot