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

## Get rgb from rbgd / bgrd
if [[ " $@ " =~ " --bgrd " ]]; then
    python3 utils/rgdb_to_rgb.py --input_path "$output_directory/rgbd.npy" --output_path $output_directory --bgrd
else
    python3 utils/rgdb_to_rgb.py --input_path "$output_directory/rgbd.npy" --output_path $output_directory
fi

### Run Segmentation on the image to find where the mug is
cd ~/robot-grasp/third_party/Grounded-SAM-2
python3 grounded_sam2_local_demo.py --text_prompt "mug." --img_path $output_directory/rgb.jpg --output_dir $output_directory 

### Create an image only with the segmented object within the bounding box
cd ~/robot-grasp/scripts
python3 utils/create_image_with_segmented_object.py --input_path $output_directory/rgb.jpg --segmentation_path $output_directory/grounded_sam_seg_mug.json --output_path $output_directory/seg_mug.jpg


## Run Affordance Mask
cd ../third_party/UCL-AffCorrs/demos
python3 show_part_annotation_correspondence.py --reference_dir_path $mask_reference_path --target_image_path $output_directory/seg_mug.jpg --output_dir_path $output_directory


## Zero out the features from the robot

### Run Grounded-SAM-2 on the image
cd ~/robot-grasp/third_party/Grounded-SAM-2
python3 grounded_sam2_local_demo.py --text_prompt "robot." --img_path $output_directory/seg_mug.jpg --output_dir $output_directory


### Zero out the features from the robot
cd ~/robot-grasp/scripts/utils
python3 zero_out_features.py --affordance_path $output_directory/affordance_mask.npy --zero_out_features $output_directory/grounded_sam_seg_robot.json --output_dir $output_directory


# ## Prepare input for AnyGrasp
# cd ~/robot-grasp/scripts
# python3 utils/anygrasp_input_preprocessing.py --data_dir $output_directory --bgrd









# # Prepare input for Contact Graspnet
cd ~/robot-grasp/scripts
python3 utils/contact_graspnet_input_preprocessing.py --data_dir $output_directory --bgrd

deactivate

# # Run Contact Graspnet
source ~/miniconda3/etc/profile.d/conda.sh
conda activate contact_graspnet_env
cd ~/robot-grasp/third_party/contact_graspnet
python3 contact_graspnet/inference.py --np_path=$output_directory/contact_graspnet_input.npy --local_regions --filter_grasps

# Execute the grasp on the robot