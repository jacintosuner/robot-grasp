#!/bin/bash

# Examples:
# cd scripts
# ./pipeline_contact_graspnet.sh --rgbdk_path ~/robot-grasp/data/rgbdks/rgbdk.npy --mask_reference_path ~/robot-grasp/data/mask_references/mug_reference
# ./pipeline_contact_graspnet.sh --mask_reference_path ~/robot-grasp/data/mask_references/mug_reference


# Creating folder with results
echo "Creating folder with pipeline inputs and results..."
output_directory=~/robot-grasp/data/contact_graspnet_pipeline_results
mkdir -p $output_directory



# Parse arguments
echo "Parsing arguments..."
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --rgbdk_path) rgbdk_input_path="$2"; shift ;;
    --mask_reference_path) mask_reference_path="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

if [ -z "$mask_reference_path" ]; then
  echo "Error: --mask_reference_path not provided."
  exit 1
fi

if [ -z "$rgbdk_input_path" ]; then
  echo "No --rgbdk_path provided. Capturing RGBDK data."
  cd ~/robot-grasp/scripts
  python3 utils/capture_rgbdk.py --output_path $output_directory/rgbdk.npy
else
  echo "Using provided --rgbdk_path."
  cp $rgbdk_input_path $output_directory/rgbdk.npy
fi


# Find affordance mask
# Activate virtual environment for the first part of the pipeline
echo "Activating virtual environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate main_env

## Get rgb from rbgd / bgrd
python3 utils/rgbdk_to_rgb.py --input_path "$output_directory/rgbdk.npy" --output_path $output_directory

### Run Segmentation on the image to find where the mug is
echo "Running segmentation on the image..."
cd ~/robot-grasp/third_party/Grounded-SAM-2
python3 grounded_sam2_local_demo.py --text_prompt "mug." --img_path $output_directory/rgb.jpg --output_dir $output_directory 

# ### Create an image only with the segmented object within the bounding box
echo "Creating an image only with the segmented object within the bounding box..."
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
python3 utils/contact_graspnet_input_preprocessing.py --data_dir $output_directory

conda deactivate

# # Run Contact Graspnet
conda activate contact_graspnet_env
cd ~/robot-grasp/third_party/contact_graspnet
python3 contact_graspnet/inference.py --np_path=$output_directory/contact_graspnet_input.npy --local_regions --filter_grasps --results_path=$output_directory

# Execute the grasp on the robot
cd ~/robot-grasp/scripts/utils
python execute_contact_graspnet_grasps.py --grasps_file_path $output_directory/contact_graspnet_results.npz