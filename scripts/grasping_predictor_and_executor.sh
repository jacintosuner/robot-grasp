#!/bin/bash

# Examples:
# cd scripts
# ./grasping_predictor_and_executor.sh --mask_reference_path ~/robot-grasp/data/mask_references/mug_20250113 --object_name mug
# ./grasping_predictor_and_executor.sh --mask_reference_path ~/robot-grasp/data/mask_references/duck_reference_20241218 --object_name yellow-duck

# Creating folder with results
echo "Creating folder with pipeline inputs and results..."
# Extract date from mask_reference_path if it contains a date, otherwise use current date
date_suffix=$(date '+%Y%m%d_%H%M%S')
output_directory=~/robot-grasp/data/contact_graspnet_pipeline_results_${date_suffix}
mkdir -p $output_directory

# Activate virtual environment for the first part of the pipeline
echo "Activating virtual environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate main_env


# Parse arguments
echo "Parsing arguments..."
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --rgbdk_path) rgbdk_input_path="$2"; shift ;;
    --mask_reference_path) mask_reference_path="$2"; shift ;;
    --object_name) object_name="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

if [ -z "$mask_reference_path" ]; then
  echo "Error: --mask_reference_path not provided."
  exit 1
fi

if [ -z "$object_name" ]; then
  echo "Error: --object_name not provided."
  exit 1
fi

if [ -z "$rgbdk_input_path" ]; then
  echo "No --rgbdk_path provided. Capturing RGBDK data."
  cd ~/robot-grasp/scripts
  python3 utils/capture_rgbdk.py --output_path $output_directory --device_id 1
else
  echo "Using provided --rgbdk_path."
  cp $rgbdk_input_path $output_directory/rgbdk.npy
fi

# Generate affordance mask reference from reference image



# Find affordance mask

## Get rgb from rbgd / bgrd
python3 utils/rgbdk_to_rgb.py --input_path "$output_directory/rgbdk.npy" --output_path $output_directory

## Run Segmentation on the image to find where the object is
echo "############################# Running segmentation on the image..."
cd ~/robot-grasp/third_party/Grounded-SAM-2
python3 grounded_sam2_local_demo.py --text_prompt "$object_name." --img_path $output_directory/rgb.jpg --output_dir $output_directory 

## Create an image only with the segmented object within the bounding box
echo "############################# Creating an image only with the segmented object within the bounding box..."
cd ~/robot-grasp/scripts
python3 utils/create_image_with_segmented_object.py --input_path $output_directory/rgb.jpg --segmentation_path $output_directory/grounded_sam_seg_${object_name}.json --output_path $output_directory/seg_${object_name}.jpg

## Run Affordance Mask
echo "############################# Running Affordance Mask..."
cd ../third_party/UCL-AffCorrs/demos
python3 show_part_annotation_correspondence.py --reference_dir_path $mask_reference_path --target_image_path $output_directory/seg_${object_name}.jpg --output_dir_path $output_directory


# Zero out the features from the robot
### Run Grounded-SAM-2 on the image
cd ~/robot-grasp/third_party/Grounded-SAM-2
python3 grounded_sam2_local_demo.py --text_prompt "robot." --img_path $output_directory/seg_${object_name}.jpg --output_dir $output_directory


### Zero out the features from the robot
echo "############################# Zeroing out the features from the robot..."
cd ~/robot-grasp/scripts/utils
python3 zero_out_features.py --affordance_path $output_directory/affordance_mask.npy --zero_out_features $output_directory/grounded_sam_seg_robot.json --output_dir $output_directory


# Prepare input for Contact Graspnet
echo "############################# Preparing input for Contact Graspnet..."
cd ~/robot-grasp/scripts
python3 utils/contact_graspnet_input_preprocessing.py --data_dir $output_directory --object_name $object_name

conda deactivate

# Run Contact Graspnet
echo "############################# Running Contact Graspnet..."
conda activate contact_graspnet_env
cd ~/robot-grasp/third_party/contact_graspnet
python3 contact_graspnet/inference.py --np_path $output_directory/contact_graspnet_input.npy --filter_grasps --results_path $output_directory --ckpt_dir ~/robot-grasp/third_party/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_0025
conda deactivate

# TAXPOSED
# Prepare input for TAXPOSED
# Predict TAXPOSED transform
echo "############################# Preparing input for TAXPOSED..."
conda activate taxposed
cd ~/robot-grasp/third_party/taxposeD
python3 taxposed_inference_wrapper.py --input_path $output_directory/contact_graspnet_input.npy --output_dir $output_directory --gsam2_pred_path $output_directory/grounded_sam_seg_${object_name}.json --cfg plan.yaml --debug
conda deactivate

# Visualize the results
# conda activate main_env
# cd ~/robot-grasp/scripts/utils
# python3 visualize_taxposed_prediction.py --taxposed_prediction $output_directory/taxposed_prediction.npy --point_cloud $output_directory/point_cloud.npy --segmented_point_cloud $output_directory/segmented_point_cloud.npy
# conda deactivate


# Run Robot
## Select and execute the grasp on the robot
conda activate ros_noetic
cd ~/robot-grasp/third_party/frankapy
bash ./bash_scripts/start_control_pc.sh -i iam-dopey
source catkin_ws/devel/setup.bash
python ~/robot-grasp/scripts/utils/execute_contact_graspnet_grasps.py --grasps_file_path $output_directory/contact_graspnet_results.npz --taxposed_prediction $output_directory/taxposed_prediction.npy

