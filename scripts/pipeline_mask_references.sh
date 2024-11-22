#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate main_env
cd utils
# python capture_rgbdk.py --output_path ~/robot-grasp/data/mask_references/new_reference --name initial_scene
# echo Sleeping for 4 seconds
# sleep 4
# echo Done sleeping
# python capture_rgbdk.py --output_path ~/robot-grasp/data/mask_references/new_reference --name hand_grasping


# conda deactivate
# conda activate wilor
# cd ~/robot-grasp/third_party/WiLoR
# python demo_rgbdk.py --npy_folder ~/robot-grasp/data/mask_references/new_reference --out_folder ~/robot-grasp/data/mask_references/new_reference --save_mesh

conda deactivate
conda activate main_env
cd ~/robot-grasp/scripts/utils
python visualize_rgbdk_or_obj.py ~/robot-grasp/data/mask_references/new_reference/initial_scene.npy ~/robot-grasp/data/mask_references/new_reference/hand_grasping_0.obj
python generate_affordance_mask.py --initial_scene ~/robot-grasp/data/mask_references/new_reference/initial_scene.npy --hand ~/robot-grasp/data/mask_references/new_reference/hand_grasping_0.obj --output_dir ~/robot-grasp/data/mask_references/new_reference --output_dir ~/robot-grasp/data/mask_references/new_reference