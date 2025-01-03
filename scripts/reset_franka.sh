#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ros_noetic

cd ~/robot-grasp/third_party/frankapy
bash ./bash_scripts/start_control_pc.sh -i iam-dopey
source catkin_ws/devel/setup.bash


python ~/robot-grasp/third_party/frankapy/scripts/reset_arm.py -z 0.06
