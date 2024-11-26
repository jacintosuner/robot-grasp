#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ros_noetic

cd ~/robot-grasp/third_party/frankapy
bash ./bash_scripts/start_control_pc.sh -i iam-dopey
source catkin_ws/devel/setup.bash

cd ~/robot-grasp/third_party/rtc_vision_toolbox
python -m scripts.robot_camera_calibration
mv ~/robot-grasp/third_party/rtc_vision_toolbox/T_base2camera_* ~/robot-grasp/data/camera_calibration/camera2robot.npz
rm ~/robot-grasp/third_party/rtc_vision_toolbox/T_*