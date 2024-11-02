#!/bin/bash

# Submodules setup
# git submodule init
# git submodule update --init --recursive

# # Main environment setup
source ~/miniconda3/etc/profile.d/conda.sh
# conda env create -f $(pwd)/configs/conda_envs/main_env.yml
# conda activate main_env
# pip install git+https://github.com/lucasb-eyer/pydensecrf.git
# conda deactivate

# # AnyGrasp setup
# conda env create -f  $(pwd)/configs/conda_envs/anygrasp_env.yml
# cd $(pwd)/third_party/anygrasp_sdk/pointnet2
# conda activate anygrasp_env
## MinkowskiEngine installation
# sudo apt install build-essential python3-dev libopenblas-dev
# pip install torch ninja
# pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
# conda activate anygrasp_env
# python setup.py install


# # Contact Graspnet setup
conda env create -f $(pwd)/configs/conda_envs/contact_graspnet_env.yml
conda activate contact_graspnet_env
cd $(pwd)/third_party/contact_graspnet
sh compile_pointnet_tfops.sh


# Grounded-SAM-2 setup
# cd $(pwd)/robot-grasp/third_party/Grounded-SAM-2/checkpoints
# bash download_ckpts.sh
# cd ../gdino_checkpoints
# bash download_ckpts.sh


# Frankapy setup
# mamba create -n ros_noetic python=3.8 ros-noetic-ros-base ros-noetic-franka-gripper -c robostack -c conda-forge
# sudo pip3 install -U catkin_tools
