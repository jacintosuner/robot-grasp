#!/bin/bash

# example: ./pipeline_anygrasp.sh /home/jacinto/robot-grasp/data/rgbdks/rgbdk.npy


export CUDA_HOME=/usr/local/cuda-11.7
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

rgbdk_file_path=${1:-""}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate anygrasp_env
cd ~/robot-grasp/third_party/anygrasp_sdk/grasp_detection

if [ -n "$rgbdk_file_path" ]; then
    python demo.py --checkpoint_path log/checkpoint_detection.tar --top_down_grasp --debug --rgbdk_file_path $rgbdk_file_path
else
    python demo.py --checkpoint_path log/checkpoint_detection.tar --top_down_grasp --debug
fi