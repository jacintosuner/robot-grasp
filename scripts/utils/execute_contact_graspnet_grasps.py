import numpy as np
import argparse
import math
import robot_controller
from frankapy.franka_constants import FrankaConstants as FC

# Example
# python3 execute_contact_graspnet_grasps.py --grasps_file_path ~/robot-grasp/data/contact_graspnet_pipeline_results/contact_graspnet_results.npz

# See https://frankaemika.github.io/docs/control_parameters.html
JOINT_LIMITS_MIN = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
JOINT_LIMITS_MAX = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
HOME_JOINTS = [0, 0, 0, -math.pi / 2, 0, math.pi / 2, math.pi / 4]
CAMERA_CALIBRATION_FILE = '/home/lifanyu/robot-grasp/data/camera_calibration/cam0_calibration.npz'

d = math.pi/8


def main(input_file):
    # Load the .npz file
    data = np.load(input_file, allow_pickle=True)

    # # Check the keys inside the npz file (optional, to understand its structure)
    # print("Keys in the .npz file:", data.files)

    # Access the 'scores' array
    scores = data['scores'].item()
    pred_grasps_cam = data['pred_grasps_cam'].item()

    # Get the grasp with the highest score for the first item
    best_grasp = pred_grasps_cam[1][np.argmax(scores[1])]

    # Load the camera calibration transformation matrix
    calibration_data = np.load(CAMERA_CALIBRATION_FILE, allow_pickle=True)
    T_cam_to_world = np.linalg.inv(calibration_data['T'])

    # Transform the best grasp from camera reference to world reference
    print(best_grasp)
    best_grasp_world = T_cam_to_world @ best_grasp
    print(best_grasp_world)

    # Move the Robot to the location of the grasp
    controller = robot_controller.FrankaOSCController(
        controller_type="OSC_POSE",
        visualizer=False)
    
    controller.reset_franka()
    # controller.reset(joint_positions = HOME_JOINTS)
    # controller.reset(joint_positions = FC.READY_JOINTS)

    # # Move the robot to the target position and orientation

    # # controller.move_to(target_pos=best_grasp_world[:3, 3], target_rot=best_grasp_world[:3, :3], use_rot=True)
    # controller.move_to(target_pos=best_grasp_world[:3, 3], target_rot=np.eye(3), use_rot=True, duration=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the input .npz file.')
    parser.add_argument('--grasps_file_path', type=str, help='Path to the input .npz file')
    args = parser.parse_args()

    main(args.grasps_file_path)
