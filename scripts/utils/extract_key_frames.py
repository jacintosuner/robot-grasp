import os
import cv2
import numpy as np
import json
import argparse
import subprocess
from typing import Tuple
from pyk4a import PyK4APlayback
from pyk4a.calibration import CalibrationType


def read_frame_at_index(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read frame at index {frame_idx}")
    return frame


def save_frame_data_to_rgbdk(dir_path: str, frame_idx: int, output_name: str) -> Tuple[np.ndarray, np.ndarray]:

    if frame_idx is None:
        return

    video_path = os.path.join(dir_path, "video.mkv")
    
    # Open playback using OpenK4APlayback
    with PyK4APlayback(video_path) as playback:
        for _ in range(frame_idx + 1):
            capture = playback.get_next_capture()
            if capture is None:
                raise ValueError(f"Could not read frame at index {frame_idx}")

        # Read RGB frame
        rgb_frame = capture.color[:, :, :3][:, :, ::-1]
        if rgb_frame is None:
            raise ValueError(f"Could not read RGB frame at index {frame_idx}")

        # Read depth frame
        depth_frame = capture.transformed_depth
        if depth_frame is None:
            raise ValueError(f"Could not read depth frame at index {frame_idx}")

        # Calibration data
        calibration = playback.calibration.get_camera_matrix(CalibrationType.COLOR)

    # Save both frames in a single .npy file
    output_path = os.path.join(dir_path, f"{output_name}.npy")
    np.save(output_path, {
        'rgb': rgb_frame,
        'depth': depth_frame,
        'K': calibration,
    })
    print(f"Saved frames to {output_path}")


def extract_frame_index(dir_path: str, frame_type: str = "clear") -> int:
    results_path = os.path.join(dir_path, 'result_hands23.json')
    
    if not os.path.isfile(results_path):
        print(f"Error: {results_path} does not exist.")
        raise FileNotFoundError(f"{results_path} does not exist.")
    
    with open(results_path, 'r') as f:
        data = json.load(f)

    if frame_type == "initial":
        return 0
    
    elif frame_type == "clear":
        indexes = [i for i, item in enumerate(data['images']) if not item.get("predictions")]
        return indexes[0]
    
    elif frame_type == "initial_grasping":
        indexes = [i for i, item in enumerate(data['images']) if any(pred.get("obj_touch") == "neither_,_held" for pred in item.get("predictions", []))]
        if not indexes:
            raise ValueError("No frames with the specified condition found.")
        return indexes[int(len(indexes) * 0.05)]

    elif frame_type == "final_grasping":
        indexes = [i for i, item in enumerate(data['images']) if any(pred.get("obj_touch") == "neither_,_held" for pred in item.get("predictions", []))]
        if not indexes:
            raise ValueError("No frames with the specified condition found.")
        return indexes[int(len(indexes) * 0.95)]
    
    print("No frames with the specified condition found.")
    return


def main():
    parser = argparse.ArgumentParser(description='Extract clear and grasping frames from a directory.')
    parser.add_argument('--dir_path', type=str, required=True, help='Path to the directory containing the video, the images and the results from Detectron2.')
    parser.add_argument('--extract_frames', nargs='+', choices=['initial', 'clear', 'initial_grasping', 'final_grasping'], default=[], help='List of frame types to extract.')
    parser.add_argument('--frame_numbers', type=int, nargs='+', help='List of specific frame numbers to extract. Has to be the same length as --extract_frames.')
    args = parser.parse_args()
    dir_path = args.dir_path

    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a valid directory.")
        return

    for i, frame_type in enumerate(args.extract_frames):
        print(f"Extracting {frame_type} scene...")
        if args.frame_numbers:
            frame_idx = args.frame_numbers[i]
        else:
            frame_idx = extract_frame_index(dir_path, frame_type)
        save_frame_data_to_rgbdk(dir_path, frame_idx, f"{frame_type}_scene")
    return
    

if __name__ == "__main__":
    main()
