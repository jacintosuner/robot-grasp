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


def save_frame_data(dir_path: str, frame_idx: int, output_name: str) -> Tuple[np.ndarray, np.ndarray]:

    if frame_idx is None:
        return

    video_path = os.path.join(dir_path, "demo.mkv")
    
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
    results_path = os.path.join(dir_path, 'result.json')
    
    if not os.path.isfile(results_path):
        print(f"Error: {results_path} does not exist.")
        raise FileNotFoundError(f"{results_path} does not exist.")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    
    if frame_type == "clear":
        indexes = [i for i, item in enumerate(data['images']) if not item.get("predictions")]
        return indexes[-1]
    elif frame_type == "grasping":
        # Option 1: Median index
        # indexes = [i for i, item in enumerate(data['images']) if any(pred.get("obj_touch") == "neither_,_held" for pred in item.get("predictions", []))]
        # return indexes[len(indexes) // 2]
        
        # Option 2: Max confidence score
        scores = []
        for i, item in enumerate(data['images']):
            for pred in item.get("predictions", []):
                if pred.get("obj_touch") == "neither_,_held":
                    scores.append((i, float(pred["obj_touch_scores"]["neither_,_held"])))
        
        if not scores:
            raise ValueError("No frames with the specified condition found.")
        
        highest_score_index = max(scores, key=lambda x: x[1])[0]
        return highest_score_index
    
    print("No frames with the specified condition found.")
    return
    
    return median_index


def main():
    parser = argparse.ArgumentParser(description='Extract clear and grasping frames from a directory.')
    parser.add_argument('--dir_path', type=str, required=True, help='Path to the directory containing the video, the images and the results from Detectron2.')
    
    args = parser.parse_args()
    dir_path = args.dir_path

    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a valid directory.")
        return
    

    # Extract clear scene
    print("Extracting clear scene...")
    frame_idx = extract_frame_index(dir_path, "clear")
    save_frame_data(dir_path, frame_idx, "clear_scene")

    # Extract grasping scene
    print("Extracting grasping scene...")
    frame_idx = extract_frame_index(dir_path, "grasping")
    save_frame_data(dir_path, frame_idx, "hand_grasping")
    

if __name__ == "__main__":
    main()