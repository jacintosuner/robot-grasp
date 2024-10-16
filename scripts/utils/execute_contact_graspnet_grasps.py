import numpy as np
import argparse

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

    print("Best grasp:", best_grasp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the input .npz file.')
    parser.add_argument('--grasps_file_path', type=str, help='Path to the input .npz file')
    args = parser.parse_args()

    main(args.grasps_file_path)
