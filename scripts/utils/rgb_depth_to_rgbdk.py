import os
import numpy as np
from PIL import Image
import argparse

def save_rgb_depth_to_npy(color_path, depth_path, output_path):
    # Load color and depth images
    colors = np.array(Image.open(color_path))
    depths = np.array(Image.open(depth_path))

    # Camera intrinsics
    fx, fy = 927.17, 927.37
    cx, cy = 651.32, 349.62
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Save to npy file
    data = {'rgb': colors, 'depth': depths, 'K': K}
    np.save(output_path, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save RGB and Depth images to npy file.")
    parser.add_argument("--color_path", type=str, help="Path to the color image file.")
    parser.add_argument("--depth_path", type=str, help="Path to the depth image file.")
    parser.add_argument("--output_path", type=str, help="Path to the output npy file.")

    args = parser.parse_args()
    save_rgb_depth_to_npy(args.color_path, args.depth_path, args.output_path)