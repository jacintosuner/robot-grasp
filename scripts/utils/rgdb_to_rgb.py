import argparse
import os
import shutil
from PIL import Image
import numpy as np

def process_files(input_path, output_path, bgrd=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if input_path.endswith('.npy') or input_path.endswith('.npz'):
        full_file_name = input_path
        if input_path.endswith('.npy'):
            rgbd_data = np.load(full_file_name)
        else:
            with np.load(full_file_name) as data:
                rgbd_data = data['arr_0']  # Assuming the array is stored with the key 'arr_0'

        if bgrd:
            # Extract RGB channels (assuming DBGR format is [D, B, G, R])
            rgb_data = rgbd_data[:, :, 2::-1]  # Skip the first channel (D)
        else:
            # Extract RGB channels (assuming RGBD format is [R, G, B, D])
            rgb_data = rgbd_data[:, :, :3]  # Take the first three channels (R, G, B)

        # Convert to an image
        img = Image.fromarray(rgb_data.astype(np.uint8))
        # Save as .jpg
        output_file_name = os.path.join(output_path, 'rgb.jpg')
        img.save(output_file_name)
        print(f"Converted {full_file_name} to {output_file_name}")
    else:
        print(f"Input path {input_path} is not a .npy or .npz file")

def main():
    parser = argparse.ArgumentParser(description='Process RGBD to RGB.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--bgrd', required=False, action='store_true', help='Flag to indicate if the input format is DBGR')
    
    args = parser.parse_args()
    
    process_files(args.input_path, args.output_path, args.bgrd)

if __name__ == "__main__":
    main()