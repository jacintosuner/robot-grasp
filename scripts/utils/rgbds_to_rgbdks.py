import numpy as np
import argparse

def main(input_path, output_path, camera_matrix, bgrd=False):
    # Load the BGRD data from the .npy file
    rgbd_data = np.load(input_path)
    
    # Extract RGB and Depth data
    depth_data = rgbd_data[:, :, 3]

    if bgrd:
        # Extract RGB channels (assuming DBGR format is [D, B, G, R])
        rgb_data = rgbd_data[:, :, 2::-1]  # Skip the first channel (D)
    else:
        # Extract RGB channels (assuming RGBD format is [R, G, B, D])
        rgb_data = rgbd_data[:, :, :3]  # Take the first three channels (R, G, B)
    
    # Save the data to a new .npy file as a dictionary
    output_data = {
        'rgb': rgb_data,
        'depth': depth_data,
        'K': camera_matrix
    }
    np.save(output_path, output_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BGRD data and save RGB, Depth, and Camera Matrix.")
    parser.add_argument("input_path", type=str, help="Path to the input .npy file containing BGRD data")
    parser.add_argument("output_path", type=str, help="Path to the output .npy file to save the processed data")
    parser.add_argument("--bgrd", action="store_true", help="Flag to indicate if the input data is in BGRD format")

    K = np.array([[613.32427146,  0.,        633.94909346],
       [ 0.,        614.36077155, 363.33858573],
       [ 0.,          0.,          1.       ]])
    
    args = parser.parse_args()
    main(args.input_path, args.output_path, K, args.bgrd)