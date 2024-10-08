import argparse
import numpy as np
import os

def main(directory, bgrd=False):
    camera_matrix = np.array([[613.32427146,  0.,        633.94909346],
       [ 0.,        614.36077155, 363.33858573],
       [ 0.,          0.,          1.       ]])
    
    # RBGD data
    file_path = os.path.join(directory, 'rgbd.npy')
    data = np.load(file_path, allow_pickle=True)

    
    # Affordance data
    seg_file_path = os.path.join(directory, 'affordance_mask.npy')
    seg_data = np.load(seg_file_path, allow_pickle=True)


    # Transform the data into a dictionary
    print("BGRD: ", bgrd)
    transformed_data = {
        'rgb': data[:, :, 2::-1] if bgrd else data[:, :, :3],  # Extract RGB channels
        'depth': data[:, :, 3] / 1000,  # Extract depth channel and divide by 1000 to go from millimiters to meters
        'K': camera_matrix,  # Camera matrix
        'seg': seg_data
    }

    # Save the processed data
    print('Saving the processed data for Contact Graspnet...')
    processed_file_path = os.path.join(directory, 'contact_graspnet_input.npy')
    np.save(processed_file_path, transformed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact GraspNet Input Preprocessing")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with all the data needed to construct the Contact GraspNet input')
    parser.add_argument('--bgrd', required=False, action='store_true', help='Flag to indicate if the input format is DBGR')
    
    args = parser.parse_args()
    main(args.data_dir, args.bgrd)