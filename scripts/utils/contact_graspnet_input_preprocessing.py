import argparse
import numpy as np
import os
import json
from PIL import Image

def main(directory, bgrd=False):
    camera_matrix = np.array([[613.32427146,  0.,        633.94909346],
       [ 0.,        614.36077155, 363.33858573],
       [ 0.,          0.,          1.       ]])
    
    # RBGD data
    file_path = os.path.join(directory, 'rgbd.npy')
    data = np.load(file_path, allow_pickle=True)

    
    # Load the affordance mask bounding box from the JSON file
    affordance_mask_path = os.path.join(directory, 'affordance_mask.npy')
    affordance_mask = np.load(affordance_mask_path)
    json_file_path = os.path.join(directory, 'grounded_sam_seg_mug.json')
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    # Extract bounding box coordinates
    x1, y1, x2, y2 = map(int, json_data['annotations'][0]['bbox'])

    # Load the affordance mask from the .npy file
    affordance_mask_path = os.path.join(directory, 'affordance_mask.npy')
    affordance_mask = np.load(affordance_mask_path)

    # Create an empty segmentation mask with the same size as the original image
    seg_data = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)

    # Place the affordance mask within the bounding box in the segmentation mask
    seg_data[y1:y2, x1:x2] = np.array(affordance_mask)


    # Transform the data into a dictionary
    print("BGRD: ", bgrd)
    transformed_data = {
        'rgb': data[:, :, 2::-1] if bgrd else data[:, :, :3],  # Extract RGB channels
        'depth': data[:, :, 3] / 1000,  # Extract depth channel and divide by 1000 to go from millimiters to meters
        'K': camera_matrix,  # Camera matrix
        'seg': seg_data
    }

    import matplotlib.pyplot as plt

    # Overlay the segmentation mask on the RGB image
    rgb_image = transformed_data['rgb']
    seg_mask = np.zeros_like(rgb_image)
    seg_mask[:, :, 0] = seg_data * 255  # Red channel for segmentation mask

    # Blend the RGB image and the segmentation mask
    blended_image = np.clip(rgb_image + seg_mask, 0, 255).astype(np.uint8)

    # Plot the blended image using matplotlib
    plt.imshow(blended_image)
    plt.title('Segmentation Overlay on RGB Image')
    plt.axis('off')  # Hide axes
    plt.show()

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