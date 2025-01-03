import argparse
import numpy as np
import os
import json
from PIL import Image
from pycocotools import mask as maskUtils

def main(directory, object_name):
    # RBGD data
    file_path = os.path.join(directory, 'rgbdk.npy')
    data = np.load(file_path, allow_pickle=True)

    
    # Load the affordance mask bounding box from the JSON file
    affordance_mask_path = os.path.join(directory, 'affordance_mask.npy')
    affordance_mask = np.load(affordance_mask_path)
    json_file_path = os.path.join(directory, f'grounded_sam_seg_{object_name}.json')
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    # Extract bounding box coordinates
    x1, y1, x2, y2 = map(int, json_data['annotations'][0]['bbox'])

    # Extract segmentation mask from RLE
    # rle = json_data['annotations'][0]['segmentation']
    # seg_data = maskUtils.decode(rle)

    # # Load the affordance mask from the .npy file
    affordance_mask_path = os.path.join(directory, 'affordance_mask.npy')
    affordance_mask = np.load(affordance_mask_path)

    # Transform the data into a dictionary and add the segmentation mask
    data_dict = {key: data.item().get(key) for key in data.item().keys()}
    
    # # Create an empty segmentation mask with the same size as the original image
    seg_data = np.zeros((data_dict['rgb'].shape[0], data_dict['rgb'].shape[1]), dtype=np.uint8)

    # Place the affordance mask within the bounding box in the segmentation mask
    seg_data[y1:y2, x1:x2] = np.array(affordance_mask)

    data_dict['seg'] = seg_data

    # Convert depth data from millimeters to meters
    data_dict['depth'] = data_dict['depth'] / 1000.0
    

    import matplotlib.pyplot as plt

    # Overlay the segmentation mask on the RGB image
    seg_mask = np.zeros_like(data_dict['rgb'])

    seg_mask[:, :, 0] = seg_data * 255  # Red channel for segmentation mask

    # Blend the RGB image and the segmentation mask
    blended_image = np.clip(data_dict['rgb'] * 0.7 + seg_mask * 0.3, 0, 255).astype(np.uint8)

    # Plot the blended image using matplotlib
    plt.imshow(blended_image)
    plt.title('Segmentation Overlay on RGB Image')
    plt.axis('off')  # Hide axes
    plt.show()

    # Save the processed data
    print('Saving the processed data for Contact Graspnet...')
    processed_file_path = os.path.join(directory, 'contact_graspnet_input.npy')
    np.save(processed_file_path, data_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact GraspNet Input Preprocessing")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with all the data needed to construct the Contact GraspNet input')
    parser.add_argument('--object_name', type=str, required=True, help='Name of the object for which the segmentation mask is provided')
    args = parser.parse_args()
    main(args.data_dir, args.object_name)