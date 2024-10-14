import argparse
import numpy as np
import os
import json
from PIL import Image

def main(directory, bgrd=False):
    # Load RBGD data
    file_path = os.path.join(directory, 'rgbdk.npy')
    data = np.load(file_path, allow_pickle=True).item()

    # Load Affordance data
    seg_file_path = os.path.join(directory, 'affordance_mask.npy')
    seg_data = np.load(seg_file_path, allow_pickle=True)

    # Add the affordance data to the existing data
    data['seg'] = seg_data

    import matplotlib.pyplot as plt

    # Overlay the segmentation mask on the RGB image
    rgb_image = data['rgb']
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
    np.save(processed_file_path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact GraspNet Input Preprocessing")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with all the data needed to construct the Contact GraspNet input')
    parser.add_argument('--bgrd', required=False, action='store_true', help='Flag to indicate if the input format is DBGR')
    
    args = parser.parse_args()
    main(args.data_dir, args.bgrd)