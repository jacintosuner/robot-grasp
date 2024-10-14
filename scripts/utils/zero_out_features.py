import numpy as np
import argparse
import os
import json
from pycocotools import mask as maskUtils

def zero_out_features(affordance_path, to_zero_out_path, output_dir):
    # Load the affordance mask
    affordance_mask = np.load(affordance_path)
    
    # Load the to_zero_out mask from JSON
    with open(to_zero_out_path, 'r') as f:
        zero_out_data = json.load(f)
    
    # Extract the segmentation counts and size
    segmentation = zero_out_data['annotations'][0]['segmentation']
    # Decode the RLE (Run-Length Encoding) counts
    zero_out_mask = maskUtils.decode(segmentation)
    # Zero out the features in affordance_mask where zero_out_features is non-zero
    affordance_mask[zero_out_mask] = 0
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the modified affordance mask
    output_path = os.path.join(output_dir, 'affordance_mask.npy')
    np.save(output_path, affordance_mask)
    print(f"Zeroed affordance mask saved to {output_path}")

    import matplotlib.pyplot as plt

    # Plot the initial affordance mask
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Initial Affordance Mask')
    plt.imshow(np.load(affordance_path), cmap='gray')
    plt.axis('off')

    # Plot the zeroed affordance mask
    plt.subplot(1, 2, 2)
    plt.title('Zeroed Affordance Mask')
    plt.imshow(affordance_mask, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zero out features from the affordance mask.')
    parser.add_argument('--affordance_path', type=str, required=True, help='Path to the affordance mask file.')
    parser.add_argument('--zero_out_features', type=str, required=True, help='Path to the file with features to zero out.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output file.')
    
    args = parser.parse_args()
    
    zero_out_features(args.affordance_path, args.zero_out_features, args.output_dir)