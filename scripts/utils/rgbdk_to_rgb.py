import numpy as np
import argparse
import os
from PIL import Image

def rgbdk_to_rgb(input_path, output_path, output_file):
    # Load the RGB-DK numpy file
    data = np.load(input_path, allow_pickle=True)

    # Save the RGB data as a new numpy file
    # Extract the RGB data
    rgb_data = data.item().get('rgb')

    # Convert the numpy array to an image
    image = Image.fromarray(rgb_data.astype('uint8')).convert('RGB')

    # Save the image as a JPG file
    output_file = os.path.join(output_path, output_file)
    image.save(output_file, 'JPEG')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RGB-DK numpy file to RGB numpy file.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input RGB-DK numpy file.")
    parser.add_argument('--output_path', type=str, required=True, help="Directory to save the output jpg file.")
    parser.add_argument('--output_file', type=str, default='rgb.jpg', help="Name of the output jpg file")
    
    args = parser.parse_args()
    
    rgbdk_to_rgb(args.input_path, args.output_path, args.output_file)