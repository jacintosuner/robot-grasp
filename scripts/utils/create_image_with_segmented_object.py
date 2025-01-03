import argparse
import json
import numpy as np
from PIL import Image

from pycocotools import mask as maskUtils

def parse_args():
    parser = argparse.ArgumentParser(description="Create an image with segmented object.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--segmentation_path', type=str, required=True, help='Path to the segmentation JSON file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output image.')
    return parser.parse_args()

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def load_segmentation_and_bbox(segmentation_path):
    with open(segmentation_path, 'r') as f:
        segmentation_data = json.load(f)
    return maskUtils.decode(segmentation_data['annotations'][0]['segmentation']), segmentation_data['annotations'][0]['bbox']

def create_segmented_image(image, segmentation_data, bbox):
    mask = segmentation_data.astype(np.uint8) * 255  # Convert mask to binary format

    # Create a new image with black background
    segmented_image = Image.new("RGB", image.size, (0, 0, 0))
    image_np = np.array(image)
    segmented_image_np = np.array(segmented_image)

    # Apply mask to the image
    segmented_image_np = np.where(mask[:, :, None], image_np, 0)

    # Crop the image to the bounding box
    x1, y1, x2, y2 = map(int, bbox)
    cropped_segmented_image = segmented_image_np[y1:y2, x1:x2]
    # cropped_segmented_image = image_np[y1:y2, x1:x2]

    return Image.fromarray(cropped_segmented_image)

def main():
    args = parse_args()
    image = load_image(args.input_path)
    segmentation_data, bbox = load_segmentation_and_bbox(args.segmentation_path)
    segmented_image = create_segmented_image(image, segmentation_data, bbox)
    segmented_image.save(args.output_path)

if __name__ == "__main__":
    main()