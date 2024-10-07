"""
Show part query correspondence in single object images with AffCorrs
"""
# Standard imports
import os
import sys
from PIL import Image
import yaml
import numpy as np
import matplotlib.pyplot as plt
# Vision imports
import torch
import cv2
sys.path.append("..")
from models.correspondence_functions import (overlay_segment, resize)
from models.aff_corrs import AffCorrs_V1
import argparse

# # User-defined constants
# SUPPORT_DIR = "../affordance_database/usb/"
# TARGET_IMAGE_PATH = "./images/demo_affordance/eth.jpg"
SUPPORT_DIR = "../affordance_database/mug_multimask"
TARGET_IMAGE_PATH = "./images/demo_affordance/mug3.jpeg"

# Other constants
PATH_TO_CONFIG  = "../config/default_config.yaml"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COLORS = [[255,0,0],[255,255,0],[255,0,255],
          [0,255,0],[0,0,255],[0,255,255]]

# Load arguments
with open(PATH_TO_CONFIG) as f:
    args = yaml.load(f, Loader=yaml.Loader)
args['low_res_saliency_maps'] = False
args['load_size'] = 256

# Helper functions
def load_rgb(path):
    """ Loading RGB image with OpenCV
    : param path: string, image path name. Must point to a file.
    """
    assert os.path.isfile(path), f"Path {path} doesn't exist"
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

def viz_correspondence(im_a, im_b, parts_a, parts_b):
    """ Visualizes the correspondences
    : param im_a: np.ndarray, RGB image a
    : param im_b: np.ndarray, RGB image b
    : param parts_a: List[np.ndarray], list of part masks in a
    : param parts_b: List[np.ndarray], list of part masks in b
    """
    quer_img = im_a.astype(np.uint8)
    corr_img = im_b.astype(np.uint8)
    for i, part_i in enumerate(parts_a):
        quer_img = overlay_segment(quer_img, part_i,
                                  COLORS[i], alpha=0.3)
        part_out_i = resize(parts_b[i],corr_img.shape[:2]) > 0
        corr_img = overlay_segment(corr_img, part_out_i,
                                  COLORS[i], alpha=0.3)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(quer_img)
    ax[0].set_title('Support', fontsize=16)
    ax[1].imshow(corr_img)
    ax[1].set_title('Target', fontsize=16)
    plt.show()


def generate_affordance_mask(parts_out, shape):
    """ Generates a unique 2D affordance mask from parts_out
    : param parts_out: List[np.ndarray], list of part masks in the target image
    : param shape: tuple, shape of the target image
    : return: np.ndarray, affordance mask
    """
    affordance_mask = np.zeros(shape, dtype=np.int32)
    for i, part in enumerate(parts_out):
        resized_part = resize(part, shape) > 0
        affordance_mask[resized_part] = i + 1
    return affordance_mask

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Show part query correspondence in single object images with AffCorrs')
    parser.add_argument('--reference_dir_path', type=str, required=True, help='Path to the reference image')
    parser.add_argument('--target_image_path', type=str, required=True, help='Path to the target image')
    parser.add_argument('--output_dir_path', type=str, required=True, help='Directory to save the output affordance')
    
    args_cli = parser.parse_args()

    # Redefine TARGET_IMAGE_PATH from command line argument
    TARGET_IMAGE_PATH = args_cli.target_image_path
    SUPPORT_DIR = args_cli.reference_dir_path


    # The models are ran with no_grad since they are 
    # unsupervised. This preserves GPU memory
    with torch.no_grad():
        model = AffCorrs_V1(args)

        # Prepare inputs
        img1_path = f"{SUPPORT_DIR}/prototype.png"
        aff1_path = f"{SUPPORT_DIR}/affordance.npy"
        rgb_a = load_rgb(img1_path)
        parts = np.load(aff1_path, allow_pickle=True).item()['masks']
        affordances = [None for _ in parts]
        rgb_b = load_rgb(TARGET_IMAGE_PATH)

        ## Produce correspondence
        model.set_source(Image.fromarray(rgb_a), parts, affordances)
        model.generate_source_clusters()

        model.set_target(Image.fromarray(rgb_b))
        model.generate_target_clusters()

        parts_out, aff_out = model.find_correspondences()

        ## Display correspondence
        viz_correspondence(rgb_a, rgb_b, parts, parts_out)

        # Generate affordance mask
        affordance_mask = generate_affordance_mask(parts_out, rgb_b.shape[:2])

        # Save affordance mask
        output_path = os.path.join(args_cli.output_dir_path, 'affordance_mask.npy')
        np.save(output_path, affordance_mask)