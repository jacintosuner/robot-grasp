import numpy as np
import argparse
import open3d as o3d
import cv2
import os

HAND_KEYPOINTS = [
    47, 48, 66, 67, 68, 70, 71, 72, 73, 74, 
    77, 93, 94, 142, 146, 147, 148, 151, 152, 
    157, 165, 166, 194, 268, 271, 275, 278, 
    280, 328, 340, 341, 342, 343, 356, 357, 
    358, 359, 370, 375, 376, 378, 379, 392, 
    402, 438, 440, 452, 453, 454, 455, 468, 
    469, 470, 485, 513, 566, 570, 572, 573, 
    683, 690
]
PAD_PERCENT = 2.5
DISTANCE_THRESHOLD = 0.02

def load_obj(file_path, scale=1000.0):
    """Load OBJ file and return pointcloud with selected keypoints"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    points = []
    colors = []
    for idx, line in enumerate(lines):
        if line.startswith('v ') and idx in HAND_KEYPOINTS:
            parts = line.strip().split()
            x, y, z = map(float, parts[1:4])
            r, g, b = map(float, parts[4:7]) if len(parts) > 6 else (0.5, 0.5, 0.5)
            points.append([x, y, z])
            colors.append([r, g, b])

    points = np.array(points, dtype=np.float32) / scale
    colors = np.array(colors, dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def load_npy(file_path, scale=1000.0):
    """Load NPY file containing RGB-D-K data and return pointcloud"""
    data = np.load(file_path, allow_pickle=True).item()
    colors = np.array(data.get('rgb'), dtype=np.float32) / 255.0
    depths = np.array(data.get('depth'))
    K = data.get('K')
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Generate pointcloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # Create mask and points
    mask = np.ones_like(points_z, dtype=bool)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)

    # Create Open3D pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def color_points_near_hand(hand_pcd, scene_pcd):
    hand_points = np.asarray(hand_pcd.points)
    scene_points = np.asarray(scene_pcd.points)
    scene_colors = np.asarray(scene_pcd.colors)

    for hand_point in hand_points:
        distances = np.linalg.norm(scene_points - hand_point, axis=1)
        mask = distances < DISTANCE_THRESHOLD
        scene_colors[mask] = [1.0, 0.0, 0.0]  # Set color to red

    scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors)

def save_padded_mask_and_overlay(mask, image, output_dir='./', pad_percent=0.3):
    """Save padded mask, overlay, and original cropped image"""
    # Find mask boundaries
    rows, cols = np.where(mask)
    if len(rows) == 0 or len(cols) == 0:
        print("No mask area found")
        return
        
    # Calculate boundaries
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    # Calculate padding
    height = max_row - min_row
    width = max_col - min_col
    pad_h = int(height * pad_percent)
    pad_w = int(width * pad_percent)
    
    # Calculate new boundaries with padding
    new_min_row = max(0, min_row - pad_h)
    new_max_row = min(image.shape[0], max_row + pad_h)
    new_min_col = max(0, min_col - pad_w)
    new_max_col = min(image.shape[1], max_col + pad_w)
    
    # Crop mask and images
    cropped_mask = mask[new_min_row:new_max_row, new_min_col:new_max_col]
    cropped_original = image[new_min_row:new_max_row, new_min_col:new_max_col].copy()
    
    # Create overlay on cropped region
    cropped_overlay = cropped_original.copy()
    cropped_mask_region = mask[new_min_row:new_max_row, new_min_col:new_max_col]
    cropped_overlay[cropped_mask_region] = [0, 0, 255]  # Red color for mask
    cropped_overlay = cv2.addWeighted(cropped_overlay, 0.5, cropped_original, 0.5, 0)
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'affordance.npy'), cropped_mask)
    cv2.imwrite(os.path.join(output_dir, 'annotation.png'), cropped_overlay)
    cv2.imwrite(os.path.join(output_dir, 'prototype.png'), cropped_original)

def main(hand_file_path, initial_scene_file_path, output_dir):
    # Load both pointclouds
    hand_pcd = load_obj(hand_file_path)
    scene_pcd = load_npy(initial_scene_file_path)

    # Reduce the opacity of the colors in the scene pointcloud
    scene_colors = np.asarray(scene_pcd.colors)
    scene_colors *= 0.5  # Reduce opacity by 50%
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors)

    # Apply the function to color points near the hand
    color_points_near_hand(hand_pcd, scene_pcd)
    
    # Create coordinate frame for reference
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    
    # Visualize both pointclouds
    o3d.visualization.draw_geometries(
        [camera_frame, hand_pcd, scene_pcd],
        lookat=scene_pcd.get_center(),
        up=np.array([0.0, -1.0, 0.0]),
        front=-scene_pcd.get_center(),
        zoom=1
    )

    # Load the original image
    data = np.load(initial_scene_file_path, allow_pickle=True).item()
    image = data.get('rgb')

    # Create a mask for the points near the hand
    mask = np.zeros(image.shape[:2], dtype=bool)
    scene_points = np.asarray(scene_pcd.points)
    hand_points = np.asarray(hand_pcd.points)
    for hand_point in hand_points:
        distances = np.linalg.norm(scene_points - hand_point, axis=1)
        mask |= (distances < DISTANCE_THRESHOLD).reshape(mask.shape)

    # Overlay the mask on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    save_padded_mask_and_overlay(mask, image, output_dir, pad_percent=PAD_PERCENT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an OBJ file and an NPY file.")
    parser.add_argument('--hand', required=True, help="Path to the OBJ file")
    parser.add_argument('--initial_scene', required=True, help="Path to the NPY file")
    parser.add_argument('--output_dir', default='./', help="Directory to save the output files")
    args = parser.parse_args()
    main(args.hand, args.initial_scene, args.output_dir)