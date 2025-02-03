import numpy as np
import open3d as o3d
import argparse
import json
from pycocotools import mask as maskUtils

def load_data(np_path, gsam2_pred_path=None):
    data = np.load(np_path, allow_pickle=True).item()
    rgb = data['rgb']
    depth = data['depth']
    K = data['K']
    
    if gsam2_pred_path:
        with open(gsam2_pred_path, 'r') as f:
            segmentation_data = json.load(f)
        segmentation = maskUtils.decode(segmentation_data['annotations'][0]['segmentation'])
    else:
        segmentation = data['seg']
    
    return rgb, depth, K, segmentation

def create_point_cloud(rgb, depth, K, segmentation):
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    xv, yv = np.meshgrid(x, y)

    z = depth / 1000.0  # Convert depth to meters
    x = (xv - cx) * z / fx
    y = (yv - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0

    mask = segmentation.reshape(-1) > 0
    segmented_points = points[mask]
    segmented_colors = colors[mask]

    rest_points = points[~mask]
    rest_colors = colors[~mask]

    segmented_pcd = o3d.geometry.PointCloud()
    segmented_pcd.points = o3d.utility.Vector3dVector(segmented_points)
    segmented_pcd.colors = o3d.utility.Vector3dVector(segmented_colors)

    rest_pcd = o3d.geometry.PointCloud()
    rest_pcd.points = o3d.utility.Vector3dVector(rest_points)
    rest_pcd.colors = o3d.utility.Vector3dVector(rest_colors)

    return segmented_pcd, rest_pcd

def main(np_path, gsam2_pred_path=None):
    rgb, depth, K, segmentation = load_data(np_path, gsam2_pred_path)
    segmented_pcd, rest_pcd = create_point_cloud(rgb, depth, K, segmentation)

    # Apply the default transformation
    # default_transformation = np.eye(4)
    default_transformation = np.array([
        [ 5.69133091e-01, -5.88802046e-01,  5.73933512e-01, -7.20034118e-05],
        [ 8.04948581e-01,  5.41394404e-01, -2.42795966e-01,  1.39073387e-04],
        [-1.67765630e-01,  6.00170184e-01,  7.82080842e-01, -5.72023363e-05],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    ])
    segmented_pcd.transform(default_transformation)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(rest_pcd)
    vis.add_geometry(segmented_pcd)

    global transformation_matrix
    # Initialize the transformation matrix with the default transformation
    transformation_matrix = default_transformation.copy()

    def translate_pcd(pcd, translation):
        global transformation_matrix
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation
        transformation_matrix = translation_matrix @ transformation_matrix
        pcd.transform(translation_matrix)
        vis.update_geometry(pcd)

    def rotate_pcd(pcd, rotation):
        global transformation_matrix
        R = pcd.get_rotation_matrix_from_xyz(rotation)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = R
        transformation_matrix = rotation_matrix @ transformation_matrix
        pcd.transform(rotation_matrix)
        vis.update_geometry(pcd)

    vis.register_key_callback(ord("W"), lambda vis: translate_pcd(segmented_pcd, [0, 0, 0.00001]))
    vis.register_key_callback(ord("S"), lambda vis: translate_pcd(segmented_pcd, [0, 0, -0.00001]))
    vis.register_key_callback(ord("A"), lambda vis: translate_pcd(segmented_pcd, [-0.00001, 0, 0]))
    vis.register_key_callback(ord("D"), lambda vis: translate_pcd(segmented_pcd, [0.00001, 0, 0]))
    vis.register_key_callback(ord("Q"), lambda vis: translate_pcd(segmented_pcd, [0, 0.00001, 0]))
    vis.register_key_callback(ord("E"), lambda vis: translate_pcd(segmented_pcd, [0, -0.00001, 0]))

    vis.register_key_callback(ord("I"), lambda vis: rotate_pcd(segmented_pcd, [0.01, 0, 0]))
    vis.register_key_callback(ord("K"), lambda vis: rotate_pcd(segmented_pcd, [-0.01, 0, 0]))
    vis.register_key_callback(ord("J"), lambda vis: rotate_pcd(segmented_pcd, [0, 0.01, 0]))
    vis.register_key_callback(ord("L"), lambda vis: rotate_pcd(segmented_pcd, [0, -0.01, 0]))
    vis.register_key_callback(ord("U"), lambda vis: rotate_pcd(segmented_pcd, [0, 0, 0.01]))
    vis.register_key_callback(ord("O"), lambda vis: rotate_pcd(segmented_pcd, [0, 0, -0.01]))

    vis.run()
    vis.destroy_window()

    # Save the transformation matrix of the segmented point cloud
    np.save("segmented_transformation_matrix.npy", transformation_matrix)
    print(transformation_matrix)
    print("Transformation matrix saved to segmented_transformation_matrix.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually transform segmented object")
    parser.add_argument("--np_path", type=str, required=True, help="Path to the .npy file")
    parser.add_argument("--gsam2_pred_path", type=str, help="Path to the gsam2 prediction file")
    args = parser.parse_args()
    main(args.np_path, args.gsam2_pred_path)