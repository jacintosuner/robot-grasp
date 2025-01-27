import numpy as np
import argparse
import trimesh
import os
import json
from pycocotools import mask as maskUtils
import open3d as o3d


# Example usage: python taxposed_train_data_processing.py --dir_path ~/robot-grasp/data/demos/demos_20241230_173916/13 --object_name mug


def kabsch_numpy(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = np.dot(p.T, q)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # RMSD
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])

    return R, t, centroid_P, rmsd


def load_obj(file_path):
        mesh = trimesh.load(file_path, file_type='obj')
        return mesh.vertices[:, :3]

def load_npy_file(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    rgb = data['rgb']
    depth = data['depth']
    K = data['K']
    return rgb, depth, K

def depth_to_point_cloud(depth, K):
    h, w = depth.shape
    i, j = np.indices((h, w))
    z = depth
    x = (j - K[0, 2]) * z / K[0, 0]
    y = (i - K[1, 2]) * z / K[1, 1]
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points

def transform_point_cloud(point_cloud, R, t, centering):
    return np.dot(point_cloud - centering, R.T) + centering + t

def visualize_hand_poses(initial_grasping_scene, final_grasping_scene, R, t, centering):
    # Create Open3D point clouds
    initial_pcd = o3d.geometry.PointCloud()
    initial_pcd.points = o3d.utility.Vector3dVector(initial_grasping_scene)
    initial_pcd.paint_uniform_color([1, 0, 0])  # Red for initial hand pose
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(final_grasping_scene)
    final_pcd.paint_uniform_color([0, 1, 0])  # Green for final hand pose

    # Transform the initial hand pose
    transformed_initial_pcd = o3d.geometry.PointCloud()
    transformed_initial_pcd.points = o3d.utility.Vector3dVector(transform_point_cloud(initial_grasping_scene, R, t, centering))
    transformed_initial_pcd.paint_uniform_color([0, 0, 1])  # Blue for transformed initial hand pose

    # Visualize the hand poses
    o3d.visualization.draw_geometries([initial_pcd, final_pcd, transformed_initial_pcd])

    hand_initial_pcd = initial_pcd
    hand_final_pcd = final_pcd
    hand_transformed_initial_pcd = transformed_initial_pcd
    return hand_initial_pcd, hand_final_pcd, hand_transformed_initial_pcd


def main(dir_path, object_name):

    # Find the rigid transformation from the initial hand pose to the final hand pose
    initial_grasping_scene = load_obj(f"{dir_path}/initial_grasping_scene_0.obj")
    final_grasping_scene = load_obj(f"{dir_path}/final_grasping_scene_0.obj")

    R, t, centering, rmsd = kabsch_numpy(initial_grasping_scene, final_grasping_scene)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", t)
    print("RMSD:\n", rmsd)

    ## Visualize the final hand pose and the transformed initial hand pose
    hand_initial_pcd, hand_final_pcd, hand_transformed_initial_pcd = visualize_hand_poses(initial_grasping_scene, final_grasping_scene, R, t, centering)

    # Transform the "action points" (the object point cloud) using the hand transformation
    ## Extract the segmentation mask from grounded sam results
    json_file_path = os.path.join(dir_path, f'grounded_sam_seg_{object_name}.json')
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    rle = json_data['annotations'][0]['segmentation']
    seg_data = maskUtils.decode(rle)
    
    ## Extract the point cloud
    npy_file_path = os.path.join(dir_path, f'initial_scene.npy')
    rgb, depth, K = load_npy_file(npy_file_path)
    point_cloud = depth_to_point_cloud(depth, K)
    
    ## Find the point cloud object mask, i.e. a numpy array with 1 if the point is part of the object, 0 otherwise
    seg_data = seg_data.flatten()
    
    ## Apply the transformation to the object point cloud
    object_points = point_cloud[seg_data == 1]
    transformed_object_points = transform_point_cloud(object_points, R, t, centering)

    ## Visualize the original and transformed object point clouds
    ### Assign blue color to transformed object points
    transformed_colors = np.zeros_like(transformed_object_points)
    transformed_colors[:] = [0, 0, 1]  # Blue for transformed object points
    ### Assign colors based on segmentation mask
    colors = np.zeros_like(point_cloud)
    colors[seg_data == 0] = [0, 0, 0]  # Black for background
    colors[seg_data == 1] = [1, 0, 0]  # Green for object
    ### Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    ### Create Open3D point cloud for transformed object points
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_object_points)
    transformed_pcd.colors = o3d.utility.Vector3dVector(transformed_colors)
    ### Visualize the original and transformed point clouds together
    o3d.visualization.draw_geometries([pcd, transformed_pcd, hand_initial_pcd, hand_final_pcd, hand_transformed_initial_pcd])
    # o3d.visualization.draw_geometries([pcd, transformed_pcd])

    
    ## Change the object point cloud in the original point cloud to the transformed object point cloud
    point_cloud[seg_data == 1] = transformed_object_points 

    ## Save the taxposed ready file
    final_dir_name = os.path.basename(os.path.normpath(dir_path))
    print(final_dir_name)
    np.savez(
        os.path.join(dir_path, f"{final_dir_name}_teleport_obj_points.npz"),
        clouds=point_cloud,
        masks=seg_data-1, # taxpose: -1 for environment, 0,1,... for object masks
        classes=1-seg_data, # taxpose: 0 for object, 1 for background
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--dir_path', type=str, help='Path to the directory')
    parser.add_argument('--object_name', type=str, help='Name of the object')
    args = parser.parse_args()
    main(args.dir_path, args.object_name)