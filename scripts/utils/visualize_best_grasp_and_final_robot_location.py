import argparse
import numpy as np
import open3d as o3d
import os

# Example usage: python visualize_best_grasp_and_final_robot_location.py --rgbd_file ~/robot-grasp/data/rgbdks/robot_close_mug/rgbdk.npy --grasps_file ~/robot-grasp/data/contact_graspnet_pipeline_results_20241126_110632/contact_graspnet_results.npz

def load_rgbd_pointcloud(file_path, scale=1000.0):
    """Load RGB-D data and convert to pointcloud"""
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

def load_best_grasp(grasps_file):
    """Load best grasp from contact graspnet results"""
    data = np.load(grasps_file, allow_pickle=True)
    scores = data['scores'].item()
    pred_grasps_cam = data['pred_grasps_cam'].item()
    
    # Get grasp with highest score
    best_grasp = pred_grasps_cam[1][np.argmax(scores[1])]
    return best_grasp

def create_grasp_frame(grasp_pose, size=0.1):
    """Create coordinate frame for grasp pose"""
    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    grasp_frame.transform(grasp_pose)
    return grasp_frame

def main():
    parser = argparse.ArgumentParser(description='Visualize pointcloud and best grasp')
    parser.add_argument('--rgbd_file', required=True, help='Path to RGB-D data file')
    parser.add_argument('--grasps_file', required=True, help='Path to grasps results file')
    args = parser.parse_args()

    # Load camera calibration
    CAMERA_CALIBRATION_FILE = os.path.expanduser('~/robot-grasp/data/camera_calibration/camera2robot.npz')
    T_world_to_cam = np.load(CAMERA_CALIBRATION_FILE, allow_pickle=True)
    T_cam_to_world = np.linalg.inv(T_world_to_cam)
    
    # Load pointcloud
    pcd = load_rgbd_pointcloud(args.rgbd_file)
    
    # Load and transform best grasp
    best_grasp_cam = load_best_grasp(args.grasps_file)

    # Create coordinate frames
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    world_frame.transform(T_cam_to_world)
    
    # Create grasp frames in both references
    grasp_frame_cam = create_grasp_frame(best_grasp_cam)

    # Visualize everything
    o3d.visualization.draw_geometries(
        [camera_frame, world_frame, grasp_frame_cam, pcd],
        lookat=pcd.get_center(),
        up=np.array([0.0, -1.0, 0.0]),
        front=-pcd.get_center(),
        zoom=1
    )

if __name__ == '__main__':
    main()