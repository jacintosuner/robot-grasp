import argparse
import numpy as np
import open3d as o3d
import os

parser = argparse.ArgumentParser()
parser.add_argument('rgbdk_file_paths', nargs='+', help='Paths to the RGB-D + K (camera matrix) data files')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()

# Example
# python3 visualize_rgbdk.py ~/robot-grasp/data/rgbdks/rgbdk.npy

def visualize_rgbdk(rgbdk_file_paths):
    combined_pcd = o3d.geometry.PointCloud()

    for rgbdk_file_path in rgbdk_file_paths:
        # Load the RGB-DK numpy file
        data = np.load(rgbdk_file_path, allow_pickle=True)
        colors = np.array(data.item().get('rgb'), dtype=np.float32) / 255.0
        depths = np.array(data.item().get('depth'))
        fx, fy = data.item().get('K')[0, 0], data.item().get('K')[1, 1]
        cx, cy = data.item().get('K')[0, 2], data.item().get('K')[1, 2]

        # get point cloud
        scale = 1000.0
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        # set your workspace to crop point cloud
        mask = np.ones_like(points_z, dtype=bool)
        # mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Combine point clouds
        combined_pcd += pcd
    
    # Create coordinate frame for reference
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    # Load the camera calibration transformation matrix
    CAMERA_CALIBRATION_FILE = os.path.expanduser('~/robot-grasp/data/camera_calibration/robot2camera.npz')
    T_world_to_cam = np.load(CAMERA_CALIBRATION_FILE, allow_pickle=True)
    T_cam_to_world = np.linalg.inv(T_world_to_cam)

    # Create world frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    world_frame.transform(T_cam_to_world)

    # Add reference frames to the visualization
    o3d.visualization.draw_geometries(
        [camera_frame, world_frame, combined_pcd],
        lookat=combined_pcd.get_center(),
        up=np.array([0.0, -1.0, 0.0]),
        front=-combined_pcd.get_center(),
        zoom=1
    )


if __name__ == '__main__':
    visualize_rgbdk(cfgs.rgbdk_file_paths)