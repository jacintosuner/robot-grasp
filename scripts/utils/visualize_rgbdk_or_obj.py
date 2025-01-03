# from frankapy import FrankaArm
import numpy as np
import open3d as o3d
import os

# example usage: python3 visualize_rgbdk_or_obj.py ~/robot-grasp/scripts/utils/tests/rgbdk.npy

def visualize_rgbdk_with_eef_pose(file_paths, eef_pose):
    
    combined_pcd = o3d.geometry.PointCloud()

    for file_path in file_paths:
        scale = 1000.0  # Scale factor for the depth values
        if file_path.endswith('.npy'):
            # Load the RGB-DK numpy file
            data = np.load(file_path, allow_pickle=True)
            colors = np.array(data.item().get('rgb'), dtype=np.float32) / 255.0
            depths = np.array(data.item().get('depth'))
            fx, fy = data.item().get('K')[0, 0], data.item().get('K')[1, 1]
            cx, cy = data.item().get('K')[0, 2], data.item().get('K')[1, 2]

            # get point cloud
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
        
        elif file_path.endswith('.obj'):
            # Load the OBJ file
            with open(file_path, 'r') as f:
                lines = f.readlines()

            points = []
            colors = []
            for line in lines:
                if line.startswith('v '):
                    parts = line.strip().split()
                    x, y, z = map(float, parts[1:4])
                    r, g, b = map(float, parts[4:7])
                    points.append([x, y, z])
                    colors.append([r, g, b])

            points = np.array(points, dtype=np.float32) / scale
            colors = np.array(colors, dtype=np.float32)

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Combine point clouds
        combined_pcd += pcd

    # Create coordinate frame for reference
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    # Load the camera calibration transformation matrix
    CAMERA_CALIBRATION_FILE = os.path.expanduser('~/robot-grasp/data/camera_calibration/camera2robot.npz')
    T_cam_to_world = np.load(CAMERA_CALIBRATION_FILE, allow_pickle=True)
    T_world_to_cam = np.linalg.inv(T_cam_to_world)

    # Transform EEF_POSE from robot frame to camera frame
    eef_pose_cam_frame = T_world_to_cam @ eef_pose

    # Create EEF pose frame
    eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    eef_frame.transform(eef_pose_cam_frame)

    # Create world frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    world_frame.transform(T_world_to_cam)

    # Add reference frames to the visualization
    o3d.visualization.draw_geometries(
        [camera_frame, world_frame, combined_pcd, eef_frame],
        lookat=combined_pcd.get_center(),
        up=np.array([0.0, -1.0, 0.0]),
        front=-combined_pcd.get_center(),
        zoom=1
    )

def visualize_rgbdk(file_paths):
    combined_pcd = o3d.geometry.PointCloud()

    for file_path in file_paths:
        scale = 1000.0  # Scale factor for the depth values
        if file_path.endswith('.npy'):
            # Load the RGB-DK numpy file
            data = np.load(file_path, allow_pickle=True)
            colors = np.array(data.item().get('rgb'), dtype=np.float32) / 255.0
            depths = np.array(data.item().get('depth'))
            fx, fy = data.item().get('K')[0, 0], data.item().get('K')[1, 1]
            cx, cy = data.item().get('K')[0, 2], data.item().get('K')[1, 2]

            # get point cloud
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
        
        elif file_path.endswith('.obj'):
            # Load the OBJ file
            with open(file_path, 'r') as f:
                lines = f.readlines()

            points = []
            colors = []
            for line in lines:
                if line.startswith('v '):
                    parts = line.strip().split()
                    x, y, z = map(float, parts[1:4])
                    r, g, b = map(float, parts[4:7])
                    points.append([x, y, z])
                    colors.append([r, g, b])

            points = np.array(points, dtype=np.float32) / scale
            colors = np.array(colors, dtype=np.float32)

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Combine point clouds
        combined_pcd += pcd

    # Create coordinate frame for reference
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    # Load the camera calibration transformation matrix
    CAMERA_CALIBRATION_FILE = os.path.expanduser('~/robot-grasp/data/camera_calibration/camera2robot.npz')
    T_cam_to_world = np.load(CAMERA_CALIBRATION_FILE, allow_pickle=True)
    T_world_to_cam = np.linalg.inv(T_cam_to_world)


    # Create world frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    world_frame.transform(T_world_to_cam)

    # Add reference frames to the visualization
    o3d.visualization.draw_geometries(
        [camera_frame, world_frame, combined_pcd],
        lookat=combined_pcd.get_center(),
        up=np.array([0.0, -1.0, 0.0]),
        front=-combined_pcd.get_center(),
        zoom=1
    )


if __name__ == '__main__':
    # print('Starting robot')
    # fa = FrankaArm()
    # EEF_POSE = np.array(fa._state_client._get_current_robot_state().robot_state.O_T_EE).reshape(4, 4).transpose()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', nargs='+', help='Paths to the RGB-D + K (camera matrix) data files')
    cfgs = parser.parse_args()

    # visualize_rgbdk_with_eef_pose(cfgs.file_paths, EEF_POSE)
    visualize_rgbdk(cfgs.file_paths)