import numpy as np
import argparse
import open3d as o3d

# example usage: python3 hand_visualizer.py --hand ~/robot-grasp/data/mask_references/carton_milk_reference_20241203/hand_frames/hand_grasping_0.obj

HAND_KEYPOINTS = [
    48, 49, 67, 68, 69, 70, 71, 72, 73, 74, 75, 
    78, 95, 142, 143, 147, 148, 149, 152, 153, 
    158, 166, 167, 195, 269, 272, 276, 281, 289,
    329, 341, 342, 343, 344, 357, 358, 
    359, 360, 371, 376, 377, 379, 380, 387, 388, 393, 
    403, 439, 441, 453, 454, 455, 456, 469, 
    470, 471, 472, 486, 487, 490, 497, 498, 514, 567, 571, 573, 574, 
    684, 691, 756, 757, 764
]

def load_obj_with_indices(file_path, scale=1000.0):
    """Load OBJ file and return pointcloud with all vertices"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    points = []
    colors = []
    vertex_indices = []
    
    for idx, line in enumerate(lines):
        if line.startswith('v '):
            parts = line.strip().split()
            x, y, z = map(float, parts[1:4])
            if idx in HAND_KEYPOINTS:
                r, g, b = 1.0, 0.0, 0.0
            else:
                r, g, b = map(float, parts[4:7]) if len(parts) > 6 else (0.5, 0.5, 0.5)
            points.append([x, y, z])
            colors.append([r, g, b])
            vertex_indices.append(idx)

    points = np.array(points, dtype=np.float32) / scale
    colors = np.array(colors, dtype=np.float32)
    
    return points, colors, vertex_indices

def custom_draw_geometry_with_picking(pcd):
    """Interactive visualization that shows vertex index when clicked"""
    # Create visualization window with editing enabled
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    
    # Add geometry
    vis.add_geometry(pcd)
    
    # Set up the visualization
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    
    # Get view control
    view_control = vis.get_view_control()
    view_control.set_up([0, -1, 0])
    view_control.set_front([0, 0, -1])
    view_control.set_lookat(pcd.get_center())
    view_control.set_zoom(0.8)
    
    print("Shift + Left Click to pick points")
    print("Press 'Q' to exit and display picked points")
    
    # Run picking interface
    picked_indices = vis.run()  # Returns indices of picked points
    
    # Print picked points info
    points = np.asarray(pcd.points)
    for idx in picked_indices:
        point = points[idx]
        print(f"Vertex {idx}: position {point}")
        
    vis.destroy_window()
    return picked_indices

def main(hand_file_path):
    # Load pointcloud with indices
    points, colors, indices = load_obj_with_indices(hand_file_path)
    
    # Create main pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Apply rotation to match visualization
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    pcd.rotate(R, center=(0, 0, 0))
    
    # Visualize with picking functionality
    custom_draw_geometry_with_picking(pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hand mesh vertices with indices")
    parser.add_argument('--hand', required=True, help="Path to the OBJ file")
    
    args = parser.parse_args()
    main(args.hand)