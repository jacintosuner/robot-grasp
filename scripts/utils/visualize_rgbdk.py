import numpy as np
import matplotlib.pyplot as plt

def visualize_point_cloud(bgr_image, depth_image, camera_matrix):
    # Get the height and width of the images
    height, width, _ = bgr_image.shape

    # Create a meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the coordinates and depth image
    x = x.flatten()
    y = y.flatten()
    depth = depth_image.flatten()

    # Filter out points with zero depth
    valid = depth > 0
    x_valid = x[valid]
    y_valid = y[valid]
    depth_valid = depth[valid]

    # Convert depth to meters for visualization
    depth_meters = depth_valid / 1000.0  # if depth is in mm

    # Use camera matrix to get 3D coordinates
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Compute 3D points in camera coordinates
    X = (x_valid - cx) * depth_meters / fx
    Y = (y_valid - cy) * depth_meters / fy
    Z = depth_meters

    # Stack the 3D points
    points = np.vstack((X, Y, Z)).T

    # Get color values from the BGR image (invert the channels for RGB)
    colors = bgr_image[y_valid, x_valid] / 255.0  # Normalize to [0, 1]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud
    ax.scatter(X, Y, Z, c=colors, s=0.1)  # You can adjust the point size

    # Set labels
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')

    # Set the aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    # Plot the camera position as a point (optional)
    ax.scatter(0, 0, 0, c='r', marker='o', s=100)  # Camera reference point

    plt.show()

# Example usage
if __name__ == "__main__":
    # Load your data
    input_npy_path = "/home/jacinto/robot-grasp/data/rgbds/processed_data.npy"
    data_dict = np.load(input_npy_path, allow_pickle=True).item()

    bgr_image = data_dict["bgr"]
    depth_image = data_dict["depth"]
    camera_matrix = data_dict["K"]

    visualize_point_cloud(bgr_image, depth_image, camera_matrix)
