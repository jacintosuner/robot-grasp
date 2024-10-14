import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_rgbdk(output_data):
    rgb_data = output_data['rgb']
    depth_data = output_data['depth']
    camera_matrix = output_data['K']

    # Display RGB image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB))
    plt.title('RGB Image')
    plt.axis('off')

    # Display Depth image
    plt.subplot(1, 2, 2)
    plt.imshow(depth_data, cmap='gray')
    plt.title('Depth Image')
    plt.axis('off')

    plt.show()

    # Print Camera Matrix
    print("Camera Matrix (K):")
    print(camera_matrix)

# Example usage
if __name__ == "__main__":
    # Load data from npy file
    data_path = '/home/jacinto/robot-grasp/data/rgbdks/rgbdk_data.npy'
    loaded_data = np.load(data_path, allow_pickle=True).item()

    rgb_data = loaded_data['rgb']
    depth_data = loaded_data['depth']
    camera_matrix = loaded_data['K']

    output_data = {
        'rgb': rgb_data,
        'depth': depth_data,
        'K': camera_matrix
    }

    visualize_rgbdk(output_data)