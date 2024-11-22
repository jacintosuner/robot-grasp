import os
import numpy as np
import imageio
from pyk4a import PyK4A, Config
from pyk4a.calibration import CalibrationType

# Use
# python3 capture_rgbdk.py --output_path ../../data/rgbdks/
# python3 capture_rgbdk.py --output_path ../../data/rgbdks/ --to_images
# rsync -r --progress ~/robot-grasp/data/rgbdks/rgbdk.npy jacinto@ham1.pc.cs.cmu.edu:/home/jacinto/robot-grasp/data/rgbdks

def capture_rgbd(output_path, to_images, name):
    # Create a output_path if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Initialize the Azure Kinect camera with the desired configuration
    k4a = PyK4A()  # Adjust config as needed
    k4a.start()

    try:
        # Capture a single frame
        capture = k4a.get_capture()

        if capture.color is not None and capture.depth is not None:
            # Convert depth to millimeters and ensure it's uint16
            depth_image = capture.depth.astype(np.uint16)  # Keep depth in mm
            
            if to_images:
                # Save RGB and Depth images as PNG files
                rgb_path = os.path.join(output_path, "rgb_frame.png")
                depth_path = os.path.join(output_path, "depth_frame.png")

                # Save images
                imageio.imwrite(rgb_path, capture.color[:, :, :3][:, :, ::-1])
                imageio.imwrite(depth_path, depth_image)

                print(f"Saved RGB image to {rgb_path}")
                print(f"Saved Depth image to {depth_path}")
            else:
                # Resize depth image to match RGB image dimensions
                
                # Prepare data dictionary
                data_dict = {
                    "rgb": capture.color[:, :, :3][:, :, ::-1],  # Convert BGR to RGB
                    "depth": capture.transformed_depth,
                    "K": k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
                }

                # Save the dictionary as a .npy file
                npy_path = os.path.join(output_path, f"{name}.npy")
                np.save(npy_path, data_dict)

                print(f"Saved RGBD data to {npy_path}")
                
        else:
            print("Failed to capture RGBD frame.")

    finally:
        k4a.stop()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Capture RGBD data using Azure Kinect.')
    parser.add_argument('--output_path', type=str, help='Directory to save RGBD data')
    parser.add_argument('--to_images', action='store_true', help='Save RGB and Depth data as a single .npy file')
    parser.add_argument('--name', type=str, default='rgbdk', help='Base name for the saved files')

    args = parser.parse_args()
    capture_rgbd(args.output_path, args.to_images, args.name)
