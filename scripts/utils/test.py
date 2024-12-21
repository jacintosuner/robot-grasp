import subprocess

# Define the input and output file paths
input_file = '/home/jacinto/robot-grasp/data/mask_references/reference_20241218_214159/demo.mkv'
output_file = "/home/jacinto/robot-grasp/data/mask_references/reference_20241218_214159/calibration.json"

# Create the ffmpeg command
ffmpeg_command = [
    "ffmpeg", 
    "-dump_attachment:t:0", 
    output_file, 
    "-i", 
    input_file,
]

# Run the ffmpeg command
try:
    subprocess.run(ffmpeg_command, check=True)
    print(f"Attachment extracted successfully to {output_file}")
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
