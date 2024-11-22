from pathlib import Path
from typing import Optional, Union
from pyk4a import PyK4ARecord, K4AException, PyK4A, Config
import time
import argparse
import subprocess
import numpy as np  # Required for depth image normalization

# Example usage: python capture_video.py --duration 4 --output_path ../../data/videos/output.mkv --stream_type color

class VideoCapture:
    def __init__(self, output_path: Union[str, Path], config: Config, device: Optional[PyK4A] = None):
        self.output_path = Path(output_path)
        self.record = PyK4ARecord(self.output_path, config, device)
        self.device = device

    def start(self):
        self.record.create()

    def stop(self):
        self.record.close()

    def capture_frame(self):
        if not self.device:
            raise K4AException("Device not initialized.")
        
        capture = self.device.get_capture()
        self.record.write_capture(capture)
        self.record.write_capture(capture)

    def flush(self):
        self.record.flush()

    def convert_to_mp4(self, stream_type: str):
        if stream_type == 'color':
            mp4_path = self.output_path.with_name('color.mp4')
            subprocess.run([
                'ffmpeg', '-i', str(self.output_path), '-map', '0:0',
                '-c:v', 'libx264', str(mp4_path)
            ], check=True)
        elif stream_type == 'depth':
            mp4_path = self.output_path.with_name('depth.mp4')
            subprocess.run([
                'ffmpeg', '-i', str(self.output_path), '-map', '0:1',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-vf', 'scale=640:576,format=gray,eq=brightness=0.1',
                '-crf', '23', '-preset', 'medium', str(mp4_path)
            ], check=True)
        else:
            raise ValueError(f"Invalid stream type: {stream_type}. Choose 'color' or 'depth'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture video from a device.")
    parser.add_argument("--duration", type=int, help="Duration of the video capture in seconds.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video.")
    parser.add_argument("--stream_type", type=str, choices=['color', 'depth'], default='color', help="Stream type to export (color or depth).")
    args = parser.parse_args()

    config = Config()  # Initialize with appropriate configuration
    device = PyK4A(config)
    device.start()

    video_capture = VideoCapture(args.output_path, config, device)
    video_capture.start()

    start_time = time.time()
    try:
        while time.time() - start_time < args.duration:
            video_capture.capture_frame()
    except KeyboardInterrupt:
        pass
    finally:
        video_capture.flush()
        video_capture.stop()
        device.stop()
        video_capture.convert_to_mp4(args.stream_type)
