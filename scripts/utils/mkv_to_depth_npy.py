import numpy as np
from pyk4a import PyK4APlayback
import argparse

def extract_depths_from_mkv(mkv_file_path, output_npy_path):
    playback = PyK4APlayback(mkv_file_path)
    playback.open()

    depth_frames = []
    while True:
        try:
            capture = playback.get_next_capture()
            if capture.transformed_depth is not None:
                depth_frames.append(capture.transformed_depth / 1000.0)
        except EOFError:
            break

    playback.close()

    np.save(output_npy_path, np.array(depth_frames))

def main():
    parser = argparse.ArgumentParser(description="Extract depth frames from an MKV file and save them as a numpy array.")
    parser.add_argument("--mkv_file_path", type=str, help="Path to the input MKV file.")
    parser.add_argument("--output_npy_path", type=str, help="Path to the output numpy file.")
    args = parser.parse_args()


    extract_depths_from_mkv(args.mkv_file_path, args.output_npy_path)

if __name__ == "__main__":
    main()