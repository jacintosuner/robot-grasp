from pathlib import Path
import subprocess
import argparse
from typing import Union, Optional

class MKVToMP4Converter:
    def __init__(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else self.input_path.with_suffix('.mp4')

    def convert(self):
        subprocess.run([
            'ffmpeg', '-i', str(self.input_path),
            '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental',
            str(self.output_path)
        ], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MKV file to MP4.")
    parser.add_argument("--input_path", type=str, help="Path to the input MKV file.")
    parser.add_argument("--output_path", type=str, help="Path to save the output MP4 file.")
    args = parser.parse_args()

    converter = MKVToMP4Converter(args.input_path, args.output_path)
    converter.convert()