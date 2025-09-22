# download.py
import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--kernel_id", type=str,
                    default="leventecsibi/stable-diffusion-batch-generator",
                    help="Kaggle kernel ID to fetch output from")
parser.add_argument("--dest", type=str, default="output_images",
                    help="Local folder to save output images")
args = parser.parse_args()

dest_path = Path(args.dest)
dest_path.mkdir(parents=True, exist_ok=True)

subprocess.run([
    "python", "-m", "kaggle.cli", "kernels", "output",
    args.kernel_id,
    "-p", str(dest_path)
])
