# download.py
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

KERNEL_ID = "leventecsibi/stable-diffusion-batch-generator"

def run(dest="output_images", kernel_id=None):
    """Download output images from Kaggle kernel"""
    kernel_id = kernel_id or KERNEL_ID
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Downloading output from kernel {kernel_id} to {dest_path}")
    
    subprocess.run([
        "python", "-m", "kaggle.cli", "kernels", "output",
        kernel_id,
        "-p", str(dest_path)
    ], check=True)
    
    logging.info("Download completed")
