# download.py
import subprocess
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

KERNEL_ID = "leventecsibi/stable-diffusion-batch-generator"

def run(dest="output_images", kernel_id=None):
    """Download output images from Kaggle kernel"""
    kernel_id = kernel_id or KERNEL_ID
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Downloading output from kernel {kernel_id} to {dest_path}")
    
    kaggle_cmd = _get_kaggle_command()
    result = subprocess.run([
        *kaggle_cmd, "kernels", "output",
        kernel_id,
        "-p", str(dest_path)
    ], check=True, capture_output=True, text=True, encoding='utf-8')
    
    # Log the output safely
    if result.stdout:
        logging.debug(f"Kaggle output: {result.stdout}")
    if result.stderr:
        logging.debug(f"Kaggle stderr: {result.stderr}")
    
    logging.info("Download completed")


def _get_kaggle_command():
    """
    Get the appropriate command to run Kaggle CLI.
    
    Returns:
        list: Command parts to execute kaggle CLI
    """
    # Check if poetry is available and we're in a poetry project
    if shutil.which("poetry") and Path("pyproject.toml").exists():
        try:
            # Test if poetry can run the kaggle command
            result = subprocess.run(
                ["poetry", "run", "python", "-c", "import kaggle"],
                capture_output=True,
                check=False
            )
            if result.returncode == 0:
                return ["poetry", "run", "python", "-m", "kaggle.cli"]
        except Exception:
            pass
    
    # Fallback to direct python call
    return ["python", "-m", "kaggle.cli"]
