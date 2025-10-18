# download.py
import subprocess
from pathlib import Path
import logging
import shutil
import os

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

KERNEL_ID = "leventecsibi/stable-diffusion-batch-generator"

def run(dest="output_images", kernel_id=None):
    """Download output images from Kaggle kernel"""
    kernel_id = kernel_id or KERNEL_ID
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Downloading output from kernel {kernel_id} to {dest_path}")
    
    kaggle_cmd = _get_kaggle_command()

    # result = subprocess.run([
    #     *kaggle_cmd, "kernels", "output",
    #     kernel_id,
    #     "-p", str(dest_path).replace("\\", "/")
    # ], capture_output=True, text=True)


    # Capture everything to UTF-8 log files
    stdout_log = dest_path / "kaggle_cli_stdout.log"
    stderr_log = dest_path / "kaggle_cli_stderr.log"

    result = subprocess.run(
        [*kaggle_cmd, "kernels", "output", kernel_id, "-p", str(dest_path).replace("\\", "/")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    
    logging.info(f"Download completed with return code {result.returncode}")
    if result.returncode != 0:
        logging.warning("Kaggle command returned non-zero exit code, but download may have succeeded")
    
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
