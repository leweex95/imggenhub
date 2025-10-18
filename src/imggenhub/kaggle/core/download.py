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
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run([
        *kaggle_cmd, "kernels", "output",
        kernel_id,
        "-p", str(dest_path).replace("\\", "/")
    ], capture_output=True, env=env, check=False)
    
    stdout_str = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
    stderr_str = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""
    
    logging.info(f"Kaggle stdout: {stdout_str}")
    if result.stderr:
        logging.error(f"Kaggle stderr: {stderr_str}")
    
    if result.returncode != 0:
        # Check if download actually succeeded despite return code
        if "Output file downloaded to" in stdout_str:
            logging.info("Download succeeded despite return code, continuing...")
        else:
            raise subprocess.CalledProcessError(result.returncode, result.args, stdout_str, stderr_str)
    
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
