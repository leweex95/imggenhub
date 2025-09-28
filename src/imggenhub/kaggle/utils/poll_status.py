import subprocess
import time
import re
import logging
import shutil
from pathlib import Path

KERNEL_ID = "leventecsibi/stable-diffusion-batch-generator"
POLL_INTERVAL = 10  # seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run(kernel_id=None, poll_interval=None):
    """Poll Kaggle kernel status until complete or error"""
    kernel_id = kernel_id or KERNEL_ID
    poll_interval = poll_interval or POLL_INTERVAL

    while True:
        # Run `kaggle kernels status`
        kaggle_cmd = _get_kaggle_command()
        result = subprocess.run(
            [*kaggle_cmd, "kernels", "status", kernel_id],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            logging.error("Error fetching status: %s", result.stderr.strip())
            break

        match = re.search(r'has status "(.*)"', result.stdout)
        status = match.group(1) if match else "unknown"
        logging.info("Kernel status: %s", status)

        if status.lower() in ["kernelworkerstatus.complete", "kernelworkerstatus.error"]:
            logging.info("Kernel finished with status: %s", status)
            break

        time.sleep(poll_interval)


def cancel_kernel(kernel_id=None):
    """Cancel a running Kaggle kernel"""
    kernel_id = kernel_id or KERNEL_ID
    
    logging.info(f"Attempting to cancel kernel: {kernel_id}")
    
    try:
        # First check current status
        kaggle_cmd = _get_kaggle_command()
        status_result = subprocess.run(
            [*kaggle_cmd, "kernels", "status", kernel_id],
            capture_output=True, text=True, check=False
        )
        
        if status_result.returncode == 0:
            logging.info(f"Current status: {status_result.stdout.strip()}")
        
        # Cancel the kernel
        cancel_result = subprocess.run(
            [*kaggle_cmd, "kernels", "cancel", kernel_id],
            capture_output=True, text=True, check=False
        )
        
        if cancel_result.returncode == 0:
            logging.info("Kernel canceled successfully")
            logging.info(cancel_result.stdout.strip())
            return True
        else:
            logging.warning(f"Cancel command returned exit code {cancel_result.returncode}")
            logging.warning(f"stdout: {cancel_result.stdout.strip()}")
            logging.warning(f"stderr: {cancel_result.stderr.strip()}")
            return False
            
    except Exception as e:
        logging.error(f"Error canceling kernel: {e}")
        return False


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
