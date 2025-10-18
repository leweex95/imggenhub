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
            capture_output=True, text=True, encoding='utf-8'
        )
        if result.returncode != 0:
            logging.error("Error fetching status: %s", result.stderr.strip())
            return "unknown"

        match = re.search(r'has status "(.*)"', result.stdout)
        status = match.group(1) if match else "unknown"
        logging.info("Kernel status: %s", status)

        if status.lower() == "unknown":
            logging.error("Unable to parse kernel status from output")
            return "unknown"

        if status.lower() in ["kernelworkerstatus.complete", "kernelworkerstatus.error"]:
            logging.info("Kernel finished with status: %s", status)
            return status.lower()

        time.sleep(poll_interval)

def _get_kaggle_command():
    """
    Get the appropriate command to run Kaggle CLI.
    
    Returns:
        list: Command parts to execute kaggle CLI
    """
    # Check if poetry is available and we're in a poetry project
    def find_pyproject_toml():
        current = Path.cwd()
        while current != current.parent:  # Stop at root
            if (current / "pyproject.toml").exists():
                return True
            current = current.parent
        return False
    
    if shutil.which("poetry") and find_pyproject_toml():
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
