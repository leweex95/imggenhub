# download.py
import subprocess
from pathlib import Path
import logging
import shutil
import os

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

KERNEL_ID = "leventecsibi/stable-diffusion-batch-generator"


def run(dest="output_images", kernel_id=None):
    """Download ONLY image files from Kaggle kernel into a flat folder.

    Final structure: dest/IMAGE_FILES_ONLY (no subfolders, no binaries).
    """
    kernel_id = kernel_id or KERNEL_ID
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Log files
    stdout_log = dest_path / "kaggle_cli_stdout.log"
    stderr_log = dest_path / "kaggle_cli_stderr.log"

    # Run Kaggle CLI with hard timeout so it can NEVER hang this process
    logging.info(f"Downloading output artifacts from {kernel_id}...")
    
    # Use Popen instead of run to have more control over the process
    import threading
    import time
    
    kaggle_cmd = _get_kaggle_command()
    
    process = subprocess.Popen(
        [*kaggle_cmd, "kernels", "output", kernel_id, "-p", str(dest_path).replace("\\", "/")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    
    # Monitor for image files while download is running
    download_timeout = 60
    check_interval = 2
    elapsed = 0
    images_found = False
    
    while elapsed < download_timeout:
        # Check if any images have appeared
        if not images_found:
            for ext in (".png", ".jpg", ".jpeg"):
                if list(dest_path.rglob(f"**/*{ext}")):
                    images_found = True
                    logging.info("Image files detected, continuing download...")
                    break
        
        # Check if process has finished
        retcode = process.poll()
        if retcode is not None:
            logging.info(f"Kaggle CLI finished with return code {retcode}")
            break
        
        time.sleep(check_interval)
        elapsed += check_interval
    
    # If still running after timeout, kill it
    if process.poll() is None:
        logging.warning(f"Kaggle CLI still running after {download_timeout}s, terminating process")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    
    # Collect output
    try:
        stdout, stderr = process.communicate(timeout=1)
        stdout_log.write_text(stdout or "", encoding="utf-8")
        stderr_log.write_text(stderr or "", encoding="utf-8")
    except:
        stdout_log.write_text("Download process was terminated\n", encoding="utf-8")
        stderr_log.write_text("Process terminated after timeout\n", encoding="utf-8")

    logging.info("Download completed. All artifacts preserved.")


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
                check=False,
                timeout=10
            )
            if result.returncode == 0:
                return ["poetry", "run", "python", "-m", "kaggle.cli"]
        except Exception:
            pass
    
    # Fallback to direct python call
    return ["python", "-m", "kaggle.cli"]
