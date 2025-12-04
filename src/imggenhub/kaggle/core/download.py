"""Selective Kaggle kernel output downloader - images only.

Uses subprocess to download kernel outputs and monitors the local directory.
Exits immediately once all expected images are downloaded, preventing unnecessary
CMake and binary files from being transferred.
"""
import logging
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import List, Set

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
KERNEL_ID = "leventecsibi/stable-diffusion-batch-generator"
ALLOWED_NON_IMAGE_FILES = {"cli_command.txt", "stable-diffusion-batch-generator.log"}
ARTIFACT_DIRECTORIES = {"stable-diffusion.cpp"}


def _get_kaggle_command() -> List[str]:
    """Get the appropriate command to invoke Kaggle CLI."""
    import sys
    
    def find_pyproject_toml():
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                return True
            current = current.parent
        return False
    
    if shutil.which("poetry") and find_pyproject_toml():
        try:
            result = subprocess.run(
                ["poetry", "run", "python", "-c", "import kaggle"],
                capture_output=True, check=False, timeout=10
            )
            if result.returncode == 0:
                return ["poetry", "run", "python", "-m", "kaggle.cli"]
        except Exception:
            pass
    
    return [sys.executable, "-m", "kaggle.cli"]


def _handle_remove_readonly(func, path, exc_info):
    """Force-remove read-only files during rmtree."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as err:
        logger.warning(f"Failed to delete artifact component {path}: {err}")


def _list_local_image_files(dest_path: Path) -> Set[str]:
    """Scan local directory for image files. Returns set of file names (not full paths)."""
    image_files = set()
    for file_path in dest_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.add(file_path.name)
    return image_files


def run(kernel_id: str = KERNEL_ID, dest: str = "output_images") -> bool:
    """Download ONLY image files from Kaggle kernel by monitoring local directory.
    
    Starts the Kaggle CLI download process and monitors the destination directory.
    As soon as image files are detected and stable (no new files for 5 seconds),
    immediately terminates the download process to prevent unnecessary CMake/binary
    files from being transferred.
    
    Args:
        kernel_id: Kaggle kernel ID (e.g., "username/kernel-name")
        dest: Destination directory for outputs
        
    Returns:
        True if images were successfully downloaded, False otherwise
    """
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting selective download from {kernel_id} to {dest_path}...")
    
    kaggle_cmd = _get_kaggle_command()
    
    # Start the download process
    process = subprocess.Popen(
        [*kaggle_cmd, "kernels", "output", kernel_id, "-p", str(dest_path).replace("\\", "/")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    # Monitor for images
    start_time = time.time()
    timeout_seconds = 120
    check_interval = 1
    stable_threshold = 2  # seconds of no new images before we exit
    required_stable_checks = 1
    consecutive_stable_checks = 0
    last_image_count = 0
    last_new_image_time = None
    
    logger.info(f"Monitoring {dest_path} for image files...")
    
    try:
        while time.time() - start_time < timeout_seconds:
            current_images = _list_local_image_files(dest_path)
            current_count = len(current_images)
            
            if current_count > 0:
                if current_count > last_image_count:
                    # New images detected
                    logger.info(f"✓ Detected {current_count} image file(s): {current_images}")
                    last_image_count = current_count
                    last_new_image_time = time.time()
                    consecutive_stable_checks = 0
                else:
                    # No new images
                    if last_new_image_time:
                        elapsed = time.time() - last_new_image_time
                        if elapsed >= stable_threshold:
                            consecutive_stable_checks += 1
                            if consecutive_stable_checks >= required_stable_checks:
                                # Images are stable - exit now
                                logger.info(
                                    f"✓ All {current_count} image(s) stable for {stable_threshold}s. "
                                    "Terminating download to save bandwidth."
                                )
                                process.terminate()
                                try:
                                    process.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                
                                # Verify only images (plus expected metadata) remain
                                all_files = list(dest_path.rglob("*"))
                                residual_files = []
                                for detected in all_files:
                                    if not detected.is_file():
                                        continue
                                    if detected.suffix.lower() in IMAGE_EXTENSIONS:
                                        continue
                                    if detected.name in ALLOWED_NON_IMAGE_FILES:
                                        continue
                                    residual_files.append(detected)

                                if residual_files:
                                    for extra in residual_files:
                                        try:
                                            extra.unlink()
                                            logger.info(f"Deleted non-image artifact: {extra.name}")
                                        except Exception as err:
                                            logger.warning(f"Could not delete {extra}: {err}")
                                    logger.info(
                                        f"Removed {len(residual_files)} non-image artifact(s) after download kill."
                                    )
                                else:
                                    logger.info("✓ Confirmed: ONLY image files downloaded, no CMake/artifacts.")

                                for artifact_dir in ARTIFACT_DIRECTORIES:
                                    dir_path = dest_path / artifact_dir
                                    if not dir_path.exists():
                                        continue
                                    try:
                                        shutil.rmtree(
                                            dir_path,
                                            ignore_errors=False,
                                            onerror=_handle_remove_readonly,
                                        )
                                    except Exception as err:
                                        logger.warning(f"Failed to remove artifact directory {artifact_dir}: {err}")
                                        continue
                                    if dir_path.exists():
                                        logger.warning(f"Artifact directory still present: {artifact_dir}")
                                    else:
                                        logger.info(f"Removed artifact directory: {artifact_dir}")
                                
                                return True
            
            # Check if process died
            if process.poll() is not None:
                logger.info(f"✓ Download process completed naturally")
                return current_count > 0
            
            time.sleep(check_interval)
        
        # Timeout reached
        logger.warning(f"Timeout ({timeout_seconds}s) reached, terminating process")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        return last_image_count > 0
        
    except Exception as e:
        logger.error(f"Error during download: {e}")
        if process.poll() is None:
            process.kill()
        return False

