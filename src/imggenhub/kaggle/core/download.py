# download.py
import subprocess
from pathlib import Path
import logging
import shutil
import os
import sys
import time
from typing import List

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

KERNEL_ID = "leventecsibi/stable-diffusion-batch-generator"

# Monitoring configuration (tunable for tests)
DOWNLOAD_TIMEOUT = 60   # seconds
CHECK_INTERVAL = 5      # seconds
QUIET_PERIOD = 15       # seconds with no new images before stopping the CLI

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _list_image_files(dest_path: Path) -> List[Path]:
    return [
        file_path
        for file_path in dest_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _move_images_to_final_folder(dest_path: Path, folder_name: str = "images") -> Path:
    # If dest_path already ends with the target folder name, use it directly (no subfolder)
    # This prevents double nesting when caller already provides a properly named directory
    if dest_path.name == folder_name:
        return dest_path
    
    images_dir = dest_path / folder_name
    images_dir.mkdir(parents=True, exist_ok=True)

    for image_path in _list_image_files(dest_path):
        if image_path.parent == images_dir:
            continue

        target_path = images_dir / image_path.name
        counter = 1
        while target_path.exists():
            target_path = images_dir / f"{image_path.stem}_{counter}{image_path.suffix}"
            counter += 1

        shutil.move(str(image_path), str(target_path))

    return images_dir


def run(dest="output_images", kernel_id=None):
    """Download ONLY image files from Kaggle kernel into a flat folder.

    Final structure: dest/IMAGE_FILES_ONLY (no subfolders, no binaries).
    """
    kernel_id = kernel_id or KERNEL_ID
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Log files (opened explicitly so we can stream output without blocking)
    stdout_log = dest_path / "kaggle_cli_stdout.log"
    stderr_log = dest_path / "kaggle_cli_stderr.log"
    stdout_handle = stdout_log.open("w", encoding="utf-8")
    stderr_handle = stderr_log.open("w", encoding="utf-8")

    # Run Kaggle CLI with hard timeout so it can NEVER hang this process
    logging.info(f"Downloading output artifacts from {kernel_id}...")

    kaggle_cmd = _get_kaggle_command()

    process = subprocess.Popen(
        [*kaggle_cmd, "kernels", "output", kernel_id, "-p", str(dest_path).replace("\\", "/")],
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        encoding="utf-8"
    )

    start_time = time.time()
    last_image_activity = None
    images_detected = False

    def _terminate_process():
        if process.poll() is not None:
            return
        logging.info("Stopping Kaggle CLI download process...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logging.warning("Kaggle CLI did not exit gracefully, killing process")
            process.kill()

    try:
        while True:
            current_images = _list_image_files(dest_path)
            if current_images:
                latest_mtime = max(image.stat().st_mtime for image in current_images)
                if not images_detected:
                    logging.info(
                        "Image files detected locally. Waiting up to %s seconds for additional prompts to finish...",
                        QUIET_PERIOD,
                    )
                images_detected = True
                last_image_activity = latest_mtime

            now = time.time()

            if images_detected and last_image_activity and (now - last_image_activity) >= QUIET_PERIOD:
                logging.info(
                    "No new image files detected in the last %s seconds. Assuming download complete.",
                    QUIET_PERIOD,
                )
                _terminate_process()
                break

            retcode = process.poll()
            if retcode is not None:
                logging.info(f"Kaggle CLI finished with return code {retcode}")
                break

            if now - start_time >= DOWNLOAD_TIMEOUT:
                logging.warning(
                    "Kaggle CLI still running after %s seconds, terminating process",
                    DOWNLOAD_TIMEOUT,
                )
                _terminate_process()
                break

            time.sleep(CHECK_INTERVAL)
    finally:
        # Make sure the process has fully exited and file handles are flushed
        if process.poll() is None:
            _terminate_process()
        process.wait()
        stdout_handle.flush()
        stderr_handle.flush()
        stdout_handle.close()
        stderr_handle.close()

    images_dir = _move_images_to_final_folder(dest_path)
    image_count = len(list(images_dir.glob("*")))

    if image_count > 0:
        logging.info(f"Download completed. {image_count} image(s) available at {images_dir}")
    else:
        logging.warning("Download completed but no images found.")

def _find_pyproject_toml() -> bool:
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return True
        current = current.parent
    return False


def _get_kaggle_command():
    """Return the command used to invoke the Kaggle CLI."""
    if shutil.which("poetry") and _find_pyproject_toml():
        try:
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

    # Fallback to the current Python interpreter to guarantee module availability
    return [sys.executable, "-m", "kaggle.cli"]
