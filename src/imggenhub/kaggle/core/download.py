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

    kaggle_cmd = _get_kaggle_command()

    # Log files
    stdout_log = dest_path / "kaggle_cli_stdout.log"
    stderr_log = dest_path / "kaggle_cli_stderr.log"

    # Run Kaggle CLI with hard timeout so it can NEVER hang this process
    logging.info(f"Downloading output artifacts from {kernel_id}...")
    try:
        result = subprocess.run(
            [*kaggle_cmd, "kernels", "output", kernel_id, "-p", str(dest_path).replace("\\", "/")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=60
        )
    except subprocess.TimeoutExpired:
        # Kill the process and continue; we will just scan the folder
        logging.warning("Kaggle CLI download timed out after 60 seconds; proceeding with whatever files are present.")
        result = None

    # Write outputs to log files (if available) but DO NOT block on them later
    if result is not None:
        try:
            stdout_log.write_text(result.stdout or "", encoding="utf-8")
            stderr_log.write_text(result.stderr or "", encoding="utf-8")
        except Exception:
            pass
    else:
        try:
            stdout_log.write_text("Download timed out after 60 seconds\n", encoding="utf-8")
            stderr_log.write_text("Timeout - CLI process terminated\n", encoding="utf-8")
        except Exception:
            pass

    # Post-process: we DO NOT care about logs or binaries, only final images
    # 1. Move all images from any nested folder to dest root
    files_to_move = list(dest_path.rglob("**/*"))
    image_files = []
    for f in files_to_move:
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            image_files.append(f)
            if f.parent != dest_path:
                target = dest_path / f.name
                counter = 1
                original_stem = f.stem
                while target.exists():
                    target = dest_path / f"{original_stem}_{counter}{f.suffix}"
                    counter += 1
                try:
                    shutil.move(str(f), str(target))
                    logging.debug(f"Moved image: {f}")
                except Exception:
                    pass

    if not image_files:
        logging.warning("No image files found after download and cleanup.")

    # 2. Remove ALL subdirectories (stable-diffusion.cpp, output_images, etc.)
    # After this, dest_path will be flat.
    for item in list(dest_path.iterdir()):
        if item.is_dir():
            try:
                shutil.rmtree(str(item))
                logging.debug(f"Removed directory: {item}")
            except Exception:
                pass

    # 3. Remove any non-image files from dest root (including logs)
    preserve_files = {"cli_command.txt"}
    for item in list(dest_path.iterdir()):
        if item.is_file() and item.name not in preserve_files and item.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            try:
                item.unlink()
                logging.debug(f"Removed non-image file: {item}")
            except Exception:
                pass


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
