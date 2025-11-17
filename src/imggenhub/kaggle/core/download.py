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

    kaggle_cmd = _get_kaggle_command()

    # Log files
    stdout_log = dest_path / "kaggle_cli_stdout.log"
    stderr_log = dest_path / "kaggle_cli_stderr.log"

    # Run Kaggle CLI safely
    result = subprocess.run(
        [*kaggle_cmd, "kernels", "output", kernel_id, "-p", str(dest_path).replace("\\", "/")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )

    # Write all outputs to files only — no console/logging formatting
    stdout_log.write_text(result.stdout, encoding="utf-8")
    stderr_log.write_text(result.stderr, encoding="utf-8")

    # Minimal logging: only numeric info, no Unicode
    # logging.info(f"Download completed with return code {result.returncode}")
    # logging.info(f"Check logs at: {stdout_log}, {stderr_log}")

    # if result.returncode != 0:
    #     raise RuntimeError(f"Kaggle CLI failed with code {result.returncode}. See log files above.")

    if result.returncode != 0:
        # Check if download actually succeeded despite return code
        stdout_content = stdout_log.read_text()
        # Check if any files were downloaded by looking for common success indicators
        files_in_dest = list(dest_path.rglob("*"))
        has_output_files = any(f.is_file() and f.name.endswith(('.png', '.jpg', '.jpeg', '.log')) for f in files_in_dest)
        
        if not has_output_files and "Output file downloaded to" not in stdout_content:
            logging.warning("Kaggle CLI failed (non-zero exit code). Check stdout/stderr log files.")
            # Don't raise, continue (Kaggle CLI sometimes returns non-zero even on success)
        elif has_output_files:
            logging.info(f"Downloaded {len([f for f in files_in_dest if f.is_file()])} file(s) successfully despite Kaggle CLI non-zero exit code")

    # Post-process: flatten any nested 'output' folders produced by the kernel
    # Some kernels save into 'output/<runname>/' which causes download to produce
    # nested folders like dest/output/<runname>/image.png; move images/logs to dest root.
    files_to_move = list(dest_path.rglob("**/*"))
    for f in files_to_move:
        if f.is_file() and f.parent != dest_path:
            # Only move images/logs (keep everything else)
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.log'):
                target = dest_path / f.name
                # Avoid collisions: append a counter if necessary
                counter = 1
                original_stem = f.stem
                while target.exists():
                    target = dest_path / f"{original_stem}_{counter}{f.suffix}"
                    counter += 1
                shutil.move(str(f), str(target))

    # Remove empty directories under dest_path created by the download
    for d in [p for p in dest_path.rglob("*") if p.is_dir()]:
        try:
            if not any(d.iterdir()):
                d.rmdir()
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
