"""
Filesystem utilities for directory and file management.
"""
from pathlib import Path


def ensure_output_directory(base_dir: str = "output") -> Path:
    """
    Ensure the base output directory exists.

    Args:
        base_dir: Name of the base output directory (default: "output")

    Returns:
        Path: Path to the base output directory
    """
    output_base = Path(base_dir)
    output_base.mkdir(exist_ok=True)
    return output_base


def create_run_directory(base_dir: str = "output", run_name: str = "output_images") -> Path:
    """
    Create a run-specific output directory.

    Args:
        base_dir: Base directory name
        run_name: Name for the run directory

    Returns:
        Path: Path to the created run directory
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = ensure_output_directory(base_dir)

    full_run_name = f"{run_name}_{timestamp}"
    dest_path = output_base / full_run_name
    dest_path.mkdir(parents=True, exist_ok=True)

    return dest_path