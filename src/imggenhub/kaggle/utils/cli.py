"""
CLI utilities for command logging and reconstruction.
"""
import sys
from pathlib import Path
from datetime import datetime


def reconstruct_command() -> str:
    """
    Reconstruct the full CLI command as it should be run with poetry.

    Returns:
        str: The full copy-pasteable poetry command
    """
    # Always reconstruct as poetry command for consistency
    args_part = " ".join(sys.argv[1:])
    return f"poetry run python -m imggenhub.kaggle.main {args_part}"


def setup_output_directory(base_name: str = None, base_dir: str = None) -> Path:
    """
    Create a timestamp-based output directory for the current run.

    Output structure: outputs/TIMESTAMP/ (or outputs/base_name_TIMESTAMP/ if base_name provided)
                      OR base_dir directly when base_dir is provided (no timestamp subfolder)

    Args:
        base_name: Optional prefix for the timestamp folder (default: None, uses timestamp only)
        base_dir: Optional base directory path - when provided, used directly without creating timestamp subfolder

    Returns:
        Path: Path to the created output directory
    """
    if base_dir:
        # When base_dir is explicitly provided, use it DIRECTLY without creating timestamp subfolder
        # This is the caller's responsibility to provide a unique directory
        dest_path = Path(base_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        return dest_path
    
    # Default behavior: create timestamp-based directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path.cwd() / "outputs"
    output_base.mkdir(parents=True, exist_ok=True)

    if base_name:
        run_name = f"{base_name}_{timestamp}"
    else:
        run_name = timestamp
    dest_path = output_base / run_name
    dest_path.mkdir(parents=True, exist_ok=True)

    return dest_path


def log_cli_command(dest_path: Path) -> None:
    """
    Log the reconstructed CLI command to the output directory.

    Args:
        dest_path: Path to the output directory
    """
    raw_command = reconstruct_command()

    # Print to console for immediate feedback
    print(f"CLI Command: {raw_command}")

    # Log to file
    cli_command_file = dest_path / "cli_command.txt"
    with open(cli_command_file, "w", encoding="utf-8") as f:
        f.write(raw_command)

    print(f"CLI command logged to: {cli_command_file}")
