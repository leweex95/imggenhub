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
    args_parts = []
    for arg in sys.argv[1:]:
        # Quote arguments that contain spaces
        if ' ' in arg:
            args_parts.append(f'"{arg}"')
        else:
            args_parts.append(arg)
    
    args_part = " ".join(args_parts)
    return f"poetry run python -m imggenhub.kaggle.main {args_part}"


def setup_output_directory(base_name: str = None) -> Path:
    """
    Create a timestamp-based output directory for the current run.

    Args:
        base_name: Base name for the output directory. If None, uses just timestamp.

    Returns:
        Path: Path to the created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path("output")
    output_base.mkdir(exist_ok=True)

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