"""Type definitions for Vast.ai integration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class RemoteRunResult:
    """Structured result for a remote pipeline execution."""

    exit_code: int
    stdout: str
    stderr: str
    log_file: Optional[Path]
    output_dir: Path
