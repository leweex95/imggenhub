"""Automation helpers for deploying and running lightweight scripts on Vast.ai GPUs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from imggenhub.vast_ai.api import VastAiClient
from imggenhub.vast_ai.ssh import SSHClient


@dataclass(slots=True)
class DummyDeploymentResult:
    """Represents the outcome of a dummy deployment run."""

    instance_id: int
    exit_code: int
    stdout: str
    stderr: str
    log_file: Path


class DummyDeploymentRunner:
    """Deploys and executes a lightweight verification script on a Vast.ai instance."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        log_dir: Optional[Path] = None,
        ssh_key: Optional[str] = None,
        ssh_password: Optional[str] = None,
    ) -> None:
        if not ssh_key and not ssh_password:
            raise ValueError("Provide either ssh_key or ssh_password for remote execution")

        self.vast_client = VastAiClient(api_key)
        self.log_dir = Path(log_dir or Path("output") / "vast_ai_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ssh_key = ssh_key
        self.ssh_password = ssh_password

    def run(
        self,
        instance_id: int,
        script_path: Optional[str] = None,
        script_args: Optional[str] = None,
    ) -> DummyDeploymentResult:
        """Upload and execute a dummy script on a running Vast.ai instance."""
        if not isinstance(instance_id, int) or instance_id <= 0:
            raise ValueError("instance_id must be a positive integer")

        instance = self.vast_client.get_instance(instance_id)
        script_content = self._load_script(script_path)
        remote_script_path = f"/tmp/imggenhub_dummy_{instance_id}.py"
        log_file = self._build_log_file(instance_id)

        ssh_client = SSHClient(
            host=instance.ssh_host,
            port=instance.ssh_port,
            username=instance.ssh_user,
            private_key_path=self.ssh_key,
            password=self.ssh_password,
        )

        temp_path: Optional[Path] = None
        stdout: str = ""
        stderr: str = ""
        exit_code: int = 1

        try:
            with NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8") as tmp:
                tmp.write(script_content)
                temp_path = Path(tmp.name)

            ssh_client.connect()
            ssh_client.upload_file(str(temp_path), remote_script_path)

            command = f"python3 {remote_script_path}"
            if script_args:
                command = f"{command} {script_args}"

            with log_file.open("w", encoding="utf-8") as handle:
                def _on_chunk(chunk: str) -> None:
                    print(chunk, end="")
                    handle.write(chunk)
                    handle.flush()

                exit_code, stdout, stderr = ssh_client.execute_streaming(command, on_chunk=_on_chunk)

            ssh_client.execute(f"rm -f {remote_script_path}")
        finally:
            ssh_client.disconnect()
            if temp_path and temp_path.exists():
                temp_path.unlink()

        return DummyDeploymentResult(
            instance_id=instance_id,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            log_file=log_file,
        )

    def _build_log_file(self, instance_id: int) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"dummy_{instance_id}_{timestamp}.log"

    @staticmethod
    def _load_script(script_path: Optional[str]) -> str:
        if not script_path:
            return DummyDeploymentRunner._default_script()

        path = Path(script_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dummy script not found: {script_path}")

        return path.read_text(encoding="utf-8")

    @staticmethod
    def _default_script() -> str:
        return """
import os
import socket
import time
from datetime import datetime

print("=== imggenhub dummy deployment ===")
print(f"Hostname: {socket.gethostname()}")
print(f"Working directory: {os.getcwd()}")
print(f"Timestamp: {datetime.utcnow().isoformat()}Z")

for step in range(5):
    print(f"Step {step + 1}/5 :: alive and running")
    time.sleep(2)

print("Dummy deployment completed successfully")
""".strip()
