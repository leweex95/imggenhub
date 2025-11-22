"""CLI for automating Forge UI deployments on Vast.ai instances."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from imggenhub.vast_ai.api import VastAiClient
from imggenhub.vast_ai.ssh import SSHClient


@dataclass(slots=True)
class ForgeDeploymentResult:
    exit_code: int
    stdout: str
    stderr: str
    log_file: Path


def deploy_forge_ui(
    *,
    api_key: Optional[str],
    instance_id: int,
    ssh_key: Optional[str],
    ssh_password: Optional[str],
    script_path: Optional[str] = None,
) -> ForgeDeploymentResult:
    if not ssh_key and not ssh_password:
        raise ValueError("Provide either ssh_key or ssh_password for Forge UI deployment")

    client = VastAiClient(api_key)
    instance = client.get_instance(instance_id)

    script = Path(script_path) if script_path else Path(__file__).parent / "deploy" / "forge_deploy.sh"
    if not script.is_file():
        raise FileNotFoundError(f"Forge UI script not found: {script}")

    log_dir = Path("output") / "vast_ai_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"forge_{instance_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"

    ssh_client = SSHClient(
        host=instance.ssh_host,
        port=instance.ssh_port,
        username=instance.ssh_user,
        private_key_path=ssh_key,
        password=ssh_password,
    )

    remote_script = "/tmp/forge_deploy.sh"
    stdout = ""
    stderr = ""
    exit_code = 1

    try:
        ssh_client.connect()
        ssh_client.upload_file(str(script), remote_script)
        ssh_client.execute(f"chmod +x {remote_script}")

        with log_file.open("w", encoding="utf-8") as handle:
            def _on_chunk(chunk: str) -> None:
                print(chunk, end="")
                handle.write(chunk)
                handle.flush()

            exit_code, stdout, stderr = ssh_client.execute_streaming(
                f"bash {remote_script}",
                on_chunk=_on_chunk,
            )
    finally:
        ssh_client.disconnect()

    return ForgeDeploymentResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        log_file=log_file,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy Forge UI on a Vast.ai GPU")
    parser.add_argument("--instance-id", type=int, required=True, help="Active Vast.ai instance ID")
    parser.add_argument("--api-key", dest="api_key", type=str, default=None, help="Vast.ai API key")
    parser.add_argument("--ssh-key", dest="ssh_key", type=str, default=None, help="Path to SSH private key")
    parser.add_argument("--ssh-password", dest="ssh_password", type=str, default=None, help="SSH password")
    parser.add_argument(
        "--script-path",
        dest="script_path",
        type=str,
        default=None,
        help="Override path to forge_deploy.sh",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    try:
        result = deploy_forge_ui(
            api_key=args.api_key,
            instance_id=args.instance_id,
            ssh_key=args.ssh_key,
            ssh_password=args.ssh_password,
            script_path=args.script_path,
        )
    except Exception as exc:
        print(f"Forge UI deployment failed: {exc}", file=sys.stderr)
        return 1

    print(f"Forge UI deployment finished with exit code {result.exit_code}")
    print(f"Logs: {result.log_file}")
    if result.stderr:
        print(f"stderr:\n{result.stderr}", file=sys.stderr)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
