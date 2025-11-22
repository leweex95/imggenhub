"""Command-line entrypoint for running dummy deployments on Vast.ai GPUs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from imggenhub.vast_ai.dummy import DummyDeploymentRunner


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dummy diagnostic script on a Vast.ai GPU")
    parser.add_argument("--instance-id", type=int, required=True, help="Active Vast.ai instance ID")
    parser.add_argument("--api-key", dest="api_key", type=str, default=None, help="Vast.ai API key (optional if .env configured)")
    parser.add_argument("--ssh-key", dest="ssh_key", type=str, default=None, help="Path to SSH private key")
    parser.add_argument("--ssh-password", dest="ssh_password", type=str, default=None, help="SSH password for the instance")
    parser.add_argument("--script-path", dest="script_path", type=str, default=None, help="Local path to a Python script to run remotely")
    parser.add_argument("--script-args", dest="script_args", type=str, default=None, help="Arguments passed to the remote script")
    parser.add_argument("--log-dir", dest="log_dir", type=str, default=None, help="Directory where stdout/stderr logs should be saved")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if not args.ssh_key and not args.ssh_password:
        print("Provide either --ssh-key or --ssh-password for SSH access", file=sys.stderr)
        return 2

    log_dir_path = Path(args.log_dir) if args.log_dir else None

    try:
        runner = DummyDeploymentRunner(
            api_key=args.api_key,
            log_dir=log_dir_path,
            ssh_key=args.ssh_key,
            ssh_password=args.ssh_password,
        )
        result = runner.run(
            instance_id=args.instance_id,
            script_path=args.script_path,
            script_args=args.script_args,
        )
    except Exception as exc:
        print(f"Dummy deployment failed: {exc}", file=sys.stderr)
        return 1

    status = "succeeded" if result.exit_code == 0 else "failed"
    print(f"Dummy deployment {status} for instance {result.instance_id}")
    print(f"Log file saved to: {result.log_file}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
