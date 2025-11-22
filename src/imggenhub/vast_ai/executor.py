"""Remote pipeline execution logic."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from imggenhub.vast_ai.api import VastAiClient
from imggenhub.vast_ai.orchestration import RemoteExecutor
from imggenhub.vast_ai.types import RemoteRunResult

logger = logging.getLogger(__name__)


def run_remote_pipeline(
    api_key: Optional[str],
    instance_id: int,
    model_name: str,
    guidance: float,
    steps: int,
    precision: str,
    repo_path: str,
    config_path: str,
    prompt: Optional[str] = None,
    prompts_file: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    refiner_model_name: Optional[str] = None,
    refiner_guidance: Optional[float] = None,
    refiner_steps: Optional[int] = None,
    refiner_precision: Optional[str] = None,
    hf_token: Optional[str] = None,
    ssh_key: Optional[str] = None,
) -> RemoteRunResult:
    """Execute image generation pipeline on a remote Vast.ai instance."""
    if not isinstance(instance_id, int) or instance_id <= 0:
        raise ValueError("instance_id must be a positive integer")

    repo_path_obj = Path(repo_path)
    if not repo_path_obj.is_dir():
        raise ValueError(f"Repository path not found: {repo_path}")

    config_path_obj = Path(config_path)
    if not config_path_obj.is_dir():
        raise ValueError(f"Config path not found: {config_path}")

    executor: Optional[RemoteExecutor] = None
    stdout: str = ""
    stderr: str = ""
    output_dir = Path("output") / "vast_ai_results"

    try:
        client = VastAiClient(api_key)

        logger.info("Fetching instance %s...", instance_id)
        instance = client.get_instance(instance_id)
        logger.info("Instance found: %s (%s)", instance.label, instance.gpu_name)

        executor = RemoteExecutor(
            api_key=api_key,
            instance=instance,
            ssh_private_key=ssh_key,
        )

        logger.info("Setting up remote environment...")
        executor.setup_environment()

        logger.info("Uploading codebase...")
        executor.upload_codebase(str(repo_path_obj))

        logger.info("Uploading configuration...")
        executor.upload_config(str(config_path_obj))

        logger.info("Installing dependencies...")
        executor.install_dependencies()

        logger.info("Running pipeline (%s)...", model_name)
        exit_code, stdout, stderr = executor.run_pipeline(
            model_name=model_name,
            guidance=guidance,
            steps=steps,
            precision=precision,
            prompt=prompt,
            prompts_file=prompts_file,
            negative_prompt=negative_prompt,
            refiner_model_name=refiner_model_name,
            refiner_guidance=refiner_guidance,
            refiner_steps=refiner_steps,
            refiner_precision=refiner_precision,
            hf_token=hf_token,
        )

        if exit_code != 0:
            logger.error("Pipeline failed with exit code %s", exit_code)
            logger.error("Stdout:\n%s", stdout)
            logger.error("Stderr:\n%s", stderr)
            return RemoteRunResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                log_file=executor.last_log_path if executor else None,
                output_dir=output_dir,
            )

        logger.info("Pipeline completed successfully")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading results to %s...", output_dir)
        executor.download_results("/workspace/output", str(output_dir))

        return RemoteRunResult(
            exit_code=0,
            stdout=stdout,
            stderr=stderr,
            log_file=executor.last_log_path if executor else None,
            output_dir=output_dir,
        )

    except Exception as exc:  # pragma: no cover - network/SSH failures
        logger.error("Remote pipeline failed: %s", exc)
        return RemoteRunResult(
            exit_code=1,
            stdout=stdout,
            stderr=str(exc),
            log_file=executor.last_log_path if executor else None,
            output_dir=output_dir,
        )
