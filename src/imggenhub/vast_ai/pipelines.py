"""High-level orchestrators for running predefined pipelines on Vast.ai."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

from imggenhub.vast_ai.executor import run_remote_pipeline
from imggenhub.vast_ai.types import RemoteRunResult
from imggenhub.vast_ai.utils import RunLogger

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineExecution:
    """Wraps a remote pipeline run with timing metadata."""

    instance_id: int
    result: RemoteRunResult
    duration_seconds: float


class PipelineOrchestrator:
    """Coordinates Stable Diffusion, FLUX, and other remote pipelines."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        repo_path: str = ".",
        config_path: str = "src/imggenhub/kaggle/config",
        log_dir: Optional[Path] = None,
    ) -> None:
        self.api_key = api_key
        self.repo_path = Path(repo_path)
        self.config_path = Path(config_path)
        self.vast_client = VastAiClient(api_key)
        self.run_logger = RunLogger(log_dir)

    def run_stable_diffusion(
        self,
        instance_id: int,
        ssh_key: Optional[str],
        prompts_file: Optional[str] = "config/prompts.json",
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        model_name: str = "stabilityai/stable-diffusion-3.5-large",
        guidance: float = 8.5,
        steps: int = 40,
        precision: str = "fp16",
        expected_images: int = 4,
        hf_token: Optional[str] = None,
    ) -> PipelineExecution:
        """Run the Stable Diffusion pipeline with sensible defaults."""
        return self.run_model(
            instance_id=instance_id,
            ssh_key=ssh_key,
            model_name=model_name,
            guidance=guidance,
            steps=steps,
            precision=precision,
            prompt=prompt,
            prompts_file=prompts_file,
            negative_prompt=negative_prompt,
            expected_images=expected_images,
            hf_token=hf_token,
        )

    def run_flux(
        self,
        instance_id: int,
        ssh_key: Optional[str],
        prompts_file: Optional[str] = "config/prompts_flux.json",
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        model_name: str = "black-forest-labs/FLUX.1-schnell",
        guidance: float = 4.0,
        steps: int = 20,
        precision: str = "fp16",
        expected_images: int = 6,
        hf_token: Optional[str] = None,
    ) -> PipelineExecution:
        """Run the Flux AI pipeline using the remote executor."""
        return self.run_model(
            instance_id=instance_id,
            ssh_key=ssh_key,
            model_name=model_name,
            guidance=guidance,
            steps=steps,
            precision=precision,
            prompt=prompt,
            prompts_file=prompts_file,
            negative_prompt=negative_prompt,
            expected_images=expected_images,
            hf_token=hf_token,
        )

    def run_model(
        self,
        *,
        instance_id: int,
        ssh_key: Optional[str],
        model_name: str,
        guidance: float,
        steps: int,
        precision: str,
        prompt: Optional[str] = None,
        prompts_file: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        expected_images: int = 4,
        hf_token: Optional[str] = None,
    ) -> PipelineExecution:
        """Run any supported model by specifying its Hugging Face identifier."""
        return self._execute_pipeline(
            instance_id=instance_id,
            ssh_key=ssh_key,
            model_name=model_name,
            guidance=guidance,
            steps=steps,
            precision=precision,
            prompt=prompt,
            prompts_file=prompts_file,
            negative_prompt=negative_prompt,
            expected_images=expected_images,
            hf_token=hf_token,
        )

    def _execute_pipeline(
        self,
        *,
        instance_id: int,
        ssh_key: Optional[str],
        model_name: str,
        guidance: float,
        steps: int,
        precision: str,
        prompt: Optional[str],
        prompts_file: Optional[str],
        negative_prompt: Optional[str],
        expected_images: int,
        hf_token: Optional[str],
    ) -> PipelineExecution:
        if not self.repo_path.is_dir():
            raise FileNotFoundError(f"Repository path not found: {self.repo_path}")
        if not self.config_path.is_dir():
            raise FileNotFoundError(f"Config path not found: {self.config_path}")

        start = perf_counter()
        result = run_remote_pipeline(
            api_key=self.api_key,
            instance_id=instance_id,
            model_name=model_name,
            guidance=guidance,
            steps=steps,
            precision=precision,
            repo_path=str(self.repo_path),
            config_path=str(self.config_path),
            prompt=prompt,
            prompts_file=prompts_file,
            negative_prompt=negative_prompt,
            hf_token=hf_token,
            ssh_key=ssh_key,
        )
        duration = perf_counter() - start

        self._log_run(
            instance_id=instance_id,
            model_name=model_name,
            duration_seconds=duration,
            expected_images=expected_images,
            success=result.exit_code == 0,
            log_file=result.log_file,
        )

        return PipelineExecution(
            instance_id=instance_id,
            result=result,
            duration_seconds=duration,
        )

    def _log_run(
        self,
        *,
        instance_id: int,
        model_name: str,
        duration_seconds: float,
        expected_images: int,
        success: bool,
        log_file: Optional[Path],
    ) -> None:
        try:
            instance = self.vast_client.get_instance(instance_id)
        except Exception as exc:
            logger.warning(f"Unable to fetch instance {instance_id} for logging: {exc}")
            return

        status = "completed" if success else "failed"
        notes = f"log: {log_file}" if log_file else None

        self.run_logger.log_run(
            instance_id=str(instance.id),
            instance_type=instance.gpu_name,
            instance_dph=instance.price_per_hour,
            model_name=model_name,
            num_images=expected_images,
            duration_seconds=duration_seconds,
            status=status,
            notes=notes,
        )