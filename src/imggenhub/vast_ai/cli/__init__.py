"""CLI interface for Vast.ai remote execution."""
import argparse
import logging
from pathlib import Path
from typing import Optional

from imggenhub.vast_ai.api import VastAiClient
from imggenhub.vast_ai.orchestration import RemoteExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
) -> int:
    """
    Execute image generation pipeline on a remote Vast.ai instance.

    Args:
        api_key: Vast.ai API key. If None, loads from .env.
        instance_id: Vast.ai instance ID to use. Required.
        model_name: Base model to use. Required.
        guidance: Guidance scale. Required.
        steps: Inference steps. Required.
        precision: Precision level. Required.
        repo_path: Path to local repository. Required.
        config_path: Path to config directory. Required.
        prompt: Single prompt string. Optional.
        prompts_file: Path to prompts JSON file. Optional.
        negative_prompt: Negative prompt. Optional.
        refiner_model_name: Refiner model name. Optional.
        refiner_guidance: Refiner guidance scale. Optional.
        refiner_steps: Refiner steps. Optional.
        refiner_precision: Refiner precision. Optional.
        hf_token: HuggingFace token. Optional.
        ssh_key: Path to SSH private key. Optional.

    Returns:
        Exit code (0 for success, non-zero for failure).

    Raises:
        ValueError: If required parameters are missing.
        RuntimeError: If execution fails.
    """
    if not isinstance(instance_id, int) or instance_id <= 0:
        raise ValueError("instance_id must be a positive integer")

    # Validate repo and config paths
    repo_path_obj = Path(repo_path)
    if not repo_path_obj.is_dir():
        raise ValueError(f"Repository path not found: {repo_path}")

    config_path_obj = Path(config_path)
    if not config_path_obj.is_dir():
        raise ValueError(f"Config path not found: {config_path}")

    try:
        # Initialize clients
        client = VastAiClient(api_key)

        # Get instance details
        logger.info(f"Fetching instance {instance_id}...")
        instance = client.get_instance(instance_id)
        logger.info(f"Instance found: {instance.label} ({instance.gpu_name})")

        # Create remote executor
        executor = RemoteExecutor(
            api_key=api_key,
            instance=instance,
            ssh_private_key=ssh_key,
        )

        # Setup environment
        logger.info("Setting up remote environment...")
        executor.setup_environment()

        # Upload codebase
        logger.info("Uploading codebase...")
        executor.upload_codebase(str(repo_path_obj))

        # Upload config
        logger.info("Uploading configuration...")
        executor.upload_config(str(config_path_obj))

        # Install dependencies
        logger.info("Installing dependencies...")
        executor.install_dependencies()

        # Run pipeline
        logger.info("Running pipeline...")
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
            logger.error(f"Pipeline failed with exit code {exit_code}")
            logger.error(f"Stdout:\n{stdout}")
            logger.error(f"Stderr:\n{stderr}")
            return exit_code

        logger.info("Pipeline completed successfully")
        logger.info(f"Stdout:\n{stdout}")

        # Download results
        output_dir = Path("output") / "vast_ai_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading results to {output_dir}...")
        executor.download_results("/workspace/output", str(output_dir))

        logger.info(f"Results saved to {output_dir}")
        return 0

    except Exception as e:
        logger.error(f"Remote pipeline failed: {e}")
        return 1


def main():
    """CLI entry point for Vast.ai remote execution."""
    parser = argparse.ArgumentParser(description="Run image generation pipeline on Vast.ai GPU")
    parser.add_argument("--api_key", type=str, default=None, help="Vast.ai API key (loads from .env if not provided)")
    parser.add_argument("--instance_id", type=int, required=True, help="Vast.ai instance ID")
    parser.add_argument("--model_name", type=str, required=True, help="Model to use for image generation")
    parser.add_argument("--guidance", type=float, required=True, help="Guidance scale (7-12 recommended)")
    parser.add_argument("--steps", type=int, required=True, help="Number of inference steps")
    parser.add_argument("--precision", type=str, required=True, choices=["fp32", "fp16", "int8", "int4"], help="Precision level")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt string")
    parser.add_argument("--prompts_file", type=str, default=None, help="Path to prompts JSON file")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt")
    parser.add_argument("--refiner_model_name", type=str, default=None, help="Refiner model")
    parser.add_argument("--refiner_guidance", type=float, default=None, help="Refiner guidance scale")
    parser.add_argument("--refiner_steps", type=int, default=None, help="Refiner steps")
    parser.add_argument("--refiner_precision", type=str, default=None, choices=["fp32", "fp16", "int8", "int4"], help="Refiner precision")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--ssh_key", type=str, default=None, help="Path to SSH private key")
    parser.add_argument("--repo_path", type=str, default=".", help="Path to repository root")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config directory")

    args = parser.parse_args()

    exit_code = run_remote_pipeline(
        api_key=args.api_key,
        instance_id=args.instance_id,
        model_name=args.model_name,
        guidance=args.guidance,
        steps=args.steps,
        precision=args.precision,
        prompt=args.prompt,
        prompts_file=args.prompts_file,
        negative_prompt=args.negative_prompt,
        refiner_model_name=args.refiner_model_name,
        refiner_guidance=args.refiner_guidance,
        refiner_steps=args.refiner_steps,
        refiner_precision=args.refiner_precision,
        hf_token=args.hf_token,
        ssh_key=args.ssh_key,
        repo_path=args.repo_path,
        config_path=args.config_path,
    )

    return exit_code


if __name__ == "__main__":
    exit(main())
