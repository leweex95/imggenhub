"""
Forge UI command-line interface for running image generation with Forge UI on Vast.ai.
"""
import argparse
import logging
import sys
from pathlib import Path

from imggenhub.vast_ai.client import VastAIClient
from imggenhub.vast_ai.ssh import SSHConnection
from imggenhub.vast_ai.forge import ForgeUIManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def deploy_and_run_forge_ui(
    api_key: str,
    gpu_type: str,
    max_price: float,
    model_name: str,
    guidance: float,
    steps: int,
    precision: str,
    prompt: str = None,
    prompts_file: str = None,
    hf_token: str | None = None,
    keep_instance: bool = False,
):
    """
    Deploy Forge UI on a Vast.ai instance and run inference.

    Args:
        api_key: Vast.ai API key
        gpu_type: GPU type filter (e.g., 'nvidia-tesla-p40')
        max_price: Max price per hour (USD)
        model_name: Model to load in Forge UI
        guidance: Guidance scale for generation
        steps: Number of inference steps
        precision: Precision level (fp32, fp16, int8, int4)
        prompt: Single prompt string
        prompts_file: Path to JSON file with multiple prompts
        hf_token: HuggingFace token for gated models
        keep_instance: If True, don't destroy instance after completion
    """
    if not api_key:
        raise ValueError("--api-key is required")
    if not gpu_type:
        raise ValueError("--gpu-type is required")
    if not model_name:
        raise ValueError("--model-name is required")
    if not prompt and not prompts_file:
        raise ValueError("--prompt or --prompts-file is required")

    logger.info("=== Deploying Forge UI on Vast.ai ===")
    logger.info(f"GPU Type: {gpu_type}")
    logger.info(f"Max Price: ${max_price}/hr")
    logger.info(f"Model: {model_name}")

    client = VastAIClient(api_key=api_key)
    instance_id = None

    try:
        # Step 1: Find and rent instance
        logger.info("Step 1: Searching for available instances...")
        instances = client.search_instances(
            gpu_type=gpu_type,
            max_price=max_price,
            min_vram=20,
        )

        if not instances:
            logger.error(f"No instances found matching criteria")
            return False

        best_instance = instances[0]
        logger.info(f"Selected instance: {best_instance['id']}")
        logger.info(f"  GPU: {best_instance['gpu_name']} ({best_instance['gpu_ram']}GB)")
        logger.info(f"  Price: ${best_instance['dph_total']}/hr")

        instance_id = client.rent_instance(instance_id=best_instance['id'])
        logger.info(f"Instance rented: {instance_id}")

        # Step 2: Wait for SSH access
        logger.info("Step 2: Waiting for SSH access...")
        ssh_info = client.wait_for_ssh(instance_id, timeout=300)
        if not ssh_info:
            logger.error("SSH access not available")
            return False

        logger.info(f"SSH available: {ssh_info['host']}:{ssh_info['port']}")

        # Step 3: Connect and set up Forge UI
        logger.info("Step 3: Setting up Forge UI environment...")
        ssh = SSHConnection(
            host=ssh_info['host'],
            port=ssh_info['port'],
            username=ssh_info['username'],
            ssh_key=ssh_info['ssh_key'],
        )

        # Deploy Forge UI
        forge = ForgeUIManager(ssh=ssh)
        if not forge.deploy():
            logger.error("Forge UI deployment failed")
            return False

        logger.info("Forge UI deployed successfully")

        # Step 4: Load model
        logger.info(f"Step 4: Loading model: {model_name}")
        if not forge.load_model(model_name=model_name, precision=precision, hf_token=hf_token):
            logger.error("Failed to load model in Forge UI")
            return False

        logger.info("Model loaded successfully")

        # Step 5: Generate images
        logger.info("Step 5: Generating images...")
        prompts = []
        if prompt:
            prompts = [prompt]
        elif prompts_file:
            import json
            with open(prompts_file, 'r') as f:
                data = json.load(f)
                prompts = data.get('prompts', []) if isinstance(data, dict) else data

        results = []
        for i, p in enumerate(prompts):
            logger.info(f"  Generating image {i+1}/{len(prompts)}: {p[:50]}...")
            result = forge.generate_image(
                prompt=p,
                guidance=guidance,
                steps=steps,
            )
            if result:
                results.append(result)
                logger.info(f"    Generated: {result}")
            else:
                logger.error(f"    Failed to generate image for prompt: {p}")

        # Step 6: Download results
        logger.info("Step 6: Downloading results...")
        output_dir = Path("output_from_forge_ui")
        output_dir.mkdir(exist_ok=True)

        for result in results:
            local_path = output_dir / Path(result).name
            ssh.download_file(remote_path=result, local_path=str(local_path))
            logger.info(f"  Downloaded: {local_path}")

        logger.info(f"All images downloaded to: {output_dir}")
        logger.info("=== Forge UI Generation Complete ===")
        return True

    except Exception as e:
        logger.error(f"Failed with exception: {e}", exc_info=True)
        return False

    finally:
        if instance_id and not keep_instance:
            logger.info("Cleaning up: destroying instance...")
            try:
                client.destroy_instance(instance_id)
                logger.info("Instance destroyed")
            except Exception as e:
                logger.error(f"Failed to destroy instance: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Forge UI on Vast.ai GPU")
    parser.add_argument("--api-key", type=str, required=True, help="Vast.ai API key")
    parser.add_argument("--gpu-type", type=str, default="nvidia-tesla-p40", help="GPU type filter")
    parser.add_argument("--max-price", type=float, default=0.15, help="Max price per hour (USD)")
    parser.add_argument("--model-name", type=str, required=True, help="Model to load in Forge UI (e.g., stabilityai/stable-diffusion-3.5-large)")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8", "int4"])
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt")
    parser.add_argument("--prompts-file", type=str, default=None, help="JSON file with multiple prompts")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token for gated models")
    parser.add_argument("--keep-instance", action="store_true", help="Don't destroy instance after completion")

    args = parser.parse_args()

    if not args.prompt and not args.prompts_file:
        logger.error("Error: --prompt or --prompts-file is required")
        sys.exit(1)

    success = deploy_and_run_forge_ui(
        api_key=args.api_key,
        gpu_type=args.gpu_type,
        max_price=args.max_price,
        model_name=args.model_name,
        guidance=args.guidance,
        steps=args.steps,
        precision=args.precision,
        prompt=args.prompt,
        prompts_file=args.prompts_file,
        hf_token=args.hf_token,
        keep_instance=args.keep_instance,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
