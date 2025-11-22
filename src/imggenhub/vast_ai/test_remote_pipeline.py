"""
End-to-end test for remote custom pipeline execution on Vast.ai.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

from imggenhub.vast_ai.client import VastAIClient
from imggenhub.vast_ai.ssh import SSHConnection
from imggenhub.vast_ai.remote_executor import RemoteExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_remote_pipeline(
    api_key: str,
    gpu_type: str,
    max_price: float,
    model_name: str,
    guidance: float,
    steps: int,
    precision: str,
    prompt: str,
    hf_token: str | None = None,
):
    """
    End-to-end test: rent GPU, set up environment, run pipeline, download results.
    """
    if not api_key:
        raise ValueError("--api-key is required")
    if not gpu_type:
        raise ValueError("--gpu-type is required")
    if max_price <= 0:
        raise ValueError("--max-price must be positive")
    if not model_name:
        raise ValueError("--model-name is required")
    if not prompt:
        raise ValueError("--prompt is required")

    logger.info("=== Vast.ai Remote Pipeline Test ===")
    logger.info(f"GPU Type: {gpu_type}")
    logger.info(f"Max Price: ${max_price}/hr")
    logger.info(f"Model: {model_name}")
    logger.info(f"Prompt: {prompt}")

    client = VastAIClient(api_key=api_key)

    try:
        # Step 1: Find available instances
        logger.info("Step 1: Searching for available instances...")
        instances = client.search_instances(
            gpu_type=gpu_type,
            max_price=max_price,
            min_vram=20,
        )

        if not instances:
            logger.error(f"No instances found matching: {gpu_type}, max_price=${max_price}")
            return False

        best_instance = instances[0]
        logger.info(f"Selected instance: {best_instance['id']}")
        logger.info(f"  GPU: {best_instance['gpu_name']}")
        logger.info(f"  VRAM: {best_instance['gpu_ram']} GB")
        logger.info(f"  Price: ${best_instance['dph_total']}/hr")

        # Step 2: Rent the instance
        logger.info("Step 2: Renting instance...")
        instance_id = client.rent_instance(instance_id=best_instance['id'])
        logger.info(f"Instance rented: {instance_id}")

        # Step 3: Wait for SSH access
        logger.info("Step 3: Waiting for SSH access (max 5 minutes)...")
        ssh_info = client.wait_for_ssh(instance_id, timeout=300)
        if not ssh_info:
            logger.error("SSH access not available after 5 minutes")
            client.destroy_instance(instance_id)
            return False

        logger.info(f"SSH available: {ssh_info['host']}:{ssh_info['port']}")

        # Step 4: Connect and set up environment
        logger.info("Step 4: Setting up remote environment...")
        ssh = SSHConnection(
            host=ssh_info['host'],
            port=ssh_info['port'],
            username=ssh_info['username'],
            ssh_key=ssh_info['ssh_key'],
        )

        setup_ok = ssh.execute_script("setup.sh", check=True)
        if not setup_ok:
            logger.error("Remote setup failed")
            client.destroy_instance(instance_id)
            return False

        logger.info("Remote environment ready")

        # Step 5: Upload test prompts
        logger.info("Step 5: Uploading test prompts...")
        test_prompts = {
            "prompts": [prompt],
            "model_config": {
                "model_name": model_name,
                "guidance": guidance,
                "steps": steps,
                "precision": precision,
            }
        }

        prompts_json = json.dumps(test_prompts)
        ssh.execute_command(f"mkdir -p ~/imggenhub_test && echo '{prompts_json}' > ~/imggenhub_test/prompts.json")

        # Step 6: Run the pipeline
        logger.info("Step 6: Running image generation pipeline...")
        cmd = (
            f"cd ~/imggenhub_test && "
            f"python -m imggenhub.kaggle.main "
            f"--prompts_file prompts.json "
            f"--model_name {model_name} "
            f"--guidance {guidance} "
            f"--steps {steps} "
            f"--precision {precision} "
            f"--prompt '{prompt}'"
        )
        if hf_token:
            cmd += f" --hf_token {hf_token}"

        output, exit_code = ssh.execute_command(cmd, check=False)
        if exit_code != 0:
            logger.error(f"Pipeline execution failed with exit code {exit_code}")
            logger.error(f"Output: {output}")
            client.destroy_instance(instance_id)
            return False

        logger.info("Pipeline execution completed successfully")

        # Step 7: Download results
        logger.info("Step 7: Downloading results...")
        local_output = Path("output_from_vast_ai")
        local_output.mkdir(exist_ok=True)
        ssh.download_directory(remote_path="~/imggenhub_test/output", local_path=str(local_output))
        logger.info(f"Results downloaded to: {local_output}")

        # Step 8: Verify results
        logger.info("Step 8: Verifying results...")
        image_files = list(local_output.glob("**/*.png")) + list(local_output.glob("**/*.jpg"))
        if image_files:
            logger.info(f"Generated {len(image_files)} image(s)")
            for img in image_files:
                logger.info(f"  - {img}")
        else:
            logger.warning("No images generated")

        logger.info("=== Test Completed Successfully ===")
        return True

    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        return False

    finally:
        # Clean up: destroy instance
        if instance_id:
            logger.info("Cleaning up: destroying instance...")
            try:
                client.destroy_instance(instance_id)
                logger.info("Instance destroyed")
            except Exception as e:
                logger.error(f"Failed to destroy instance: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test remote pipeline on Vast.ai")
    parser.add_argument("--api-key", type=str, required=True, help="Vast.ai API key")
    parser.add_argument("--gpu-type", type=str, default="nvidia-tesla-p40", help="GPU type filter (e.g., tesla-p40). Your target: Tesla P40")
    parser.add_argument("--max-price", type=float, default=0.15, help="Max price per hour (USD). Your target GPU: ~$0.113/hr")
    parser.add_argument("--model-name", type=str, default="stabilityai/stable-diffusion-3.5-large", help="Model name")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8", "int4"])
    parser.add_argument("--prompt", type=str, default="a beautiful landscape with mountains", help="Test prompt")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token for gated models")

    args = parser.parse_args()

    success = test_remote_pipeline(
        api_key=args.api_key,
        gpu_type=args.gpu_type,
        max_price=args.max_price,
        model_name=args.model_name,
        guidance=args.guidance,
        steps=args.steps,
        precision=args.precision,
        prompt=args.prompt,
        hf_token=args.hf_token,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
