"""CLI for automatic GPU instance deployment and image generation."""
import logging
import argparse
import sys
from pathlib import Path

from imggenhub.vast_ai.auto_deploy import AutoDeploymentOrchestrator, SearchCriteria
from imggenhub.vast_ai.pipelines import PipelineOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_auto_deployment(args):
    """Execute auto-deployment workflow: search → rent → deploy → cleanup."""
    # Parse search criteria
    criteria = SearchCriteria(
        min_vram=args.min_vram,
        max_price=args.max_price,
        gpu_name=args.gpu_name,
        prefer_spot=not args.no_spot,
        min_reliability=args.min_reliability,
    )

    # Initialize orchestrator
    ssh_key_path = Path(args.ssh_key).expanduser() if args.ssh_key else None
    orchestrator = AutoDeploymentOrchestrator(ssh_key_path=str(ssh_key_path))

    try:
        # Step 1: Search and rent
        logger.info("=" * 60)
        logger.info("STEP 1: SEARCHING AND RENTING GPU INSTANCE")
        logger.info("=" * 60)
        instance = orchestrator.rent_cheapest(criteria)

        # Step 2: Wait for SSH
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: WAITING FOR SSH CONNECTION")
        logger.info("=" * 60)
        if not orchestrator.wait_for_ssh_ready(timeout=args.ssh_timeout):
            raise RuntimeError("SSH connection timeout")

        # Step 3: Deploy and generate
        logger.info("\n" + "=" * 60)
        logger.info(f"STEP 3: DEPLOYING {args.model.upper()}")
        logger.info("=" * 60)

        pipeline = PipelineOrchestrator()

        if args.model == "sd":
            result = pipeline.run_stable_diffusion(
                instance_id=instance.instance_id,
                ssh_key=args.ssh_key,
                prompt=args.prompt,
                steps=args.steps,
                guidance=args.guidance,
            )
        elif args.model == "flux":
            result = pipeline.run_flux(
                instance_id=instance.instance_id,
                ssh_key=args.ssh_key,
                prompt=args.prompt,
                steps=args.steps,
                guidance=args.guidance,
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")

        logger.info(f"\n✓ Generation completed in {result.duration_seconds:.1f}s")
        logger.info(f"  Cost: ${instance.price_per_hour * (result.duration_seconds / 3600):.4f}")

        # Step 4: Cleanup (optional)
        if not args.keep_instance:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 4: CLEANING UP")
            logger.info("=" * 60)
            orchestrator.destroy_instance()
        else:
            logger.info(f"\n⏸️  Instance kept running: {instance.instance_id}")
            logger.info("   Remember to destroy it manually to avoid billing!")

        return 0

    except Exception as e:
        logger.error(f"\n✗ Deployment failed: {e}", exc_info=args.verbose)

        # Cleanup on failure
        if args.destroy_on_failure:
            logger.info("Destroying instance due to failure...")
            orchestrator.destroy_instance()

        return 1


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Automatically search, rent, and deploy on Vast.ai GPU instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-deploy Stable Diffusion with default criteria
  %(prog)s sd --prompt "a landscape"

  # Search for cheapest RTX 4090 under $0.50/hr with 40GB+ VRAM
  %(prog)s sd --gpu-name "RTX 4090" --max-price 0.50 --min-vram 40 \\
    --prompt "photorealistic portrait"

  # Keep instance running for more generations
  %(prog)s flux --prompt "a portrait" --keep-instance

  # Flux with custom inference steps and guidance
  %(prog)s flux --prompt "abstract art" --steps 25 --guidance 3.5
        """,
    )

    # Model and generation args
    parser.add_argument(
        "model",
        choices=["sd", "flux"],
        help="Model to use: 'sd' for Stable Diffusion or 'flux' for Flux",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Image generation prompt",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Inference steps (default: 30)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5 for SD, 3.5 for Flux)",
    )

    # Search criteria args
    parser.add_argument(
        "--min-vram",
        type=int,
        default=24,
        help="Minimum VRAM in GB (default: 24)",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=1.0,
        help="Maximum price per hour in USD (default: 1.0)",
    )
    parser.add_argument(
        "--gpu-name",
        help="Specific GPU name filter (e.g., 'RTX 4090', 'A100')",
    )
    parser.add_argument(
        "--min-reliability",
        type=float,
        default=95.0,
        help="Minimum reliability percentage (default: 95.0)",
    )
    parser.add_argument(
        "--no-spot",
        action="store_true",
        help="Only consider on-demand (non-spot) instances",
    )

    # Instance lifecycle args
    parser.add_argument(
        "--keep-instance",
        action="store_true",
        help="Keep instance running after deployment (default: destroy)",
    )
    parser.add_argument(
        "--destroy-on-failure",
        action="store_true",
        default=True,
        help="Destroy instance if deployment fails (default: True)",
    )

    # SSH args
    parser.add_argument(
        "--ssh-key",
        default="~/.ssh/id_ed25519",
        help="Path to SSH private key (default: ~/.ssh/id_ed25519)",
    )
    parser.add_argument(
        "--ssh-timeout",
        type=int,
        default=300,
        help="SSH ready timeout in seconds (default: 300)",
    )

    # Logging args
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    return run_auto_deployment(args)


if __name__ == "__main__":
    sys.exit(main())
