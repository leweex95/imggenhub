#!/usr/bin/env python
"""Example usage of auto-deployment: search → rent → deploy → cleanup."""
import logging
from pathlib import Path

from imggenhub.vast_ai.auto_deploy import AutoDeploymentOrchestrator, SearchCriteria

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_search_only():
    """Example 1: Just search for available GPUs without renting."""
    logger.info("=== EXAMPLE 1: Search for available GPUs ===\n")

    orchestrator = AutoDeploymentOrchestrator()

    # Search for P40 or V100 GPUs under $0.50/hr with at least 24GB VRAM
    criteria = SearchCriteria(
        min_vram=24,
        max_price=0.50,
        gpu_name=None,  # Any GPU
        min_reliability=95.0,
    )

    offers = orchestrator.search_and_rank(criteria)

    logger.info(f"\nFound {len(offers)} suitable GPUs:")
    for i, offer in enumerate(offers[:3], 1):
        logger.info(
            f"{i}. {offer['gpu_name']} ({offer['gpu_total_ram']}GB) "
            f"@ ${offer['dph_total']:.4f}/hr "
            f"(value: ${offer['value_score']:.6f}/GB)"
        )


def example_2_rent_and_connect():
    """Example 2: Rent cheapest GPU and verify SSH connection."""
    logger.info("\n=== EXAMPLE 2: Rent and verify SSH ===\n")

    ssh_key = Path.home() / ".ssh" / "id_ed25519"
    orchestrator = AutoDeploymentOrchestrator(ssh_key_path=str(ssh_key))

    criteria = SearchCriteria(
        min_vram=24,
        max_price=0.50,
        min_reliability=95.0,
    )

    try:
        # Search and rent
        instance = orchestrator.rent_cheapest(criteria)
        logger.info(f"\nRented: {instance.gpu_name} (ID: {instance.instance_id})")

        # Wait for SSH
        if orchestrator.wait_for_ssh_ready(timeout=120):
            logger.info("✓ SSH is ready, connection verified")
            
            # Execute a test command
            ssh = orchestrator.connect()
            exit_code, stdout, _ = ssh.execute("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
            logger.info(f"GPU Info: {stdout.strip()}")

        # Cleanup
        logger.info("\nCleaning up...")
        orchestrator.destroy_instance()

    except Exception as e:
        logger.error(f"Error: {e}")


def example_3_full_workflow():
    """Example 3: Full workflow (search → rent → deploy → cleanup)."""
    logger.info("\n=== EXAMPLE 3: Full Auto-Deployment Workflow ===\n")
    logger.info("This is what 'imggenhub-auto-sd' does internally:")
    logger.info("1. Search Vast.ai for cheapest P40 GPU")
    logger.info("2. Rent it")
    logger.info("3. Wait for SSH")
    logger.info("4. Deploy Stable Diffusion")
    logger.info("5. Generate image from prompt")
    logger.info("6. Destroy instance")
    logger.info("\nTo run this, execute:")
    logger.info("  poetry run imggenhub-auto-sd sd --prompt 'a landscape'")


if __name__ == "__main__":
    # Uncomment to run examples
    # example_1_search_only()
    # example_2_rent_and_connect()  # ⚠️ This will create and destroy an instance!
    example_3_full_workflow()
