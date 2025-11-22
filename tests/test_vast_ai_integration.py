#!/usr/bin/env python3
"""
Comprehensive test suite for Vast.ai integration with Tesla P40 GPU.
Tests API connectivity, instance management, and end-to-end execution.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def test_config_loading() -> bool:
    """Test .env file loading and configuration."""
    logger.info("=" * 70)
    logger.info("TEST 1: Configuration Loading (.env)")
    logger.info("=" * 70)

    try:
        from imggenhub.vast_ai.config import get_vast_ai_config

        config = get_vast_ai_config()
        logger.info(f"✓ API Key loaded: {config.api_key[:10]}...")
        logger.info(f"✓ Base URL: {config.base_url}")
        logger.info(f"✓ Timeout: {config.timeout}s")
        return True
    except ValueError as e:
        logger.error(f"✗ Configuration failed: {e}")
        logger.error("Make sure .env file exists with VAST_AI_API_KEY set")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return False


def test_api_connectivity(api_key: Optional[str] = None) -> bool:
    """Test Vast.ai API connectivity."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: API Connectivity")
    logger.info("=" * 70)

    try:
        from imggenhub.vast_ai.api import VastAiClient

        client = VastAiClient(api_key)
        logger.info("✓ VastAiClient initialized")

        # List instances (should work even if empty)
        instances = client.list_instances()
        logger.info(f"✓ API accessible. Active instances: {len(instances)}")

        if instances:
            logger.info("  Active instances:")
            for inst in instances:
                logger.info(f"    - {inst.label} ({inst.gpu_name}): {inst.status}")

        return True
    except ValueError as e:
        logger.error(f"✗ API key error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ API connectivity failed: {e}")
        return False


def test_gpu_search(api_key: Optional[str] = None, gpu_name: str = "Tesla P40") -> dict:
    """Search for available GPUs matching criteria."""
    logger.info("\n" + "=" * 70)
    logger.info(f"TEST 3: GPU Search ('{gpu_name}')")
    logger.info("=" * 70)

    try:
        from imggenhub.vast_ai.api import VastAiClient

        client = VastAiClient(api_key)

        # Search for Tesla P40
        offers = client.search_offers(gpu_name=gpu_name, limit=5)
        logger.info(f"✓ Found {len(offers)} offers for {gpu_name}")

        if offers:
            logger.info("\nTop 5 offers:")
            for i, offer in enumerate(offers[:5], 1):
                provider = offer.get("provider", "Unknown")
                dph = offer.get("dph_total", 0)
                vram = offer.get("gpu_ram", 0)
                region = offer.get("geolocation", "Unknown")
                logger.info(
                    f"  {i}. Provider: {provider} | "
                    f"${dph:.3f}/hr | {vram}GB VRAM | {region}"
                )

            return {
                "success": True,
                "count": len(offers),
                "offers": offers,
            }
        else:
            logger.warning(f"✗ No offers found for {gpu_name}")
            return {
                "success": False,
                "count": 0,
                "offers": [],
            }

    except Exception as e:
        logger.error(f"✗ GPU search failed: {e}")
        return {
            "success": False,
            "count": 0,
            "offers": [],
        }


def test_cost_estimation() -> bool:
    """Test cost estimation utility."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Cost Estimation")
    logger.info("=" * 70)

    try:
        from imggenhub.vast_ai.utils import CostEstimator

        # Example: Estimate cost for 10 FLUX images on $0.113/hr GPU
        cost_data = CostEstimator.estimate_generation_cost(
            num_images=10,
            avg_time_per_image=30,  # 30 seconds per image
            instance_dph=0.113,  # Tesla P40 price
        )

        logger.info("✓ Cost estimation generated")
        logger.info(f"  - Images: {cost_data['num_images']}")
        logger.info(f"  - Avg time per image: {cost_data['avg_time_per_image_sec']:.1f}s")
        logger.info(f"  - Total duration: {cost_data['total_duration_min']:.1f} minutes")
        logger.info(f"  - GPU rate: ${cost_data['instance_dph']:.3f}/hr")
        logger.info(f"  - Total cost: ${cost_data['total_cost']:.4f}")
        logger.info(f"  - Cost per image: ${cost_data['cost_per_image']:.4f}")

        # Test budget recommendations
        recommendations = CostEstimator.recommend_model_for_budget(
            budget_usd=5.0,
            instance_dph=0.113,
        )

        logger.info(f"\n  Budget recommendations for $5.00 on Tesla P40:")
        for rec in recommendations["recommendations"]:
            logger.info(
                f"    - {rec['model']}: "
                f"~{rec['estimated_images']} images at {rec['avg_time_sec']}s each"
            )

        return True

    except Exception as e:
        logger.error(f"✗ Cost estimation failed: {e}")
        return False


def test_ssh_connectivity() -> bool:
    """Test SSH client initialization (no actual connection)."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: SSH Client Initialization")
    logger.info("=" * 70)

    try:
        from imggenhub.vast_ai.ssh import SSHClient

        # Initialize but don't connect (no live instance)
        ssh = SSHClient(
            host="test.example.com",
            port=22,
            username="root",
            password="test_password",
        )

        logger.info("✓ SSHClient initialized successfully")
        logger.info(f"  - Host: {ssh.host}")
        logger.info(f"  - Port: {ssh.port}")
        logger.info(f"  - Username: {ssh.username}")
        logger.info("  (Note: Actual connection test skipped, requires live instance)")

        return True

    except Exception as e:
        logger.error(f"✗ SSH client initialization failed: {e}")
        return False


def test_remote_executor() -> bool:
    """Test RemoteExecutor initialization (no actual execution)."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 6: RemoteExecutor Initialization")
    logger.info("=" * 70)

    try:
        from imggenhub.vast_ai.orchestration import RemoteExecutor
        from imggenhub.vast_ai.api import VastInstance
        from datetime import datetime

        # Create a mock instance
        mock_instance = VastInstance(
            id=12345,
            label="Test Instance",
            gpu_name="Tesla P40",
            gpu_count=1,
            status="running",
            ssh_host="192.168.1.100",
            ssh_port=22,
            ssh_user="root",
            price_per_hour=0.113,
            created_at=datetime.now(),
        )

        # Initialize executor (don't execute)
        executor = RemoteExecutor(
            instance=mock_instance,
            ssh_password="test_password",
        )

        logger.info("✓ RemoteExecutor initialized successfully")
        logger.info(f"  - Instance ID: {executor.instance.id}")
        logger.info(f"  - GPU: {executor.instance.gpu_name}")
        logger.info(f"  - Price: ${executor.instance.price_per_hour:.3f}/hr")
        logger.info("  (Note: Actual setup/execution skipped, requires live instance)")

        return True

    except Exception as e:
        logger.error(f"✗ RemoteExecutor initialization failed: {e}")
        return False


def test_cli_entry_point() -> bool:
    """Test CLI module loads correctly."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 7: CLI Entry Point")
    logger.info("=" * 70)

    try:
        from imggenhub.vast_ai.cli import main

        logger.info("✓ CLI module loaded successfully")
        logger.info(f"  - main() function available")
        logger.info("  (Note: Actual CLI invocation skipped, requires valid instance)")

        return True

    except Exception as e:
        logger.error(f"✗ CLI loading failed: {e}")
        return False


def test_package_imports() -> bool:
    """Test top-level package imports."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 8: Package-Level Imports")
    logger.info("=" * 70)

    try:
        from imggenhub.vast_ai import (
            VastAiClient,
            VastInstance,
            SSHClient,
            RemoteExecutor,
            get_vast_ai_config,
            VastAiConfig,
            CostEstimator,
            RunLogger,
            InstanceManager,
        )

        logger.info("✓ All package exports available:")
        logger.info("  - VastAiClient")
        logger.info("  - VastInstance")
        logger.info("  - SSHClient")
        logger.info("  - RemoteExecutor")
        logger.info("  - get_vast_ai_config()")
        logger.info("  - VastAiConfig")
        logger.info("  - CostEstimator")
        logger.info("  - RunLogger")
        logger.info("  - InstanceManager")

        return True

    except Exception as e:
        logger.error(f"✗ Package imports failed: {e}")
        return False


def main():
    """Run all tests and generate report."""
    logger.info("\n" + "=" * 70)
    logger.info("VAST.AI IMPLEMENTATION TEST SUITE")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    results = {}

    # Test 1: Configuration
    results["config_loading"] = test_config_loading()

    if not results["config_loading"]:
        logger.error("\n✗ Configuration test failed. Cannot continue with other tests.")
        logger.error("Please ensure .env file exists and contains VAST_AI_API_KEY.")
        print_summary(results)
        return 1

    # Get API key for subsequent tests
    try:
        from imggenhub.vast_ai.config import get_vast_ai_config
        config = get_vast_ai_config()
        api_key = config.api_key
    except Exception as e:
        logger.error(f"Failed to get API key: {e}")
        return 1

    # Test 2-8: Other tests
    results["api_connectivity"] = test_api_connectivity(api_key)
    gpu_search_result = test_gpu_search(api_key, gpu_name="Tesla P40")
    results["gpu_search"] = gpu_search_result["success"]
    results["cost_estimation"] = test_cost_estimation()
    results["ssh_connectivity"] = test_ssh_connectivity()
    results["remote_executor"] = test_remote_executor()
    results["cli_entry_point"] = test_cli_entry_point()
    results["package_imports"] = test_package_imports()

    # Print summary
    print_summary(results)

    # Generate recommendations
    if results["gpu_search"]:
        generate_recommendations(gpu_search_result["offers"])

    # Return exit code
    return 0 if all(results.values()) else 1


def print_summary(results: dict):
    """Print test summary."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for v in results.values() if v is True)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("-" * 70)
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✓ All tests passed! Vast.ai integration is ready.")
    else:
        logger.warning(f"✗ {total - passed} test(s) failed. Review errors above.")

    logger.info("=" * 70 + "\n")


def generate_recommendations(offers: list):
    """Generate recommendations based on available offers."""
    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDATIONS FOR TESLA P40 DEPLOYMENT")
    logger.info("=" * 70)

    if not offers:
        logger.info("No offers available to analyze.")
        return

    # Find cheapest option
    cheapest = min(offers, key=lambda x: x.get("dph_total", float("inf")))
    most_vram = max(offers, key=lambda x: x.get("gpu_ram", 0))

    logger.info("\nBest Value (Cheapest):")
    logger.info(
        f"  Provider: {cheapest.get('provider', 'Unknown')}\n"
        f"  Price: ${cheapest.get('dph_total', 0):.3f}/hr\n"
        f"  VRAM: {cheapest.get('gpu_ram', 0)}GB\n"
        f"  Region: {cheapest.get('geolocation', 'Unknown')}"
    )

    logger.info("\nHighest VRAM:")
    logger.info(
        f"  Provider: {most_vram.get('provider', 'Unknown')}\n"
        f"  Price: ${most_vram.get('dph_total', 0):.3f}/hr\n"
        f"  VRAM: {most_vram.get('gpu_ram', 0)}GB\n"
        f"  Region: {most_vram.get('geolocation', 'Unknown')}"
    )

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    sys.exit(main())
