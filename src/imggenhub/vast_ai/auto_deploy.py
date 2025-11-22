"""Auto-deployment orchestrator: Search → Rent → Connect → Deploy → Cleanup."""
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from imggenhub.vast_ai.api import VastAiClient, VastInstance
from imggenhub.vast_ai.ssh import SSHClient

logger = logging.getLogger(__name__)


@dataclass
class SearchCriteria:
    """GPU search criteria for Vast.ai marketplace."""
    min_vram: int = 24  # Minimum VRAM in GB
    max_price: float = 1.0  # Maximum price per hour
    gpu_name: Optional[str] = None  # Specific GPU name (e.g., 'RTX 4090')
    prefer_spot: bool = True  # Prefer spot instances (cheaper)
    min_reliability: float = 95.0  # Minimum reliability percentage


@dataclass
class RentedInstance:
    """Details of a rented GPU instance."""
    instance_id: int
    gpu_name: str
    price_per_hour: float
    ssh_host: str
    ssh_port: int
    ssh_user: str


class AutoDeploymentOrchestrator:
    """Orchestrate automatic GPU instance lifecycle: search → rent → connect → deploy → cleanup."""

    def __init__(self, api_key: Optional[str] = None, ssh_key_path: Optional[str] = None):
        """Initialize orchestrator with Vast.ai API and SSH key."""
        self.client = VastAiClient(api_key)
        self.ssh_key_path = ssh_key_path
        self.rented_instance: Optional[RentedInstance] = None
        self.ssh_client: Optional[SSHClient] = None

    def search_and_rank(self, criteria: SearchCriteria) -> List[Dict[str, Any]]:
        """
        Search for available GPUs matching criteria, ranked by value (price/performance).

        Args:
            criteria: SearchCriteria object with filtering options

        Returns:
            List of ranked offers, best (cheapest/most reliable) first.

        Raises:
            requests.RequestException: If API request fails.
        """
        logger.info(f"Searching Vast.ai marketplace...")
        logger.info(f"  Min VRAM: {criteria.min_vram}GB, Max price: ${criteria.max_price}/hr")

        # Search for available offers
        offers = self.client.search_offers(
            gpu_name=criteria.gpu_name,
            max_price=criteria.max_price,
            min_vram=criteria.min_vram,
            limit=50,  # Get top 50 for ranking
        )

        if not offers:
            logger.warning("No offers found matching criteria")
            return []

        # Filter and rank offers
        filtered_offers = []
        for offer in offers:
            try:
                reliability = offer.get("reliability2", 0)
                price = offer.get("dph_total", float("inf"))
                vram = offer.get("gpu_total_ram", 0)

                # Apply reliability filter
                if reliability < criteria.min_reliability:
                    continue

                # Calculate value score (lower = better): price per VRAM
                value_score = price / vram if vram > 0 else float("inf")

                filtered_offers.append({
                    **offer,
                    "value_score": value_score,
                })
            except (KeyError, ZeroDivisionError):
                continue

        # Sort by value score (cheapest per GB)
        filtered_offers.sort(key=lambda x: x["value_score"])

        logger.info(f"Found {len(filtered_offers)} matching offers")
        for i, offer in enumerate(filtered_offers[:5]):
            logger.info(
                f"  {i+1}. {offer.get('gpu_name', 'Unknown')} "
                f"({offer.get('gpu_total_ram')}GB) @ ${offer.get('dph_total')}/hr "
                f"(reliability: {offer.get('reliability2')}%)"
            )

        return filtered_offers

    def rent_cheapest(self, criteria: SearchCriteria) -> RentedInstance:
        """
        Search for cheapest GPU matching criteria and rent it.

        Args:
            criteria: SearchCriteria with filtering options

        Returns:
            RentedInstance with connection details.

        Raises:
            ValueError: If no suitable offers found.
            requests.RequestException: If rental fails.
        """
        offers = self.search_and_rank(criteria)

        if not offers:
            raise ValueError(f"No suitable GPU offers found matching criteria: {criteria}")

        best_offer = offers[0]
        offer_id = best_offer.get("id")
        gpu_name = best_offer.get("gpu_name", "Unknown")
        price = best_offer.get("dph_total")

        logger.info(f"Renting {gpu_name} (${price:.4f}/hr)...")

        try:
            instance = self.client.create_instance(
                offer_id=offer_id,
                image="pytorch/pytorch:latest",  # Default image
                disk_size=20,
                label=f"autogen-{int(time.time())}",
            )

            self.rented_instance = RentedInstance(
                instance_id=instance.id,
                gpu_name=instance.gpu_name,
                price_per_hour=instance.price_per_hour,
                ssh_host=instance.ssh_host,
                ssh_port=instance.ssh_port,
                ssh_user=instance.ssh_user,
            )

            logger.info(f"✓ Instance rented: {self.rented_instance.instance_id}")
            logger.info(f"  SSH: {self.ssh_user}@{self.ssh_host}:{self.ssh_port}")
            logger.info(f"  Cost: ${price:.4f}/hr")

            return self.rented_instance

        except Exception as e:
            logger.error(f"Failed to rent instance: {e}")
            raise

    def wait_for_ssh_ready(self, timeout: int = 300, poll_interval: int = 5) -> bool:
        """
        Wait for SSH to be ready on the rented instance.

        Args:
            timeout: Maximum seconds to wait
            poll_interval: Seconds between connection attempts

        Returns:
            True if SSH is ready, False if timeout.
        """
        if not self.rented_instance:
            raise RuntimeError("No instance rented. Call rent_cheapest() first.")

        logger.info(f"Waiting for SSH to be ready (timeout: {timeout}s)...")

        start_time = time.time()
        attempt = 0

        while time.time() - start_time < timeout:
            attempt += 1
            try:
                ssh = SSHClient(
                    host=self.rented_instance.ssh_host,
                    port=self.rented_instance.ssh_port,
                    username=self.rented_instance.ssh_user,
                    private_key_path=self.ssh_key_path,
                )
                ssh.connect()
                exit_code, stdout, stderr = ssh.execute("echo 'SSH ready'")
                ssh.disconnect()

                logger.info(f"✓ SSH ready after {attempt} attempts")
                self.ssh_client = ssh
                return True

            except Exception as e:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.error(f"✗ SSH timeout after {elapsed:.0f}s")
                    return False
                logger.debug(f"  Attempt {attempt} failed: {type(e).__name__}, retrying...")
                time.sleep(poll_interval)

        return False

    def connect(self) -> SSHClient:
        """
        Get SSH client for the rented instance.

        Returns:
            Connected SSHClient.

        Raises:
            RuntimeError: If no instance rented or connection failed.
        """
        if not self.rented_instance:
            raise RuntimeError("No instance rented. Call rent_cheapest() first.")

        if self.ssh_client is None:
            self.ssh_client = SSHClient(
                host=self.rented_instance.ssh_host,
                port=self.rented_instance.ssh_port,
                username=self.rented_instance.ssh_user,
                private_key_path=self.ssh_key_path,
            )
            self.ssh_client.connect()

        return self.ssh_client

    def destroy_instance(self) -> bool:
        """
        Destroy the rented instance (stops billing).

        Returns:
            True if successfully destroyed.
        """
        if not self.rented_instance:
            logger.warning("No instance to destroy")
            return False

        instance_id = self.rented_instance.instance_id
        logger.info(f"Destroying instance {instance_id}...")

        try:
            success = self.client.destroy_instance(instance_id)
            if success:
                logger.info(f"✓ Instance {instance_id} destroyed")
                self.rented_instance = None
                if self.ssh_client:
                    self.ssh_client.disconnect()
                    self.ssh_client = None
            return success
        except Exception as e:
            logger.error(f"Failed to destroy instance: {e}")
            return False

    def get_instance_info(self) -> Optional[RentedInstance]:
        """Get current rented instance info."""
        return self.rented_instance
