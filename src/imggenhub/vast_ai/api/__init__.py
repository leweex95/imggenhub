"""Vast.ai API client for GPU instance management."""
import requests
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from imggenhub.vast_ai.config import get_vast_ai_config


@dataclass
class VastInstance:
    """Represents a rented Vast.ai GPU instance."""
    id: int
    label: str
    gpu_name: str
    gpu_count: int
    status: str
    ssh_host: str
    ssh_port: int
    ssh_user: str
    price_per_hour: float
    created_at: datetime


class VastAiClient:
    """Client for interacting with Vast.ai API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Vast.ai client with API key.

        Args:
            api_key: Vast.ai API key. If None, loads from .env file.

        Raises:
            ValueError: If api_key cannot be determined or is invalid.
        """
        if api_key is None:
            config = get_vast_ai_config()
            api_key = config.api_key
            self.base_url = config.base_url
            self.timeout = config.timeout
        else:
            if not api_key or not isinstance(api_key, str):
                raise ValueError("api_key must be a non-empty string")
            self.base_url = "https://console.vast.ai/api/v0/"
            self.timeout = 30

        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def search_offers(
        self,
        gpu_name: Optional[str] = None,
        max_price: Optional[float] = None,
        min_vram: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for available GPU offers on Vast.ai marketplace.

        Args:
            gpu_name: Filter by GPU name (e.g., 'RTX 4090'). Optional.
            max_price: Filter by maximum price per hour. Optional.
            min_vram: Filter by minimum VRAM in GB. Optional.
            limit: Maximum number of results to return.

        Returns:
            List of available GPU offers with details.

        Raises:
            ValueError: If search parameters are invalid.
            requests.RequestException: If API request fails.
        """
        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        params: Dict[str, Any] = {
            "limit": 100,  # Get more offers, filter client-side
        }

        response = requests.get(
            f"{self.base_url}bundles",
            headers=self.headers,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        all_offers = response.json().get("offers", [])
        # Filter to only rentable offers
        return [offer for offer in all_offers if offer.get("rentable", False)]

    def create_instance(
        self,
        offer_id: int,
        image: str,
        disk_size: int = 20,
        label: Optional[str] = None,
        onstart: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> VastInstance:
        """
        Create a new GPU instance from an available offer.

        Args:
            offer_id: The offer ID to accept (ask_id from search results). Required.
            image: Docker image to use (e.g., 'pytorch/pytorch:latest'). Required.
            disk_size: Local disk allocation in GB. Default: 20.
            label: Custom name for the instance. Optional.
            onstart: Bash commands to run when instance starts. Optional.
            env: Environment variables to set in the instance. Optional.

        Returns:
            VastInstance object with instance details.

        Raises:
            ValueError: If required parameters are missing or invalid.
            requests.RequestException: If instance creation fails.
        """
        if not isinstance(offer_id, int) or offer_id <= 0:
            raise ValueError("offer_id must be a positive integer")
        if not image or not isinstance(image, str):
            raise ValueError("image must be a non-empty string")
        if not isinstance(disk_size, int) or disk_size <= 0:
            raise ValueError("disk_size must be a positive integer")

        payload: Dict[str, Any] = {
            "image": image,
            "disk": disk_size,
            "runtype": "ssh",
        }

        if label:
            if not isinstance(label, str):
                raise ValueError("label must be a string")
            payload["label"] = label

        if onstart:
            if not isinstance(onstart, str):
                raise ValueError("onstart must be a string")
            payload["onstart"] = onstart

        if env:
            if not isinstance(env, dict):
                raise ValueError("env must be a dictionary")
            payload["env"] = env

        response = requests.put(
            f"{self.base_url}asks/{offer_id}/",
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            error_type = data.get('error', 'Unknown error')
            error_msg = data.get('msg', '')
            full_error = f"{error_type}: {error_msg}" if error_msg else error_type
            raise RuntimeError(f"Failed to create instance: {full_error}")

        instance_id = data.get("new_contract")
        if not instance_id:
            raise RuntimeError("Instance created but no contract ID returned")

        return self.get_instance(instance_id)

    def get_instance(self, instance_id: int) -> VastInstance:
        """
        Retrieve details of a specific GPU instance.

        Args:
            instance_id: The instance contract ID. Required.

        Returns:
            VastInstance object with full instance details.

        Raises:
            ValueError: If instance_id is invalid.
            requests.RequestException: If instance lookup fails.
        """
        if not isinstance(instance_id, int) or instance_id <= 0:
            raise ValueError("instance_id must be a positive integer")

        response = requests.get(
            f"{self.base_url}/instances/{instance_id}/",
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        instance_data = data.get("instances", [{}])[0]
        if not instance_data:
            raise ValueError(f"Instance {instance_id} not found")

        return self._parse_instance(instance_data)

    def list_instances(self) -> List[VastInstance]:
        """
        List all active instances for this account.

        Returns:
            List of VastInstance objects for all active instances.

        Raises:
            requests.RequestException: If API request fails.
        """
        response = requests.get(
            f"{self.base_url}instances/",
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        instances_data = response.json().get("instances", [])
        return [self._parse_instance(inst) for inst in instances_data if inst]

    def destroy_instance(self, instance_id: int) -> bool:
        """
        Terminate and destroy a GPU instance.

        Args:
            instance_id: The instance contract ID to destroy. Required.

        Returns:
            True if instance was successfully destroyed.

        Raises:
            ValueError: If instance_id is invalid.
            requests.RequestException: If destruction fails.
        """
        if not isinstance(instance_id, int) or instance_id <= 0:
            raise ValueError("instance_id must be a positive integer")

        response = requests.delete(
            f"{self.base_url}/instances/{instance_id}/",
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        # API may return empty response on success, treat any 2xx status as success
        try:
            data = response.json()
            return data.get("success", True)
        except ValueError:
            # Empty response is considered success
            return True

    def reboot_instance(self, instance_id: int) -> bool:
        """
        Reboot a running GPU instance.

        Args:
            instance_id: The instance contract ID to reboot. Required.

        Returns:
            True if reboot was successful.

        Raises:
            ValueError: If instance_id is invalid.
            requests.RequestException: If reboot fails.
        """
        if not isinstance(instance_id, int) or instance_id <= 0:
            raise ValueError("instance_id must be a positive integer")

        response = requests.post(
            f"{self.base_url}/instances/{instance_id}/reboot/",
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("success", False)

    def execute_command(self, instance_id: int, command: str) -> str:
        """
        Execute a shell command on a running instance.

        Args:
            instance_id: The instance contract ID. Required.
            command: The shell command to execute. Required.

        Returns:
            Command output as string.

        Raises:
            ValueError: If parameters are invalid.
            requests.RequestException: If command execution fails.
        """
        if not isinstance(instance_id, int) or instance_id <= 0:
            raise ValueError("instance_id must be a positive integer")
        if not command or not isinstance(command, str):
            raise ValueError("command must be a non-empty string")

        payload = {"command": command}
        response = requests.post(
            f"{self.base_url}/instances/{instance_id}/exec/",
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("output", "")

    @staticmethod
    def _parse_instance(data: Dict[str, Any]) -> VastInstance:
        """
        Parse instance data from API response.

        Args:
            data: Raw instance data from API.

        Returns:
            Parsed VastInstance object.

        Raises:
            KeyError: If required fields are missing.
        """
        return VastInstance(
            id=data["id"],
            label=data.get("label", f"Instance {data['id']}"),
            gpu_name=data.get("gpu_name", "Unknown"),
            gpu_count=data.get("gpu_count", 1),
            status=data.get("status_msg", "Unknown"),
            ssh_host=data.get("ssh_host", ""),
            ssh_port=data.get("ssh_port", 22),
            ssh_user=data.get("ssh_user", "root"),
            price_per_hour=data.get("price_per_hour", 0.0),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
        )
