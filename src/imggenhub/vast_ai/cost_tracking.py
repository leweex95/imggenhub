"""
Cost tracking and instance lifecycle management for Vast.ai GPU rentals.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CostEstimator:
    """Estimate costs for GPU rentals based on instance specs and duration."""

    @staticmethod
    def estimate_rental_cost(
        instance: Dict[str, Any],
        duration_minutes: int,
    ) -> Dict[str, float]:
        """
        Estimate rental cost for a given instance.

        Args:
            instance: Instance dict with 'dph_total' (price per hour)
            duration_minutes: Duration in minutes

        Returns:
            Dict with 'total_cost', 'hourly_rate', 'duration_hours'
        """
        hourly_rate = instance.get('dph_total', 0.0)
        duration_hours = duration_minutes / 60.0

        return {
            'hourly_rate': hourly_rate,
            'duration_hours': duration_hours,
            'total_cost': hourly_rate * duration_hours,
        }

    @staticmethod
    def estimate_generation_cost(
        num_images: int,
        avg_time_per_image: float,
        instance_dph: float,
    ) -> Dict[str, float]:
        """
        Estimate cost for generating N images.

        Args:
            num_images: Number of images to generate
            avg_time_per_image: Average seconds per image
            instance_dph: Instance price per hour

        Returns:
            Dict with cost breakdown
        """
        total_seconds = num_images * avg_time_per_image
        total_minutes = total_seconds / 60.0
        total_hours = total_minutes / 60.0

        return {
            'num_images': num_images,
            'avg_time_per_image_sec': avg_time_per_image,
            'total_duration_sec': total_seconds,
            'total_duration_min': total_minutes,
            'total_duration_hr': total_hours,
            'instance_dph': instance_dph,
            'total_cost': instance_dph * total_hours,
            'cost_per_image': (instance_dph * total_hours) / num_images if num_images > 0 else 0,
        }

    @staticmethod
    def recommend_model_for_budget(
        budget_usd: float,
        instance_dph: float,
    ) -> Dict[str, Any]:
        """
        Recommend models based on budget and GPU price.

        Args:
            budget_usd: Available budget in USD
            instance_dph: Instance price per hour

        Returns:
            Dict with model recommendations
        """
        budget_minutes = (budget_usd / instance_dph) * 60
        budget_images_flux_schnell = int(budget_minutes / 0.5)  # ~30 sec per image
        budget_images_sd35 = int(budget_minutes / 1.0)  # ~60 sec per image
        budget_images_flux_pro = int(budget_minutes / 1.5)  # ~90 sec per image

        return {
            'budget_usd': budget_usd,
            'budget_minutes': budget_minutes,
            'instance_dph': instance_dph,
            'recommendations': [
                {
                    'model': 'black-forest-labs/FLUX.1-schnell',
                    'speed': 'fastest',
                    'quality': 'good',
                    'estimated_images': budget_images_flux_schnell,
                    'avg_time_sec': 30,
                    'guidance': '3.5-5.0',
                    'steps': '10-20',
                },
                {
                    'model': 'stabilityai/stable-diffusion-3.5-large',
                    'speed': 'medium',
                    'quality': 'very-good',
                    'estimated_images': budget_images_sd35,
                    'avg_time_sec': 60,
                    'guidance': '7.5-9.0',
                    'steps': '30-40',
                },
                {
                    'model': 'black-forest-labs/FLUX.1-pro',
                    'speed': 'slow',
                    'quality': 'best',
                    'estimated_images': budget_images_flux_pro,
                    'avg_time_sec': 90,
                    'guidance': '7.5-12.0',
                    'steps': '30-50',
                },
            ],
        }


class RunLogger:
    """Log and track individual GPU run metrics."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.runs: list[Dict[str, Any]] = []

    def log_run(
        self,
        instance_id: str,
        instance_type: str,
        instance_dph: float,
        model_name: str,
        num_images: int,
        duration_seconds: float,
        status: str = "completed",
        notes: str = None,
    ) -> None:
        """
        Log a single GPU run.

        Args:
            instance_id: Vast.ai instance ID
            instance_type: GPU type (e.g., 'nvidia-tesla-p40')
            instance_dph: Price per hour
            model_name: Model used for generation
            num_images: Number of images generated
            duration_seconds: Total run duration in seconds
            status: Run status (completed, failed, cancelled)
            notes: Additional notes
        """
        if not instance_id:
            raise ValueError("instance_id is required")
        if not instance_dph or instance_dph < 0:
            raise ValueError("instance_dph must be positive")
        if not num_images or num_images < 0:
            raise ValueError("num_images must be non-negative")
        if not duration_seconds or duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")

        duration_hours = duration_seconds / 3600
        total_cost = instance_dph * duration_hours
        cost_per_image = total_cost / num_images if num_images > 0 else 0

        run_entry = {
            'timestamp': datetime.now().isoformat(),
            'instance_id': instance_id,
            'instance_type': instance_type,
            'instance_dph': instance_dph,
            'model_name': model_name,
            'num_images': num_images,
            'duration_seconds': duration_seconds,
            'duration_hours': duration_hours,
            'total_cost': total_cost,
            'cost_per_image': cost_per_image,
            'status': status,
            'notes': notes,
        }

        self.runs.append(run_entry)
        logger.info(f"Run logged: {num_images} images, ${total_cost:.4f} (${cost_per_image:.4f}/img)")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all logged runs."""
        if not self.runs:
            return {
                'total_runs': 0,
                'total_cost': 0.0,
                'total_images': 0,
                'total_duration_hours': 0.0,
                'avg_cost_per_image': 0.0,
            }

        total_cost = sum(r['total_cost'] for r in self.runs)
        total_images = sum(r['num_images'] for r in self.runs)
        total_duration = sum(r['duration_seconds'] for r in self.runs)

        return {
            'total_runs': len(self.runs),
            'total_cost': total_cost,
            'total_images': total_images,
            'total_duration_hours': total_duration / 3600,
            'avg_cost_per_image': total_cost / total_images if total_images > 0 else 0,
            'runs': self.runs,
        }

    def save_log(self, filepath: Optional[Path] = None) -> Path:
        """Save run log to JSON file."""
        if not filepath:
            filepath = self.output_dir / f"vast_ai_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        log_data = {
            'generated': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'runs': self.runs,
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Run log saved to: {filepath}")
        return filepath

    def load_log(self, filepath: Path) -> None:
        """Load existing run log from JSON file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Run log not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)
            self.runs = data.get('runs', [])

        logger.info(f"Loaded {len(self.runs)} runs from: {filepath}")


class InstanceManager:
    """Manage instance lifecycle and ensure proper cleanup."""

    def __init__(self, client):
        """Initialize with a VastAIClient instance."""
        self.client = client
        self.active_instances: Dict[str, Dict[str, Any]] = {}

    def track_instance(
        self,
        instance_id: str,
        instance_type: str,
        instance_dph: float,
        started_at: datetime,
    ) -> None:
        """Track an active instance."""
        if not instance_id:
            raise ValueError("instance_id is required")
        if not instance_type:
            raise ValueError("instance_type is required")
        if not instance_dph or instance_dph < 0:
            raise ValueError("instance_dph must be positive")

        self.active_instances[instance_id] = {
            'instance_type': instance_type,
            'instance_dph': instance_dph,
            'started_at': started_at,
        }

        logger.info(f"Tracking instance {instance_id} ({instance_type}) at ${instance_dph}/hr")

    def untrack_instance(self, instance_id: str) -> None:
        """Stop tracking an instance."""
        if instance_id in self.active_instances:
            del self.active_instances[instance_id]
            logger.info(f"Untracked instance {instance_id}")

    def cleanup_all(self) -> None:
        """Destroy all tracked active instances."""
        if not self.active_instances:
            logger.info("No active instances to clean up")
            return

        logger.warning(f"Destroying {len(self.active_instances)} active instances...")

        for instance_id in list(self.active_instances.keys()):
            try:
                self.client.destroy_instance(instance_id)
                self.untrack_instance(instance_id)
                logger.info(f"Destroyed instance {instance_id}")
            except Exception as e:
                logger.error(f"Failed to destroy instance {instance_id}: {e}")

    def get_estimated_cost(self, instance_id: str, duration_seconds: float) -> Dict[str, float]:
        """Estimate cost for a given instance and duration."""
        if instance_id not in self.active_instances:
            raise ValueError(f"Instance {instance_id} is not being tracked")

        instance_info = self.active_instances[instance_id]
        instance_dph = instance_info['instance_dph']
        duration_hours = duration_seconds / 3600

        return {
            'hourly_rate': instance_dph,
            'duration_hours': duration_hours,
            'estimated_cost': instance_dph * duration_hours,
        }

    def get_active_instances(self) -> list[Dict[str, Any]]:
        """Get list of all active tracked instances."""
        return [
            {
                'instance_id': iid,
                'info': info,
                'uptime_seconds': (datetime.now() - info['started_at']).total_seconds(),
            }
            for iid, info in self.active_instances.items()
        ]
