import yaml
from pathlib import Path
from typing import Any, Dict

def load_kaggle_config() -> Dict[str, Any]:
    """
    Load Kaggle configuration from YAML file.
    """
    config_path = Path(__file__).parent.parent / "config" / "kaggle_settings.yaml"
    if not config_path.exists():
        # Fallback to defaults if file not found
        return {
            "gpu_limit": 2,
            "deployment_timeout_minutes": 30,
            "polling_interval_seconds": 60,
            "retry_interval_seconds": 60
        }
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
