"""
Configuration module for Vast.ai integration.
Loads and validates environment variables for API keys and settings.
"""
import os
from pathlib import Path
from typing import Optional


class VastAiConfig:
    """Load and validate Vast.ai configuration from environment."""

    def __init__(self, env_file: Optional[Path] = None):
        """
        Initialize configuration.

        Args:
            env_file: Path to .env file. If None, looks for .env in project root.

        Raises:
            ValueError: If VAST_AI_API_KEY is not set.
        """
        self._load_env_file(env_file)
        self.api_key = self._get_required_env("VAST_AI_API_KEY")
        self.base_url = os.getenv("VAST_AI_BASE_URL", "https://console.vast.ai/api/v0/")
        self.timeout = int(os.getenv("VAST_AI_TIMEOUT", "30"))

    def _load_env_file(self, env_file: Optional[Path]) -> None:
        """Load .env file if it exists."""
        if env_file is None:
            env_file = Path(__file__).parent.parent.parent.parent.parent / ".env"

        if env_file.exists():
            self._parse_env_file(env_file)

    @staticmethod
    def _parse_env_file(env_file: Path) -> None:
        """Parse and load .env file into os.environ."""
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip()
                    # Remove surrounding quotes if present
                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    os.environ.setdefault(key.strip(), value)

    @staticmethod
    def _get_required_env(key: str) -> str:
        """
        Get required environment variable.

        Args:
            key: Environment variable name.

        Returns:
            Environment variable value.

        Raises:
            ValueError: If variable is not set or empty.
        """
        value = os.getenv(key)
        
        # Try alternate naming for VAST_AI_API_KEY
        if not value and key == "VAST_AI_API_KEY":
            value = os.getenv("VASTAI_API_KEY")
        
        if not value or not value.strip():
            raise ValueError(
                f"Required environment variable '{key}' is not set. "
                f"Add it to .env file in project root."
            )
        return value.strip()


def get_vast_ai_config(env_file: Optional[Path] = None) -> VastAiConfig:
    """
    Get Vast.ai configuration singleton.

    Args:
        env_file: Path to .env file (for testing).

    Returns:
        VastAiConfig instance.
    """
    return VastAiConfig(env_file)
