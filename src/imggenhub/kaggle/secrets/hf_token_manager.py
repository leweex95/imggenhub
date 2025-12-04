"""
HF Token Manager - Syncs HuggingFace token between local .env and Kaggle dataset.

This module handles:
1. Reading HF_TOKEN from local .env file
2. Comparing it with the token stored in a remote Kaggle dataset
3. Creating or updating the Kaggle dataset when the token changes
4. Providing the token for use in Kaggle notebooks via dataset attachment

The Kaggle dataset stores the token in a JSON file that notebooks can read at runtime.
This eliminates the need to inject tokens into notebook source code.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

# Dataset identifier for the HF token
HF_TOKEN_DATASET_ID = "leventecsibi/imggenhub-hf-token"
HF_TOKEN_FILENAME = "hf_token.json"


class HFTokenManager:
    """Manages HuggingFace token synchronization between local .env and Kaggle dataset."""

    def __init__(self, env_path: Optional[Path] = None):
        """
        Initialize the HF Token Manager.

        Args:
            env_path: Path to the .env file. If None, searches in standard locations.

        Raises:
            FileNotFoundError: If no .env file is found.
            ValueError: If HF_TOKEN is not set in the .env file.
        """
        self.api = KaggleApi()
        self.api.authenticate()

        self.env_path = self._find_env_file(env_path)
        self.local_token = self._load_local_token()

    def _find_env_file(self, env_path: Optional[Path] = None) -> Path:
        """
        Find the .env file.

        Args:
            env_path: Explicit path to .env file, or None to search.

        Returns:
            Path to the .env file.

        Raises:
            FileNotFoundError: If no .env file is found.
        """
        if env_path:
            if not env_path.exists():
                raise FileNotFoundError(f".env file not found at: {env_path}")
            return env_path

        # Search in common locations
        search_paths = [
            Path.cwd() / ".env",
            Path(__file__).parents[4] / ".env",  # Project root
        ]

        for path in search_paths:
            if path.exists():
                logging.info(f"Found .env file at: {path}")
                return path

        raise FileNotFoundError(
            f".env file not found. Searched: {[str(p) for p in search_paths]}"
        )

    def _load_local_token(self) -> str:
        """
        Load HF_TOKEN from the local .env file.

        Returns:
            The HF token string.

        Raises:
            ValueError: If HF_TOKEN is not set or is empty.
        """
        load_dotenv(self.env_path, override=True)
        token = os.getenv("HF_TOKEN")

        if not token:
            raise ValueError(
                f"HF_TOKEN is not set in {self.env_path}. "
                "Please add HF_TOKEN=\"your_token\" to the .env file."
            )

        # Remove quotes if present (dotenv might not strip them)
        token = token.strip('"').strip("'")

        if not token:
            raise ValueError(f"HF_TOKEN is empty in {self.env_path}")

        return token

    def get_remote_token(self) -> Optional[str]:
        """
        Retrieve the HF token from the remote Kaggle dataset.

        Returns:
            The remote token string, or None if dataset doesn't exist or is empty.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                self.api.dataset_download_file(
                    dataset=HF_TOKEN_DATASET_ID,
                    file_name=HF_TOKEN_FILENAME,
                    path=tmpdir,
                    force=True,
                    quiet=True,
                )
                token_file = Path(tmpdir) / HF_TOKEN_FILENAME
                if token_file.exists():
                    with open(token_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return data.get("HF_TOKEN")
            except Exception as e:
                logging.info(f"Could not fetch remote token (dataset may not exist): {e}")
                return None

        return None

    def _create_dataset_files(self, tmpdir: str) -> None:
        """
        Create the dataset files in a temporary directory.

        Args:
            tmpdir: Path to temporary directory for dataset files.
        """
        # Create the token JSON file
        token_file = Path(tmpdir) / HF_TOKEN_FILENAME
        with open(token_file, "w", encoding="utf-8") as f:
            json.dump({"HF_TOKEN": self.local_token}, f, indent=2)

        # Create dataset-metadata.json
        metadata = {
            "title": "imggenhub HF Token",
            "id": HF_TOKEN_DATASET_ID,
            "licenses": [{"name": "CC0-1.0"}],
        }
        metadata_file = Path(tmpdir) / "dataset-metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _dataset_exists(self) -> bool:
        """Check if the HF token dataset exists on Kaggle."""
        try:
            datasets = self.api.dataset_list(user="leventecsibi", search="imggenhub-hf-token")
            for ds in datasets:
                if ds.ref == HF_TOKEN_DATASET_ID:
                    return True
        except Exception:
            pass
        return False

    def create_dataset(self) -> None:
        """
        Create a new Kaggle dataset with the HF token.

        Raises:
            RuntimeError: If dataset creation fails.
        """
        logging.info(f"Creating new Kaggle dataset: {HF_TOKEN_DATASET_ID}")

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_dataset_files(tmpdir)

            try:
                response = self.api.dataset_create_new(
                    folder=tmpdir,
                    public=False,
                    quiet=False,
                    convert_to_csv=False,
                    dir_mode="skip",
                )
                logging.info(f"Dataset created successfully: {response}")
            except Exception as e:
                raise RuntimeError(f"Failed to create dataset: {e}") from e

    def update_dataset(self) -> None:
        """
        Update the existing Kaggle dataset with the new HF token.

        Raises:
            RuntimeError: If dataset update fails.
        """
        logging.info(f"Updating Kaggle dataset: {HF_TOKEN_DATASET_ID}")

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_dataset_files(tmpdir)

            try:
                response = self.api.dataset_create_version(
                    folder=tmpdir,
                    version_notes="Updated HF token",
                    quiet=False,
                    convert_to_csv=False,
                    delete_old_versions=True,
                    dir_mode="skip",
                )
                logging.info(f"Dataset updated successfully: {response}")
            except Exception as e:
                raise RuntimeError(f"Failed to update dataset: {e}") from e

    def sync(self) -> bool:
        """
        Sync the local HF token to the Kaggle dataset.

        Compares local and remote tokens, and updates the dataset if they differ.

        Returns:
            True if the dataset was updated, False if no update was needed.

        Raises:
            RuntimeError: If sync fails.
        """
        remote_token = self.get_remote_token()

        if remote_token == self.local_token:
            logging.info("HF token is already in sync with Kaggle dataset")
            return False

        if remote_token is None:
            # Dataset doesn't exist or is empty
            if self._dataset_exists():
                logging.info("Dataset exists but token file is missing or empty, updating...")
                self.update_dataset()
            else:
                logging.info("Dataset does not exist, creating...")
                self.create_dataset()
        else:
            logging.info("HF token has changed, updating dataset...")
            self.update_dataset()

        return True


def sync_hf_token(env_path: Optional[Path] = None) -> bool:
    """
    Convenience function to sync HF token to Kaggle dataset.

    Args:
        env_path: Optional path to .env file.

    Returns:
        True if dataset was updated, False if no update was needed.
    """
    manager = HFTokenManager(env_path)
    return manager.sync()
