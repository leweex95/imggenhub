"""Forge UI orchestration and management for remote Vast.ai instances."""
import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path
import json
import re

from imggenhub.vast_ai.ssh import SSHClient
from imggenhub.vast_ai.client import VastInstance

logger = logging.getLogger(__name__)


class ForgeUIManager:
    """Manages Forge UI deployment and inference on remote instances."""

    FORGE_REPO = "https://github.com/lllyasviel/stable-diffusion-webui-forge"
    FORGE_DIR = "/workspace/stable-diffusion-webui-forge"
    MODELS_DIR = f"{FORGE_DIR}/models"
    API_PORT = 7860

    def __init__(self, instance: VastInstance, ssh_private_key: Optional[str] = None, ssh_password: Optional[str] = None):
        """
        Initialize Forge UI manager.

        Args:
            instance: VastInstance object with SSH connection details. Required.
            ssh_private_key: Path to SSH private key. Optional if using password.
            ssh_password: SSH password. Optional if using private key.

        Raises:
            ValueError: If instance is invalid.
        """
        if not instance:
            raise ValueError("instance must be a VastInstance object")

        self.instance = instance
        self.ssh_client = SSHClient(
            host=instance.ssh_host,
            port=instance.ssh_port,
            username=instance.ssh_user,
            private_key_path=ssh_private_key,
            password=ssh_password,
        )
        self.forge_running = False

    def deploy(self) -> None:
        """
        Deploy Forge UI on remote instance.

        Raises:
            RuntimeError: If deployment fails.
        """
        logger.info("Deploying Forge UI...")

        try:
            self.ssh_client.connect()

            # Upload deployment script
            logger.info("Uploading deployment script...")
            deploy_script = Path(__file__).parent / "forge_deploy.sh"
            if not deploy_script.exists():
                raise FileNotFoundError(f"Deployment script not found: {deploy_script}")

            self.ssh_client.upload_file(str(deploy_script), "/tmp/forge_deploy.sh")

            # Run deployment script
            logger.info("Running deployment script...")
            exit_code, stdout, stderr = self.ssh_client.execute("bash /tmp/forge_deploy.sh")

            if exit_code != 0:
                logger.error(f"Deployment failed with exit code {exit_code}")
                logger.error(f"Stderr: {stderr}")
                raise RuntimeError(f"Forge UI deployment failed: {stderr}")

            logger.info("Forge UI deployment complete")
            logger.info(stdout)

        except Exception as e:
            logger.error(f"Forge UI deployment failed: {e}")
            raise
        finally:
            self.ssh_client.disconnect()

    def start_server(self, listen_port: int = 7860) -> None:
        """
        Start Forge UI server on remote instance (background process).

        Args:
            listen_port: Port to listen on. Default: 7860.

        Raises:
            ValueError: If port is invalid.
            RuntimeError: If server fails to start.
        """
        if not isinstance(listen_port, int) or listen_port <= 0 or listen_port > 65535:
            raise ValueError("listen_port must be an integer between 1 and 65535")

        logger.info(f"Starting Forge UI server on port {listen_port}...")

        try:
            self.ssh_client.connect()

            # Start server in background with nohup
            command = (
                f"cd {self.FORGE_DIR} && "
                f"nohup python launch.py --listen 0.0.0.0 --port {listen_port} "
                f"> /tmp/forge_server.log 2>&1 &"
            )

            exit_code, stdout, stderr = self.ssh_client.execute(command)
            if exit_code != 0:
                raise RuntimeError(f"Failed to start server: {stderr}")

            # Wait for server to start
            logger.info("Waiting for server to start...")
            time.sleep(5)

            # Verify server is running
            verify_cmd = f"curl -s http://localhost:{listen_port}/api/system/info || echo 'Not ready'"
            exit_code, stdout, stderr = self.ssh_client.execute(verify_cmd)

            if "Not ready" in stdout or exit_code != 0:
                logger.warning("Server may still be starting, waiting...")
                time.sleep(10)

            logger.info(f"Forge UI server started on port {listen_port}")
            self.forge_running = True

        except Exception as e:
            logger.error(f"Failed to start Forge UI server: {e}")
            raise
        finally:
            self.ssh_client.disconnect()

    def stop_server(self) -> None:
        """
        Stop Forge UI server on remote instance.

        Raises:
            RuntimeError: If stop fails.
        """
        logger.info("Stopping Forge UI server...")

        try:
            self.ssh_client.connect()
            self.ssh_client.execute("pkill -f 'python launch.py'")
            self.forge_running = False
            logger.info("Forge UI server stopped")
        except Exception as e:
            logger.error(f"Failed to stop Forge UI server: {e}")
            raise
        finally:
            self.ssh_client.disconnect()

    def download_model(self, model_id: str, model_type: str = "checkpoint") -> None:
        """
        Download a model to Forge models directory.

        Args:
            model_id: Model ID from HuggingFace (e.g., "black-forest-labs/FLUX.1-schnell"). Required.
            model_type: Type of model (checkpoint|lora|vae|controlnet). Default: checkpoint.

        Raises:
            ValueError: If parameters are invalid.
            RuntimeError: If download fails.
        """
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string")
        if model_type not in ["checkpoint", "lora", "vae", "controlnet"]:
            raise ValueError(f"model_type must be one of: checkpoint, lora, vae, controlnet. Got: {model_type}")

        logger.info(f"Downloading {model_type}: {model_id}...")

        try:
            self.ssh_client.connect()

            # Use huggingface_hub to download model
            download_cmd = (
                f"python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('{model_id}', "
                f"cache_dir='{self.MODELS_DIR}/{model_type}s', "
                f"revision='main')\""
            )

            exit_code, stdout, stderr = self.ssh_client.execute(download_cmd)

            if exit_code != 0:
                raise RuntimeError(f"Model download failed: {stderr}")

            logger.info(f"{model_type} downloaded successfully")

        except Exception as e:
            logger.error(f"Model download failed: {e}")
            raise
        finally:
            self.ssh_client.disconnect()

    def generate_image(
        self,
        prompt: str,
        model: str,
        steps: int = 50,
        guidance: float = 7.5,
        width: int = 512,
        height: int = 512,
        negative_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an image using Forge UI API.

        Args:
            prompt: Image generation prompt. Required.
            model: Model checkpoint name. Required.
            steps: Number of inference steps. Default: 50.
            guidance: Guidance scale. Default: 7.5.
            width: Image width. Default: 512.
            height: Image height. Default: 512.
            negative_prompt: Negative prompt. Optional.

        Returns:
            Dictionary with generation results including image data.

        Raises:
            ValueError: If parameters are invalid.
            RuntimeError: If generation fails.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")

        logger.info(f"Generating image with prompt: {prompt[:50]}...")

        if not self.forge_running:
            raise RuntimeError("Forge UI server is not running. Call start_server() first.")

        try:
            self.ssh_client.connect()

            # Build API request
            payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "steps": steps,
                "cfg_scale": guidance,
                "width": width,
                "height": height,
                "checkpoint": model,
            }

            payload_json = json.dumps(payload).replace('"', '\\"')

            # Call API
            api_cmd = (
                f"curl -s -X POST http://localhost:{self.API_PORT}/api/txt2img "
                f"-H 'Content-Type: application/json' "
                f"-d '{payload_json}'"
            )

            exit_code, stdout, stderr = self.ssh_client.execute(api_cmd)

            if exit_code != 0:
                raise RuntimeError(f"Image generation failed: {stderr}")

            result = json.loads(stdout)
            logger.info("Image generation complete")
            return result

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
        finally:
            self.ssh_client.disconnect()

    def get_server_status(self) -> Dict[str, Any]:
        """
        Get Forge UI server status and information.

        Returns:
            Dictionary with server status and system info.

        Raises:
            RuntimeError: If status check fails.
        """
        logger.info("Checking Forge UI server status...")

        try:
            self.ssh_client.connect()

            status_cmd = f"curl -s http://localhost:{self.API_PORT}/api/system/info"
            exit_code, stdout, stderr = self.ssh_client.execute(status_cmd)

            if exit_code != 0 or not stdout:
                return {"status": "offline", "error": stderr}

            try:
                status = json.loads(stdout)
                return {"status": "online", **status}
            except json.JSONDecodeError:
                return {"status": "unknown", "response": stdout}

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            raise
        finally:
            self.ssh_client.disconnect()

    def get_models_list(self) -> Dict[str, list]:
        """
        List available models on Forge UI.

        Returns:
            Dictionary with lists of available checkpoints, LoRAs, VAEs, and ControlNets.

        Raises:
            RuntimeError: If retrieval fails.
        """
        logger.info("Fetching available models...")

        if not self.forge_running:
            raise RuntimeError("Forge UI server is not running. Call start_server() first.")

        try:
            self.ssh_client.connect()

            models_cmd = f"curl -s http://localhost:{self.API_PORT}/api/sd-models"
            exit_code, stdout, stderr = self.ssh_client.execute(models_cmd)

            if exit_code != 0:
                raise RuntimeError(f"Failed to fetch models: {stderr}")

            models = json.loads(stdout)
            return models

        except Exception as e:
            logger.error(f"Failed to retrieve models list: {e}")
            raise
        finally:
            self.ssh_client.disconnect()

    def set_checkpoint(self, model_name: str) -> None:
        """
        Set active checkpoint for image generation.

        Args:
            model_name: Model checkpoint name to activate. Required.

        Raises:
            ValueError: If model_name is invalid.
            RuntimeError: If checkpoint setting fails.
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")

        logger.info(f"Setting checkpoint to: {model_name}")

        try:
            self.ssh_client.connect()

            payload = {"sd_model_checkpoint": model_name}
            payload_json = json.dumps(payload).replace('"', '\\"')

            set_cmd = (
                f"curl -s -X POST http://localhost:{self.API_PORT}/api/options "
                f"-H 'Content-Type: application/json' "
                f"-d '{payload_json}'"
            )

            exit_code, stdout, stderr = self.ssh_client.execute(set_cmd)

            if exit_code != 0:
                raise RuntimeError(f"Failed to set checkpoint: {stderr}")

            logger.info(f"Checkpoint set to: {model_name}")

        except Exception as e:
            logger.error(f"Failed to set checkpoint: {e}")
            raise
        finally:
            self.ssh_client.disconnect()
