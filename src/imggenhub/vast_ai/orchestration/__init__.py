"""Remote pipeline orchestration for Vast.ai GPU instances."""
import logging
from pathlib import Path
from typing import Optional, Tuple
from imggenhub.vast_ai.api import VastAiClient, VastInstance
from imggenhub.vast_ai.ssh import SSHClient

logger = logging.getLogger(__name__)


class RemoteExecutor:
    """Manages remote execution of image generation pipeline on Vast.ai instances."""

    def __init__(self, api_key: Optional[str] = None, instance: Optional[VastInstance] = None, ssh_private_key: Optional[str] = None, ssh_password: Optional[str] = None):
        """
        Initialize remote executor.

        Args:
            api_key: Vast.ai API key. If None, loads from .env. Optional.
            instance: VastInstance object with connection details. Required.
            ssh_private_key: Path to SSH private key. Optional if using password.
            ssh_password: SSH password. Optional if using private key.

        Raises:
            ValueError: If instance is missing or invalid.
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
        self.vast_client = VastAiClient(api_key)

    def setup_environment(self, setup_script_path: Optional[str] = None) -> None:
        """
        Setup Python environment on remote instance.

        Args:
            setup_script_path: Path to setup script. If None, looks in package. Optional.

        Raises:
            RuntimeError: If setup fails.
        """
        logger.info("Setting up remote environment...")

        try:
            self.ssh_client.connect()

            # Create workspace directory
            self.ssh_client.execute("mkdir -p /workspace")

            # Upload and run setup script
            logger.info("Installing system dependencies...")
            if setup_script_path is None:
                setup_script_path = str(Path(__file__).parent.parent / "deploy" / "setup.sh")

            setup_script = Path(setup_script_path)
            if not setup_script.exists():
                raise FileNotFoundError(f"Setup script not found: {setup_script_path}")

            self.ssh_client.upload_file(str(setup_script), "/tmp/setup.sh")
            exit_code, stdout, stderr = self.ssh_client.execute("bash /tmp/setup.sh")

            if exit_code != 0:
                raise RuntimeError(f"Setup script failed with exit code {exit_code}: {stderr}")

            logger.info("Remote environment setup complete")
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            raise
        finally:
            self.ssh_client.disconnect()

    def upload_codebase(self, local_repo_path: str, remote_path: str = "/workspace/imggenhub") -> None:
        """
        Upload project codebase to remote instance.

        Args:
            local_repo_path: Path to local repository root. Required.
            remote_path: Destination path on remote instance. Default: /workspace/imggenhub.

        Raises:
            FileNotFoundError: If local repository not found.
            RuntimeError: If upload fails.
        """
        if not local_repo_path or not isinstance(local_repo_path, str):
            raise ValueError("local_repo_path must be a non-empty string")

        local_path = Path(local_repo_path)
        if not local_path.is_dir():
            raise FileNotFoundError(f"Repository path not found: {local_repo_path}")

        logger.info(f"Uploading codebase to {self.instance.ssh_host}...")

        try:
            self.ssh_client.connect()
            self.ssh_client.execute(f"mkdir -p {remote_path}")
            self.ssh_client.upload_directory(str(local_path), remote_path)
            logger.info(f"Codebase uploaded to {remote_path}")
        except Exception as e:
            logger.error(f"Codebase upload failed: {e}")
            raise
        finally:
            self.ssh_client.disconnect()

    def upload_config(self, config_path: str, remote_path: str = "/workspace/config") -> None:
        """
        Upload prompts and configuration files to remote instance.

        Args:
            config_path: Path to local config directory. Required.
            remote_path: Destination path on remote instance. Default: /workspace/config.

        Raises:
            FileNotFoundError: If config directory not found.
            RuntimeError: If upload fails.
        """
        if not config_path or not isinstance(config_path, str):
            raise ValueError("config_path must be a non-empty string")

        config_dir = Path(config_path)
        if not config_dir.is_dir():
            raise FileNotFoundError(f"Config directory not found: {config_path}")

        logger.info(f"Uploading config to {self.instance.ssh_host}...")

        try:
            self.ssh_client.connect()
            self.ssh_client.execute(f"mkdir -p {remote_path}")
            self.ssh_client.upload_directory(str(config_dir), remote_path)
            logger.info(f"Config uploaded to {remote_path}")
        except Exception as e:
            logger.error(f"Config upload failed: {e}")
            raise
        finally:
            self.ssh_client.disconnect()

    def install_dependencies(self, remote_repo_path: str = "/workspace/imggenhub") -> None:
        """
        Install Python dependencies on remote instance using Poetry.

        Args:
            remote_repo_path: Path to repository on remote instance. Default: /workspace/imggenhub.

        Raises:
            RuntimeError: If installation fails.
        """
        logger.info("Installing Python dependencies...")

        try:
            self.ssh_client.connect()

            # Navigate to repo and install dependencies
            command = f"cd {remote_repo_path} && poetry install"
            exit_code, stdout, stderr = self.ssh_client.execute(command)

            if exit_code != 0:
                raise RuntimeError(f"Poetry install failed with exit code {exit_code}: {stderr}")

            logger.info("Dependencies installed successfully")
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            raise
        finally:
            self.ssh_client.disconnect()

    def run_pipeline(
        self,
        model_name: str,
        guidance: float,
        steps: int,
        precision: str,
        prompt: Optional[str] = None,
        prompts_file: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        refiner_model_name: Optional[str] = None,
        refiner_guidance: Optional[float] = None,
        refiner_steps: Optional[int] = None,
        refiner_precision: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> Tuple[int, str, str]:
        """
        Execute image generation pipeline on remote instance.

        Args:
            model_name: Base model to use. Required.
            guidance: Guidance scale (7-12 recommended). Required.
            steps: Number of inference steps (50-100 for quality). Required.
            precision: Precision level (fp32, fp16, int8, int4). Required.
            prompt: Single prompt string. Optional.
            prompts_file: Path to prompts JSON file on remote. Optional.
            negative_prompt: Negative prompt. Optional.
            refiner_model_name: Refiner model name. Optional.
            refiner_guidance: Refiner guidance scale. Optional if using refiner.
            refiner_steps: Refiner steps. Optional if using refiner.
            refiner_precision: Refiner precision. Optional if using refiner.
            hf_token: Hugging Face API token. Optional.

        Returns:
            Tuple of (exit_code, stdout, stderr).

        Raises:
            ValueError: If required parameters are missing.
            RuntimeError: If pipeline execution fails.
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name is required and must be a non-empty string")
        if not isinstance(guidance, (int, float)) or guidance <= 0:
            raise ValueError("guidance must be a positive number")
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        if precision not in ["fp32", "fp16", "int8", "int4"]:
            raise ValueError(f"precision must be one of: fp32, fp16, int8, int4. Got: {precision}")

        logger.info(f"Starting pipeline on remote instance: {self.instance.id}")

        cmd_parts = [
            "cd /workspace/imggenhub &&",
            "poetry run imggenhub",
            f"--model_name {model_name}",
            f"--guidance {guidance}",
            f"--steps {steps}",
            f"--precision {precision}",
        ]

        if prompt:
            cmd_parts.append(f'--prompt "{prompt}"')

        if prompts_file:
            cmd_parts.append(f"--prompts_file {prompts_file}")

        if negative_prompt:
            cmd_parts.append(f'--negative_prompt "{negative_prompt}"')

        if refiner_model_name:
            cmd_parts.append(f"--refiner_model_name {refiner_model_name}")
            if not refiner_guidance or not refiner_steps or not refiner_precision:
                raise ValueError("refiner_guidance, refiner_steps, and refiner_precision are required when using refiner_model_name")
            cmd_parts.append(f"--refiner_guidance {refiner_guidance}")
            cmd_parts.append(f"--refiner_steps {refiner_steps}")
            cmd_parts.append(f"--refiner_precision {refiner_precision}")

        if hf_token:
            cmd_parts.append(f"--hf_token {hf_token}")

        command = " ".join(cmd_parts)

        try:
            self.ssh_client.connect()
            logger.info(f"Executing: {command}")
            exit_code, stdout, stderr = self.ssh_client.execute(command)

            if exit_code != 0:
                logger.error(f"Pipeline failed with exit code {exit_code}")
                logger.error(f"Stderr: {stderr}")

            return exit_code, stdout, stderr
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
        finally:
            self.ssh_client.disconnect()

    def download_results(self, remote_output_dir: str, local_output_dir: str) -> None:
        """
        Download generated images and results from remote instance.

        Args:
            remote_output_dir: Path to output directory on remote instance. Required.
            local_output_dir: Destination directory on local machine. Required.

        Raises:
            ValueError: If paths are invalid.
            RuntimeError: If download fails.
        """
        if not remote_output_dir or not isinstance(remote_output_dir, str):
            raise ValueError("remote_output_dir must be a non-empty string")
        if not local_output_dir or not isinstance(local_output_dir, str):
            raise ValueError("local_output_dir must be a non-empty string")

        logger.info(f"Downloading results from {self.instance.ssh_host}...")

        try:
            self.ssh_client.connect()
            self.ssh_client.download_directory(remote_output_dir, local_output_dir)
            logger.info(f"Results downloaded to {local_output_dir}")
        except Exception as e:
            logger.error(f"Result download failed: {e}")
            raise
        finally:
            self.ssh_client.disconnect()

    def cleanup(self) -> None:
        """
        Clean up remote instance (optional - can be done manually via API).

        Raises:
            RuntimeError: If cleanup fails.
        """
        logger.info(f"Terminating instance {self.instance.id}...")

        try:
            success = self.vast_client.destroy_instance(self.instance.id)
            if success:
                logger.info(f"Instance {self.instance.id} terminated successfully")
            else:
                logger.warning(f"Instance {self.instance.id} termination may have failed")
        except Exception as e:
            logger.error(f"Instance termination failed: {e}")
            raise
