"""SSH client for executing commands on remote Vast.ai instances."""
import paramiko
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SSHClient:
    """Secure SSH client for remote command execution on Vast.ai instances."""

    def __init__(self, host: str, port: int = 22, username: str = "root", private_key_path: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize SSH client connection parameters.

        Args:
            host: SSH host/IP address. Required.
            port: SSH port number. Default: 22.
            username: SSH username. Default: root.
            private_key_path: Path to private SSH key file. Optional if using password.
            password: SSH password. Optional if using private key.

        Raises:
            ValueError: If host is empty, port is invalid, or both auth methods are missing.
        """
        if not host or not isinstance(host, str):
            raise ValueError("host must be a non-empty string")
        if not isinstance(port, int) or port <= 0 or port > 65535:
            raise ValueError("port must be an integer between 1 and 65535")
        if not username or not isinstance(username, str):
            raise ValueError("username must be a non-empty string")

        if not private_key_path and not password:
            raise ValueError("Either private_key_path or password must be provided")

        self.host = host
        self.port = port
        self.username = username
        self.private_key_path = private_key_path
        self.password = password
        self.client: Optional[paramiko.SSHClient] = None

    def connect(self) -> None:
        """
        Establish SSH connection to remote host.

        Raises:
            FileNotFoundError: If private key file does not exist.
            paramiko.AuthenticationException: If authentication fails.
            paramiko.SSHException: If SSH connection fails.
        """
        if self.client is not None and self.client.get_transport() is not None and self.client.get_transport().is_active():
            logger.debug(f"Already connected to {self.host}")
            return

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        pkey = None
        if self.private_key_path:
            key_path = Path(self.private_key_path)
            if not key_path.exists():
                raise FileNotFoundError(f"Private key file not found: {self.private_key_path}")
            try:
                pkey = paramiko.RSAKey.from_private_key_file(self.private_key_path)
            except (paramiko.PasswordRequiredException, paramiko.SSHException):
                pkey = None

        try:
            self.client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                pkey=pkey,
                password=self.password,
                timeout=10,
                allow_agent=True,
                look_for_keys=True,
            )
            logger.info(f"SSH connection established to {self.username}@{self.host}:{self.port}")
        except Exception as e:
            self.client = None
            raise

    def disconnect(self) -> None:
        """Close SSH connection."""
        if self.client:
            self.client.close()
            self.client = None
            logger.info(f"SSH connection closed to {self.host}")

    def execute(self, command: str) -> Tuple[int, str, str]:
        """
        Execute a command on the remote host.

        Args:
            command: Shell command to execute. Required.

        Returns:
            Tuple of (exit_code, stdout, stderr).

        Raises:
            ValueError: If command is empty.
            RuntimeError: If not connected or command execution fails.
        """
        if not command or not isinstance(command, str):
            raise ValueError("command must be a non-empty string")

        if not self.client or not self.client.get_transport() or not self.client.get_transport().is_active():
            raise RuntimeError("Not connected to remote host. Call connect() first.")

        try:
            stdin, stdout, stderr = self.client.exec_command(command, timeout=300)
            exit_code = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode("utf-8")
            stderr_text = stderr.read().decode("utf-8")

            logger.debug(f"Command executed: {command} (exit code: {exit_code})")
            return exit_code, stdout_text, stderr_text
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise RuntimeError(f"Command execution failed: {e}") from e

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """
        Upload a local file to the remote host.

        Args:
            local_path: Path to local file. Required.
            remote_path: Destination path on remote host. Required.

        Raises:
            FileNotFoundError: If local file does not exist.
            RuntimeError: If not connected or upload fails.
        """
        if not local_path or not isinstance(local_path, str):
            raise ValueError("local_path must be a non-empty string")
        if not remote_path or not isinstance(remote_path, str):
            raise ValueError("remote_path must be a non-empty string")

        local_file = Path(local_path)
        if not local_file.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        if not self.client or not self.client.get_transport() or not self.client.get_transport().is_active():
            raise RuntimeError("Not connected to remote host. Call connect() first.")

        try:
            sftp = self.client.open_sftp()
            sftp.put(str(local_file), remote_path)
            sftp.close()
            logger.info(f"File uploaded: {local_path} -> {self.host}:{remote_path}")
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise RuntimeError(f"File upload failed: {e}") from e

    def download_file(self, remote_path: str, local_path: str) -> None:
        """
        Download a file from remote host to local.

        Args:
            remote_path: Path to file on remote host. Required.
            local_path: Destination path on local machine. Required.

        Raises:
            ValueError: If paths are invalid.
            RuntimeError: If not connected or download fails.
        """
        if not remote_path or not isinstance(remote_path, str):
            raise ValueError("remote_path must be a non-empty string")
        if not local_path or not isinstance(local_path, str):
            raise ValueError("local_path must be a non-empty string")

        if not self.client or not self.client.get_transport() or not self.client.get_transport().is_active():
            raise RuntimeError("Not connected to remote host. Call connect() first.")

        try:
            sftp = self.client.open_sftp()
            sftp.get(remote_path, local_path)
            sftp.close()
            logger.info(f"File downloaded: {self.host}:{remote_path} -> {local_path}")
        except Exception as e:
            logger.error(f"File download failed: {e}")
            raise RuntimeError(f"File download failed: {e}") from e

    def upload_directory(self, local_dir: str, remote_dir: str) -> None:
        """
        Recursively upload a local directory to remote host.

        Args:
            local_dir: Path to local directory. Required.
            remote_dir: Destination directory on remote host. Required.

        Raises:
            FileNotFoundError: If local directory does not exist.
            RuntimeError: If not connected or upload fails.
        """
        if not local_dir or not isinstance(local_dir, str):
            raise ValueError("local_dir must be a non-empty string")
        if not remote_dir or not isinstance(remote_dir, str):
            raise ValueError("remote_dir must be a non-empty string")

        local_directory = Path(local_dir)
        if not local_directory.is_dir():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")

        if not self.client or not self.client.get_transport() or not self.client.get_transport().is_active():
            raise RuntimeError("Not connected to remote host. Call connect() first.")

        try:
            sftp = self.client.open_sftp()

            def ensure_dir(sftp_client, path):
                try:
                    sftp_client.stat(path)
                except IOError:
                    ensure_dir(sftp_client, "/".join(path.split("/")[:-1]))
                    sftp_client.mkdir(path)

            ensure_dir(sftp, remote_dir)

            for local_file in local_directory.rglob("*"):
                if local_file.is_file():
                    relative_path = local_file.relative_to(local_directory)
                    remote_file_path = f"{remote_dir}/{relative_path.as_posix()}"
                    remote_file_dir = "/".join(remote_file_path.split("/")[:-1])
                    ensure_dir(sftp, remote_file_dir)
                    sftp.put(str(local_file), remote_file_path)

            sftp.close()
            logger.info(f"Directory uploaded: {local_dir} -> {self.host}:{remote_dir}")
        except Exception as e:
            logger.error(f"Directory upload failed: {e}")
            raise RuntimeError(f"Directory upload failed: {e}") from e

    def download_directory(self, remote_dir: str, local_dir: str) -> None:
        """
        Recursively download a directory from remote host.

        Args:
            remote_dir: Path to directory on remote host. Required.
            local_dir: Destination directory on local machine. Required.

        Raises:
            ValueError: If paths are invalid.
            RuntimeError: If not connected or download fails.
        """
        if not remote_dir or not isinstance(remote_dir, str):
            raise ValueError("remote_dir must be a non-empty string")
        if not local_dir or not isinstance(local_dir, str):
            raise ValueError("local_dir must be a non-empty string")

        if not self.client or not self.client.get_transport() or not self.client.get_transport().is_active():
            raise RuntimeError("Not connected to remote host. Call connect() first.")

        try:
            sftp = self.client.open_sftp()
            local_directory = Path(local_dir)
            local_directory.mkdir(parents=True, exist_ok=True)

            def download_recursive(remote_path, local_path):
                local_path_obj = Path(local_path)
                local_path_obj.mkdir(parents=True, exist_ok=True)

                for item in sftp.listdir_attr(remote_path):
                    remote_item_path = f"{remote_path}/{item.filename}"
                    local_item_path = local_path_obj / item.filename

                    if item.filename.startswith("."):
                        continue

                    if stat.S_ISDIR(item.st_mode):
                        download_recursive(remote_item_path, str(local_item_path))
                    else:
                        sftp.get(remote_item_path, str(local_item_path))

            import stat
            download_recursive(remote_dir, local_dir)
            sftp.close()
            logger.info(f"Directory downloaded: {self.host}:{remote_dir} -> {local_dir}")
        except Exception as e:
            logger.error(f"Directory download failed: {e}")
            raise RuntimeError(f"Directory download failed: {e}") from e
