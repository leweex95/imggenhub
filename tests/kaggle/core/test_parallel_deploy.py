import pytest
from unittest.mock import patch, mock_open
from pathlib import Path
import json
import tempfile
import shutil
from imggenhub.kaggle.core.parallel_deploy import split_prompts, should_use_parallel, _create_worker_kernel_dir, _deploy_single_kernel, _poll_kernel


class TestSplitPrompts:
    """Test cases for split_prompts function."""

    def test_split_even_number(self):
        """Test splitting even number of prompts."""
        prompts = ["a", "b", "c", "d"]
        first, second = split_prompts(prompts)
        assert first == ["a", "b"]
        assert second == ["c", "d"]

    def test_split_odd_number(self):
        """Test splitting odd number of prompts."""
        prompts = ["a", "b", "c", "d", "e"]
        first, second = split_prompts(prompts)
        assert first == ["a", "b"]  # smaller half
        assert second == ["c", "d", "e"]

    def test_split_single_prompt(self):
        """Test splitting single prompt."""
        prompts = ["a"]
        first, second = split_prompts(prompts)
        assert first == []
        assert second == ["a"]

    def test_split_empty_list(self):
        """Test splitting empty list."""
        prompts = []
        first, second = split_prompts(prompts)
        assert first == []
        assert second == []


class TestShouldUseParallel:
    """Test cases for should_use_parallel function."""

    def test_should_use_parallel_above_threshold(self):
        """Test that parallel is used when prompts > 4."""
        prompts = ["a", "b", "c", "d", "e"]
        assert should_use_parallel(prompts) is True

    def test_should_use_parallel_at_threshold(self):
        """Test that parallel is not used when prompts == 4."""
        prompts = ["a", "b", "c", "d"]
        assert should_use_parallel(prompts) is False

    def test_should_use_parallel_below_threshold(self):
        """Test that parallel is not used when prompts < 4."""
        prompts = ["a", "b", "c"]
        assert should_use_parallel(prompts) is False

    def test_should_use_parallel_empty_list(self):
        """Test that parallel is not used for empty list."""
        prompts = []
        assert should_use_parallel(prompts) is False


class TestCreateWorkerKernelDir:
    """Test cases for _create_worker_kernel_dir function."""

    @patch('tempfile.mkdtemp')
    @patch('shutil.copy2')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('json.dump')
    def test_create_worker_kernel_dir(self, mock_json_dump, mock_json_load, mock_file_open, mock_exists, mock_copy, mock_mkdtemp):
        """Test creating worker kernel directory."""
        # Setup mocks
        mock_mkdtemp.return_value = "/tmp/worker_dir"
        mock_exists.return_value = True
        mock_json_load.return_value = {
            "id": "primary-kernel",
            "title": "Primary Kernel",
            "code_file": "notebook.ipynb"
        }

        # Test
        kernel_path = Path("/path/to/kernel")
        notebook = Path("notebook.ipynb")
        result = _create_worker_kernel_dir(kernel_path, notebook)

        # Assertions
        assert result == Path("/tmp/worker_dir")
        mock_mkdtemp.assert_called_once_with(prefix="kaggle_worker_")
        mock_copy.assert_called_once()
        mock_json_load.assert_called_once()
        mock_json_dump.assert_called_once()

        # Check that metadata was modified for worker
        call_args = mock_json_dump.call_args[0]
        modified_metadata = call_args[0]
        assert modified_metadata["id"] == "leventecsibi/stable-diffusion-batch-generator-worker"
        assert modified_metadata["title"] == "Stable Diffusion Batch Generator Worker"
        assert modified_metadata["code_file"] == "notebook.ipynb"


class TestDeploySingleKernel:
    """Test cases for _deploy_single_kernel function."""

    @patch('imggenhub.kaggle.core.parallel_deploy.deploy.run')
    @patch('time.sleep')
    def test_deploy_single_kernel_success(self, mock_sleep, mock_deploy_run):
        """Test successful deployment on first attempt."""
        prompts_list = ["prompt1", "prompt2"]
        notebook = Path("notebook.ipynb")
        kernel_path = Path("/path/to/kernel")
        kernel_id = "test-kernel"
        deploy_kwargs = {"gpu": True}

        result = _deploy_single_kernel(
            prompts_list=prompts_list,
            notebook=notebook,
            kernel_path=kernel_path,
            kernel_id=kernel_id,
            deploy_kwargs=deploy_kwargs
        )

        assert result == kernel_id
        mock_deploy_run.assert_called_once_with(
            prompts_list=prompts_list,
            notebook=notebook,
            kernel_path=kernel_path,
            **deploy_kwargs
        )
        mock_sleep.assert_not_called()

    @patch('imggenhub.kaggle.core.parallel_deploy.deploy.run')
    @patch('time.sleep')
    def test_deploy_single_kernel_retry_success(self, mock_sleep, mock_deploy_run):
        """Test deployment succeeds after retry on conflict error."""
        mock_deploy_run.side_effect = [Exception("409 Conflict"), None]

        prompts_list = ["prompt1"]
        notebook = Path("notebook.ipynb")
        kernel_path = Path("/path/to/kernel")
        kernel_id = "test-kernel"
        deploy_kwargs = {}

        result = _deploy_single_kernel(
            prompts_list=prompts_list,
            notebook=notebook,
            kernel_path=kernel_path,
            kernel_id=kernel_id,
            deploy_kwargs=deploy_kwargs
        )

        assert result == kernel_id
        assert mock_deploy_run.call_count == 2
        mock_sleep.assert_called_once_with(30)

    @patch('imggenhub.kaggle.core.parallel_deploy.deploy.run')
    @patch('time.sleep')
    def test_deploy_single_kernel_non_retryable_error(self, mock_sleep, mock_deploy_run):
        """Test deployment fails immediately on non-retryable error."""
        mock_deploy_run.side_effect = Exception("Invalid kernel")

        with pytest.raises(Exception, match="Invalid kernel"):
            _deploy_single_kernel(
                prompts_list=["prompt1"],
                notebook=Path("notebook.ipynb"),
                kernel_path=Path("/path/to/kernel"),
                kernel_id="test-kernel",
                deploy_kwargs={}
            )

        mock_deploy_run.assert_called_once()
        mock_sleep.assert_not_called()

    @patch('imggenhub.kaggle.core.parallel_deploy.deploy.run')
    @patch('time.sleep')
    def test_deploy_single_kernel_max_retries_exhausted(self, mock_sleep, mock_deploy_run):
        """Test deployment fails after max retries."""
        mock_deploy_run.side_effect = Exception("409 Conflict")

        with pytest.raises(Exception, match="409 Conflict"):
            _deploy_single_kernel(
                prompts_list=["prompt1"],
                notebook=Path("notebook.ipynb"),
                kernel_path=Path("/path/to/kernel"),
                kernel_id="test-kernel",
                deploy_kwargs={},
                max_retries=2
            )

        assert mock_deploy_run.call_count == 2
        assert mock_sleep.call_count == 2


class TestPollKernel:
    """Test cases for _poll_kernel function."""

    @patch('imggenhub.kaggle.core.parallel_deploy.poll_status.run')
    def test_poll_kernel_success(self, mock_poll_run):
        """Test polling succeeds."""
        mock_poll_run.return_value = "kernelworkerstatus.complete"

        result = _poll_kernel("test-kernel", max_wait=1800)

        assert result == "kernelworkerstatus.complete"
        mock_poll_run.assert_called_once_with(kernel_id="test-kernel", poll_interval=15)

    @patch('imggenhub.kaggle.core.parallel_deploy.poll_status.run')
    @patch('time.sleep')
    def test_poll_kernel_error_recovery(self, mock_sleep, mock_poll_run):
        """Test polling recovers from poll errors."""
        mock_poll_run.side_effect = [Exception("Poll error"), "kernelworkerstatus.complete"]

        result = _poll_kernel("test-kernel", max_wait=1800)

        assert result == "kernelworkerstatus.complete"
        assert mock_poll_run.call_count == 2
        mock_sleep.assert_called_once_with(15)

    @patch('imggenhub.kaggle.core.parallel_deploy.poll_status.run')
    @patch('time.sleep')
    def test_poll_kernel_error_recovery(self, mock_sleep, mock_poll_run):
        """Test polling recovers from poll errors."""
        mock_poll_run.side_effect = [Exception("Poll error"), "kernelworkerstatus.complete"]

        result = _poll_kernel("test-kernel", max_wait=1800)

        assert result == "kernelworkerstatus.complete"
        assert mock_poll_run.call_count == 2
        mock_sleep.assert_called_once_with(15)