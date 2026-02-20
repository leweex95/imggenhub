import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil
from imggenhub.kaggle.core.parallel_deploy import (
    split_prompts, 
    should_use_parallel, 
    _create_deployment2_kernel_dir, 
    _deploy_single_kernel, 
    _poll_kernel,
    _download_kernel_output,
    DEPLOYMENT2_KERNEL_ID
)

class TestSplitPrompts:
    def test_split_even_number(self):
        prompts = ["a", "b", "c", "d"]
        first, second = split_prompts(prompts)
        assert first == ["a", "b"]
        assert second == ["c", "d"]

    def test_split_odd_number(self):
        prompts = ["a", "b", "c", "d", "e"]
        first, second = split_prompts(prompts)
        assert first == ["a", "b"]
        assert second == ["c", "d", "e"]

class TestShouldUseParallel:
    def test_should_use_parallel_true(self):
        assert should_use_parallel(["a"] * 5) is True

    def test_should_use_parallel_false(self):
        assert should_use_parallel(["a"] * 4) is False

class TestCreateDeployment2Dir:
    @patch("shutil.copy2")
    def test_create_deployment2_kernel_dir(self, mock_copy2):
        with tempfile.TemporaryDirectory() as tmp_dir:
            kernel_path = Path(tmp_dir)
            notebook = Path("test.ipynb")
            metadata_file = kernel_path / "kernel-metadata.json"
            metadata_file.write_text('{"id": "test", "title": "test", "code_file": "test.ipynb"}')
            
            with patch("imggenhub.kaggle.core.parallel_deploy.tempfile.mkdtemp") as mock_mkdtemp:
                mock_mkdtemp.return_value = tmp_dir
                res = _create_deployment2_kernel_dir(kernel_path, notebook)
                assert Path(res) == Path(tmp_dir)

class TestDeploySingleKernel:
    @patch('imggenhub.kaggle.core.parallel_deploy.JobManager')
    def test_deploy_single_kernel_success(self, mock_jm_class):
        mock_jm = mock_jm_class.return_value
        prompts = ["p1"]
        notebook = Path("nb.ipynb")
        kernel_path = Path("/path")
        
        res = _deploy_single_kernel(prompts, notebook, kernel_path, "id", {"precision": "fp16"})
        assert res == "id"
        mock_jm.edit_notebook_params.assert_called_once()
        mock_jm.deploy.assert_called_once()

class TestPollKernel:
    @patch('imggenhub.kaggle.core.parallel_deploy.JobManager')
    def test_poll_kernel_success(self, mock_jm_class):
        mock_jm = mock_jm_class.return_value
        mock_jm.poll_until_complete.return_value = "complete"
        
        res = _poll_kernel("id", poll_interval=10)
        assert res == "complete"
        mock_jm_class.assert_called_with("id")
        mock_jm.poll_until_complete.assert_called_with(pool_interval=10)

class TestDownloadKernel:
    @patch('imggenhub.kaggle.core.parallel_deploy.SelectiveDownloader')
    def test_download_kernel_output(self, mock_sd_class):
        mock_sd = mock_sd_class.return_value
        mock_sd.download_images.return_value = True
        
        dest = Path("/dest")
        _download_kernel_output("id", dest, expected_count=5)
        mock_sd_class.assert_called_with("id", dest=str(dest))
        mock_sd.download_images.assert_called_with(expected_image_count=5, stable_count_patience=4)
