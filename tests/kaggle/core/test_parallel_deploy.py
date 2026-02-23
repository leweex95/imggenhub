import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil
from imggenhub.kaggle.core.parallel_deploy import (
    split_prompts,
    should_use_parallel,
    _deploy_single_kernel,
    _poll_kernel,
    _download_kernel_output,
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


class TestDeploySingleKernel:
    @patch("imggenhub.kaggle.core.parallel_deploy._notebook_to_script", return_value="# mock script\n")
    @patch("imggenhub.kaggle.core.parallel_deploy.shutil.copy2")
    @patch("imggenhub.kaggle.core.parallel_deploy.JobManager")
    def test_deploy_single_kernel_success(self, mock_jm_class, mock_copy2, mock_nb2script):
        mock_jm = mock_jm_class.return_value
        prompts = ["p1"]
        notebook = Path("nb.ipynb")
        kernel_path = Path("/path")

        res = _deploy_single_kernel(prompts, notebook, kernel_path, "id", {"precision": "fp16"})
        assert res == "id"
        mock_jm.edit_notebook_params.assert_called_once()
        mock_jm.deploy.assert_called_once()

    @patch("imggenhub.kaggle.core.parallel_deploy._notebook_to_script", return_value="# mock script\n")
    @patch("imggenhub.kaggle.core.parallel_deploy.shutil.copy2")
    @patch("imggenhub.kaggle.core.parallel_deploy.JobManager")
    def test_deploy_single_kernel_injects_lora_for_bf16_notebook(self, mock_jm_class, mock_copy2, mock_nb2script):
        mock_jm = mock_jm_class.return_value
        prompts = ["p1"]
        notebook = Path("kaggle-flux-schnell-bf16.ipynb")
        kernel_path = Path("/path")
        deploy_kwargs = {
            "precision": "bf16",
            "lora_repo_id": "Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur",
            "lora_filename": "FLUX-dev-lora-AntiBlur.safetensors",
            "lora_scale": 0.9,
        }

        _deploy_single_kernel(prompts, notebook, kernel_path, "id", deploy_kwargs)

        call_args = mock_jm.edit_notebook_params.call_args
        injected_params = call_args[0][1]
        assert injected_params["LORA_REPO_ID"] == "Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur"
        assert injected_params["LORA_FILENAME"] == "FLUX-dev-lora-AntiBlur.safetensors"
        assert injected_params["LORA_SCALE"] == 0.9

    @patch("imggenhub.kaggle.core.parallel_deploy._notebook_to_script", return_value="# mock script\n")
    @patch("imggenhub.kaggle.core.parallel_deploy.shutil.copy2")
    @patch("imggenhub.kaggle.core.parallel_deploy.JobManager")
    def test_deploy_single_kernel_injects_lora_for_dual_t4_notebook(self, mock_jm_class, mock_copy2, mock_nb2script):
        mock_jm = mock_jm_class.return_value
        prompts = ["p1"]
        notebook = Path("kaggle-flux-dual-t4.ipynb")
        kernel_path = Path("/path")
        deploy_kwargs = {
            "precision": "bf16",
            "lora_repo_id": "some/lora-repo",
            "lora_scale": 0.7,
        }

        _deploy_single_kernel(prompts, notebook, kernel_path, "id", deploy_kwargs)

        call_args = mock_jm.edit_notebook_params.call_args
        injected_params = call_args[0][1]
        assert injected_params["LORA_REPO_ID"] == "some/lora-repo"
        assert injected_params["LORA_SCALE"] == 0.7

    @patch("imggenhub.kaggle.core.parallel_deploy._notebook_to_script", return_value="# mock script\n")
    @patch("imggenhub.kaggle.core.parallel_deploy.shutil.copy2")
    @patch("imggenhub.kaggle.core.parallel_deploy.JobManager")
    def test_deploy_single_kernel_no_lora_for_gguf_notebook(self, mock_jm_class, mock_copy2, mock_nb2script):
        mock_jm = mock_jm_class.return_value
        prompts = ["p1"]
        notebook = Path("kaggle-flux-gguf.ipynb")
        kernel_path = Path("/path")
        deploy_kwargs = {
            "precision": "q4",
            "lora_repo_id": "some/lora-repo",
            "lora_scale": 0.8,
        }

        _deploy_single_kernel(prompts, notebook, kernel_path, "id", deploy_kwargs)

        call_args = mock_jm.edit_notebook_params.call_args
        injected_params = call_args[0][1]
        # LoRA must NOT be injected into GGUF notebook
        assert "LORA_REPO_ID" not in injected_params
        assert "LORA_SCALE" not in injected_params


class TestPollKernel:
    @patch("imggenhub.kaggle.core.parallel_deploy.JobManager")
    def test_poll_kernel_success(self, mock_jm_class):
        mock_jm = mock_jm_class.return_value
        mock_jm.poll_until_complete.return_value = "complete"

        res = _poll_kernel("id", poll_interval=10)
        assert res == "complete"
        mock_jm_class.assert_called_with("id")
        mock_jm.poll_until_complete.assert_called_with(poll_interval=10)


class TestDownloadKernel:
    @patch("imggenhub.kaggle.core.parallel_deploy.SelectiveDownloader")
    def test_download_kernel_output(self, mock_sd_class):
        mock_sd = mock_sd_class.return_value
        mock_sd.download_images.return_value = True

        dest = Path("/dest")
        _download_kernel_output("id", dest, expected_count=5)
        mock_sd_class.assert_called_with("id", dest=str(dest))
        mock_sd.download_images.assert_called_with(expected_image_count=5, stable_count_patience=4)

