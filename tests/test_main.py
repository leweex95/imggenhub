from unittest.mock import patch, MagicMock
import logging
import sys
from pathlib import Path
from imggenhub.kaggle import main

def test_run_pipeline_success():
    fake_image = MagicMock()
    fake_image.is_file.return_value = True
    fake_image.suffix = ".png"

    with patch('imggenhub.kaggle.main.download.run', return_value=True) as mock_download, \
         patch('imggenhub.kaggle.main.poll_status.run', return_value='kernelworkerstatus.complete') as mock_poll, \
         patch('imggenhub.kaggle.main.deploy_kaggle_notebook') as mock_deploy, \
         patch('imggenhub.kaggle.main.sync_hf_token', return_value=True), \
         patch('imggenhub.kaggle.main.resolve_prompts', return_value=['prompt']) as mock_resolve, \
         patch('imggenhub.kaggle.main.save_prompt_mapping') as mock_save_mapping, \
         patch('pathlib.Path.rglob', return_value=[fake_image]) as mock_rglob, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('logging.error') as mock_logging_error:  # Suppress error logging in tests
        
        from pathlib import Path
        dest_path = Path("output/test_run")
        main.run_pipeline(
            dest_path=dest_path,
            prompts_file='./config/prompts.json',
            notebook='./notebooks/kaggle-modern-diffusion.ipynb',
            kernel_path='./config',
            gpu=True,
            guidance=7.5,
            steps=50,
            precision="fp16"
        )
        assert mock_deploy.called
        assert mock_poll.called
        assert mock_download.called

def test_run_pipeline_kernel_error():
    with patch('imggenhub.kaggle.main.download.run', return_value=True) as mock_download, \
         patch('imggenhub.kaggle.main.poll_status.run', return_value='kernelworkerstatus.error') as mock_poll, \
         patch('imggenhub.kaggle.main.deploy_kaggle_notebook') as mock_deploy, \
         patch('imggenhub.kaggle.main.sync_hf_token', return_value=True), \
         patch('imggenhub.kaggle.main.resolve_prompts', return_value=['prompt']) as mock_resolve, \
         patch('imggenhub.kaggle.main.save_prompt_mapping') as mock_save_mapping, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('logging.error') as mock_logging_error:  # Suppress error logging in tests
        
        from pathlib import Path
        dest_path = Path("output/test_run")
        try:
            main.run_pipeline(
                dest_path=dest_path,
                prompts_file='./config/prompts.json',
                notebook='./notebooks/kaggle-modern-diffusion.ipynb',
                kernel_path='./config',
                gpu=True,
                guidance=7.5,
                steps=50,
                precision="fp16"
            )
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass  # Expected


def _run_main_with_args(args):
    with patch.object(sys, "argv", args), \
         patch("imggenhub.kaggle.main.setup_output_directory", return_value=Path("outputs/test_run")), \
         patch("imggenhub.kaggle.main.log_cli_command"), \
         patch("imggenhub.kaggle.main.validate_args"), \
         patch("imggenhub.kaggle.main.run_pipeline") as mock_run_pipeline:
        main.main()
        return mock_run_pipeline


def test_main_autodetects_modern_diffusion_notebook_for_sd35():
    mock_run_pipeline = _run_main_with_args(
        [
            "imggenhub",
            "--model-id", "stabilityai/stable-diffusion-3.5-medium",
            "--guidance", "2.0",
            "--steps", "5",
            "--precision", "fp16",
            "--img-width", "512",
            "--img-height", "512",
            "--prompt", "test prompt",
        ]
    )
    kwargs = mock_run_pipeline.call_args.kwargs
    assert kwargs["notebook"] == "./notebooks/kaggle-modern-diffusion.ipynb"
    assert kwargs["gpu"] is True


def test_main_autodetects_modern_notebook_for_illustrious_pony():
    mock_run_pipeline = _run_main_with_args(
        [
            "imggenhub",
            "--model-id", "fancy/pony-diffusion-xl-v6",
            "--guidance", "7.0",
            "--steps", "10",
            "--precision", "fp16",
            "--img-width", "512",
            "--img-height", "512",
            "--prompt", "test prompt",
        ]
    )
    kwargs = mock_run_pipeline.call_args.kwargs
    assert kwargs["notebook"] == "./notebooks/kaggle-modern-diffusion.ipynb"


def test_main_calls_parallel_run_for_many_prompts():
    # We use the mock returned by _run_main_with_args
    mock_run_pipeline = _run_main_with_args(
        [
            "imggenhub",
            "--model-id", "stabilityai/stable-diffusion-xl-base-1.0",
            "--guidance", "7.0",
            "--steps", "50",
            "--precision", "fp16",
            "--img-width", "1024",
            "--img-height", "1024",
            "--prompt", "prompt 1",
            "--prompt", "prompt 2",
            "--prompt", "prompt 3",
            "--prompt", "prompt 4",
            "--prompt", "prompt 5",
        ]
    )
    assert mock_run_pipeline.called is True
    # The prompts should be passed to run_pipeline, which then delegates to parallel
    # We can test the delegation in a separate test or assume it works because of should_use_parallel
    assert len(mock_run_pipeline.call_args.kwargs.get("prompt", [])) == 5


def test_run_pipeline_delegates_to_parallel():
    with patch("imggenhub.kaggle.main.should_use_parallel", return_value=True), \
         patch("imggenhub.kaggle.main.run_parallel_pipeline") as mock_parallel, \
         patch("imggenhub.kaggle.main.sync_hf_token", return_value=True), \
         patch("imggenhub.kaggle.main.resolve_prompts", return_value=["p1", "p2", "p3", "p4", "p5"]), \
         patch("imggenhub.kaggle.main.save_prompt_mapping") as mock_save_mapping, \
         patch("imggenhub.kaggle.main.load_kaggle_config", return_value={}):
        
        main.run_pipeline(
            dest_path=Path("outputs/test"),
            prompts_file=None,
            notebook="notebook.ipynb",
            kernel_path="config",
            prompt=["p1", "p2", "p3", "p4", "p5"],
            guidance=7.0,
            steps=50,
            precision="fp16"
        )
        assert mock_parallel.called is True
        assert mock_parallel.called is True
