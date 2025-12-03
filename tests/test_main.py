from unittest.mock import patch, MagicMock
import logging
from imggenhub.kaggle import main

def test_run_pipeline_success():
    with patch('imggenhub.kaggle.main.download_selective.run', return_value=True) as mock_download, \
         patch('imggenhub.kaggle.main.poll_status.run', return_value='kernelworkerstatus.complete') as mock_poll, \
         patch('imggenhub.kaggle.main.deploy.run') as mock_deploy, \
         patch('imggenhub.kaggle.main.resolve_prompts', return_value=['prompt']) as mock_resolve, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('logging.error') as mock_logging_error:  # Suppress error logging in tests
        
        from pathlib import Path
        dest_path = Path("output/test_run")
        main.run_pipeline(
            dest_path=dest_path,
            prompts_file='./config/prompts.json',
            notebook='./config/kaggle-notebook-image-generation.ipynb',
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
    with patch('imggenhub.kaggle.main.download_selective.run', return_value=True) as mock_download, \
         patch('imggenhub.kaggle.main.poll_status.run', return_value='kernelworkerstatus.error') as mock_poll, \
         patch('imggenhub.kaggle.main.deploy.run') as mock_deploy, \
         patch('imggenhub.kaggle.main.resolve_prompts', return_value=['prompt']) as mock_resolve, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('logging.error') as mock_logging_error:  # Suppress error logging in tests
        
        from pathlib import Path
        dest_path = Path("output/test_run")
        try:
            main.run_pipeline(
                dest_path=dest_path,
                prompts_file='./config/prompts.json',
                notebook='./config/kaggle-notebook-image-generation.ipynb',
                kernel_path='./config',
                gpu=True,
                guidance=7.5,
                steps=50,
                precision="fp16"
            )
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass  # Expected
