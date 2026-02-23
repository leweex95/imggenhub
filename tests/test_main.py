from unittest.mock import patch, MagicMock
import logging
import os
from pathlib import Path
from imggenhub.kaggle import main

def test_run_pipeline_success():
    with patch('imggenhub.kaggle.main.DatasetManager') as mock_dm_cls, \
         patch('imggenhub.kaggle.main.JobManager') as mock_jm_cls, \
         patch('imggenhub.kaggle.main.SelectiveDownloader') as mock_sd_cls, \
         patch('imggenhub.kaggle.main.resolve_prompts', return_value=['prompt']) as mock_resolve, \
         patch('imggenhub.kaggle.main.load_kaggle_config', return_value={}) as mock_config, \
         patch('imggenhub.kaggle.main._notebook_to_script', return_value='# mock script\n'), \
         patch('imggenhub.kaggle.main.shutil.copy2'):
        
        # Setup mocks
        mock_dm = mock_dm_cls.return_value
        mock_dm.sync_dataset.return_value = True
        
        mock_jm = mock_jm_cls.return_value
        mock_jm.poll_until_complete.return_value = 'complete'
        
        mock_sd = mock_sd_cls.return_value
        mock_sd.download_images.return_value = []
        
        os.environ["HF_TOKEN"] = "test_token"

        dest_path = Path("output/test_run")
        dest_path.mkdir(parents=True, exist_ok=True)
        # Create a gen_ image file matching the expected count from mocked resolve_prompts
        (dest_path / "gen_test_p1_test_prompt_20260101_000000.png").write_bytes(b"dummy")

        main.run_pipeline(
            dest_path=dest_path,
            prompts_file='./config/prompts.json',
            notebook='kaggle-stable-diffusion.ipynb',
            kernel_path='./config',
            gpu=True,
            guidance=7.5,
            steps=50,
            precision="fp16"
        )
        
        assert mock_dm.sync_dataset.called
        assert mock_jm.deploy.called
        assert mock_jm.poll_until_complete.called
        assert mock_sd.download_images.called

def test_run_pipeline_kernel_error():
    with patch('imggenhub.kaggle.main.DatasetManager') as mock_dm_cls, \
         patch('imggenhub.kaggle.main.JobManager') as mock_jm_cls, \
         patch('imggenhub.kaggle.main.SelectiveDownloader') as mock_sd_cls, \
         patch('imggenhub.kaggle.main.resolve_prompts', return_value=['prompt']) as mock_resolve, \
         patch('imggenhub.kaggle.main.load_kaggle_config', return_value={}) as mock_config, \
         patch('imggenhub.kaggle.main._notebook_to_script', return_value='# mock script\n'), \
         patch('imggenhub.kaggle.main.shutil.copy2'):
        
        # Setup mocks
        mock_dm = mock_dm_cls.return_value
        mock_dm.sync_dataset.return_value = True
        
        mock_jm = mock_jm_cls.return_value
        mock_jm.poll_until_complete.return_value = 'error'
        
        os.environ["HF_TOKEN"] = "test_token"
        
        dest_path = Path("output/test_run")
        try:
            main.run_pipeline(
                dest_path=dest_path,
                prompts_file='./config/prompts.json',
                notebook='kaggle-stable-diffusion.ipynb',
                kernel_path='./config',
                gpu=True,
                guidance=7.5,
                steps=50,
                precision="fp16"
            )
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass  # Expected
