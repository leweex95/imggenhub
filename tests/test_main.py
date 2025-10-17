import unittest
from unittest.mock import patch, MagicMock
from imggenhub.kaggle import main

class TestMainPipeline(unittest.TestCase):
    @patch('imggenhub.kaggle.main.download.run')
    @patch('imggenhub.kaggle.main.poll_status.run', return_value='kernelworkerstatus.complete')
    @patch('imggenhub.kaggle.main.deploy.run')
    @patch('imggenhub.kaggle.main.resolve_prompts', return_value=['prompt'])
    @patch('pathlib.Path.mkdir')
    def test_run_pipeline_success(self, mock_mkdir, mock_resolve, mock_deploy, mock_poll, mock_download):
        main.run_pipeline(None, './config/kaggle-notebook-image-generation.ipynb', './config', gpu=True, dest='output_images')
        mock_deploy.assert_called()
        mock_poll.assert_called()
        mock_download.assert_called()

    @patch('imggenhub.kaggle.main.download.run')
    @patch('imggenhub.kaggle.main.poll_status.run', return_value='kernelworkerstatus.error')
    @patch('imggenhub.kaggle.main.deploy.run')
    @patch('imggenhub.kaggle.main.resolve_prompts', return_value=['prompt'])
    @patch('pathlib.Path.mkdir')
    def test_run_pipeline_kernel_error(self, mock_mkdir, mock_resolve, mock_deploy, mock_poll, mock_download):
        with self.assertRaises(RuntimeError):
            main.run_pipeline(None, './config/kaggle-notebook-image-generation.ipynb', './config', gpu=True, dest='output_images')

if __name__ == '__main__':
    unittest.main()
