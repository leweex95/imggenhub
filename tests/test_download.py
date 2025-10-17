import unittest
from unittest.mock import patch, MagicMock
from imggenhub.kaggle.core import download

class TestDownload(unittest.TestCase):
    @patch('imggenhub.kaggle.core.download._get_kaggle_command', return_value=['kaggle'])
    @patch('subprocess.run')
    @patch('pathlib.Path.mkdir')
    def test_run_downloads_output(self, mock_mkdir, mock_subproc, mock_kaggle_cmd):
        download.run(dest='output_images', kernel_id='foo/bar')
        mock_mkdir.assert_called()
        mock_subproc.assert_called()

    @patch('shutil.which', return_value=True)
    @patch('pathlib.Path.exists', return_value=True)
    def test_get_kaggle_command_poetry(self, mock_exists, mock_which):
        with patch('subprocess.run', return_value=MagicMock(returncode=0)):
            cmd = download._get_kaggle_command()
            self.assertIn('poetry', cmd[0])

if __name__ == '__main__':
    unittest.main()
