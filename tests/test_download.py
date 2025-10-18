import unittest
from unittest.mock import patch, MagicMock
from imggenhub.kaggle.core import download

class TestDownload(unittest.TestCase):
    @patch('imggenhub.kaggle.core.download._get_kaggle_command', return_value=['kaggle'])
    @patch('subprocess.run')
    @patch('pathlib.Path.mkdir')
    def test_run_downloads_output(self, mock_mkdir, mock_subproc, mock_kaggle_cmd):
        mock_result = MagicMock(returncode=0, stdout='Output file downloaded to', stderr='')
        mock_subproc.return_value = mock_result
        download.run(dest='output_images', kernel_id='foo/bar')
        mock_mkdir.assert_called()
        mock_subproc.assert_called()

    @patch('imggenhub.kaggle.core.download._get_kaggle_command', return_value=['kaggle'])
    @patch('subprocess.run')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.write_text')
    def test_run_non_zero_returncode_success(self, mock_write, mock_mkdir, mock_subproc, mock_kaggle_cmd):
        # Test when returncode != 0 but download succeeded
        mock_result = MagicMock(returncode=1, stdout='Output file downloaded to /path', stderr='some error')
        mock_subproc.return_value = mock_result
        download.run(dest='output_images', kernel_id='foo/bar')
        # Should not raise, should write logs
        mock_write.assert_called()

    @patch('imggenhub.kaggle.core.download._get_kaggle_command', return_value=['kaggle'])
    @patch('subprocess.run')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.write_text')
    @patch('imggenhub.kaggle.core.download.logging.warning')
    def test_run_non_zero_returncode_failure(self, mock_warning, mock_write, mock_mkdir, mock_subproc, mock_kaggle_cmd):
        # Test when returncode != 0 and no success message
        mock_result = MagicMock(returncode=1, stdout='some output', stderr='error')
        mock_subproc.return_value = mock_result
        # Mock the log file read to return the stdout
        with patch('pathlib.Path.read_text', return_value='some output'):
            download.run(dest='output_images', kernel_id='foo/bar')
        # Should log warning
        mock_warning.assert_called()

    @patch('shutil.which', return_value=True)
    @patch('pathlib.Path.exists', return_value=True)
    def test_get_kaggle_command_poetry(self, mock_exists, mock_which):
        with patch('subprocess.run', return_value=MagicMock(returncode=0)):
            cmd = download._get_kaggle_command()
            self.assertIn('poetry', cmd[0])

    @patch('shutil.which', return_value=True)
    @patch('pathlib.Path.exists', return_value=True)
    @patch('subprocess.run', return_value=MagicMock(returncode=1))
    def test_get_kaggle_command_poetry_fallback(self, mock_subproc, mock_exists, mock_which):
        cmd = download._get_kaggle_command()
        self.assertEqual(cmd, ["python", "-m", "kaggle.cli"])

if __name__ == '__main__':
    unittest.main()
