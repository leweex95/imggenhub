import unittest
from unittest.mock import patch, MagicMock
from imggenhub.kaggle.utils import poll_status

class TestPollStatus(unittest.TestCase):
    @patch('imggenhub.kaggle.utils.poll_status._get_kaggle_command', return_value=['kaggle'])
    @patch('subprocess.run')
    def test_run_status_complete(self, mock_subproc, mock_kaggle_cmd):
        mock_subproc.return_value = MagicMock(returncode=0, stdout='has status "kernelworkerstatus.complete"', stderr='')
        status = poll_status.run(kernel_id='foo/bar', poll_interval=0)
        self.assertEqual(status, 'kernelworkerstatus.complete')

    @patch('imggenhub.kaggle.utils.poll_status._get_kaggle_command', return_value=['kaggle'])
    @patch('subprocess.run')
    def test_run_status_error(self, mock_subproc, mock_kaggle_cmd):
        mock_subproc.return_value = MagicMock(returncode=0, stdout='has status "kernelworkerstatus.error"', stderr='')
        status = poll_status.run(kernel_id='foo/bar', poll_interval=0)
        self.assertEqual(status, 'kernelworkerstatus.error')

    @patch('imggenhub.kaggle.utils.poll_status._get_kaggle_command', return_value=['kaggle'])
    @patch('subprocess.run')
    def test_run_status_unknown(self, mock_subproc, mock_kaggle_cmd):
        mock_subproc.return_value = MagicMock(returncode=1, stdout='', stderr='error')
        status = poll_status.run(kernel_id='foo/bar', poll_interval=0)
        self.assertEqual(status, 'unknown')

    @patch('shutil.which', return_value=True)
    @patch('pathlib.Path.exists', return_value=True)
    def test_get_kaggle_command_poetry(self, mock_exists, mock_which):
        with patch('subprocess.run', return_value=MagicMock(returncode=0)):
            cmd = poll_status._get_kaggle_command()
            self.assertIn('poetry', cmd[0])

if __name__ == '__main__':
    unittest.main()
