from unittest.mock import patch, MagicMock, call
from imggenhub.kaggle.utils import poll_status

@patch('imggenhub.kaggle.utils.poll_status._get_kaggle_command', return_value=['kaggle'])
@patch('imggenhub.kaggle.utils.poll_status.subprocess.run')
def test_run_status_complete(mock_subproc, mock_kaggle_cmd):
    mock_subproc.return_value = MagicMock(returncode=0, stdout='has status "kernelworkerstatus.complete"', stderr='')
    status = poll_status.run(kernel_id='foo/bar', poll_interval=0)
    assert status == 'kernelworkerstatus.complete'

    @patch('imggenhub.kaggle.utils.poll_status._get_kaggle_command', return_value=['kaggle'])
    @patch('imggenhub.kaggle.utils.poll_status.subprocess.run')
    def test_run_status_error(self, mock_subproc, mock_kaggle_cmd):
        mock_subproc.return_value = MagicMock(returncode=0, stdout='has status "kernelworkerstatus.error"', stderr='')
        status = poll_status.run(kernel_id='foo/bar', poll_interval=0)
        assert status == 'kernelworkerstatus.error'

    @patch('imggenhub.kaggle.utils.poll_status._get_kaggle_command', return_value=['kaggle'])
    @patch('imggenhub.kaggle.utils.poll_status.subprocess.run')
    @patch('imggenhub.kaggle.utils.poll_status.logging.error')  # Suppress error logging in tests
    def test_run_status_unknown(self, mock_logging_error, mock_subproc, mock_kaggle_cmd):
        mock_subproc.return_value = MagicMock(returncode=1, stdout='', stderr='error')
        status = poll_status.run(kernel_id='foo/bar', poll_interval=0)
        assert status == 'unknown'

    @patch('imggenhub.kaggle.utils.poll_status._get_kaggle_command', return_value=['kaggle'])
    @patch('imggenhub.kaggle.utils.poll_status.subprocess.run')
    @patch('time.sleep')
    def test_run_polling_multiple_times(self, mock_sleep, mock_subproc, mock_kaggle_cmd):
        # First call: in progress, second: complete
        mock_subproc.side_effect = [
            MagicMock(returncode=0, stdout='has status "kernelworkerstatus.queued"', stderr=''),
            MagicMock(returncode=0, stdout='has status "kernelworkerstatus.complete"', stderr='')
        ]
        status = poll_status.run(kernel_id='foo/bar', poll_interval=1)
        assert status == 'kernelworkerstatus.complete'
        assert mock_subproc.call_count == 2
        mock_sleep.assert_called_once_with(1)

    @patch('imggenhub.kaggle.utils.poll_status._get_kaggle_command', return_value=['kaggle'])
    @patch('imggenhub.kaggle.utils.poll_status.subprocess.run')
    def test_run_regex_no_match(self, mock_subproc, mock_kaggle_cmd):
        mock_subproc.return_value = MagicMock(returncode=0, stdout='no status here', stderr='')
        status = poll_status.run(kernel_id='foo/bar', poll_interval=0)
        assert status == 'unknown'

    @patch('shutil.which', return_value=True)
    @patch('pathlib.Path.exists', return_value=True)
    def test_get_kaggle_command_poetry(self, mock_exists, mock_which):
        with patch('subprocess.run', return_value=MagicMock(returncode=0)):
            cmd = poll_status._get_kaggle_command()
            assert 'poetry' in cmd[0]

    @patch('shutil.which', return_value=True)
    @patch('pathlib.Path.exists', return_value=False)
    def test_get_kaggle_command_fallback_no_pyproject(self, mock_exists, mock_which):
        cmd = poll_status._get_kaggle_command()
        assert cmd == ["python", "-m", "kaggle.cli"]

    @patch('shutil.which', return_value=True)
    @patch('pathlib.Path.exists', return_value=True)
    @patch('subprocess.run', side_effect=[MagicMock(returncode=1), MagicMock(returncode=0)])  # First fails, second succeeds
    def test_get_kaggle_command_poetry_fallback_on_error(self, mock_subproc, mock_exists, mock_which):
        cmd = poll_status._get_kaggle_command()
        assert cmd == ["python", "-m", "kaggle.cli"]

    @patch('shutil.which', return_value=False)
    def test_get_kaggle_command_no_poetry(self, mock_which):
        cmd = poll_status._get_kaggle_command()
        assert cmd == ["python", "-m", "kaggle.cli"]
