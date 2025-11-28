from unittest.mock import patch, MagicMock
from imggenhub.kaggle.core import download
import tempfile
from pathlib import Path
import sys


class DummyProcess:
    def __init__(self, poll_results=None, stdout="Output file downloaded", stderr=""):
        self.poll_results = poll_results[:] if poll_results is not None else [None]
        self.stdout_data = stdout
        self.stderr_data = stderr
        self.returncode = None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self.poll_results:
            result = self.poll_results.pop(0)
            if result is not None:
                self.returncode = result
            return result
        return None

    def communicate(self, timeout=None):
        return self.stdout_data, self.stderr_data

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


@patch('imggenhub.kaggle.core.download._get_kaggle_command', return_value=['kaggle'])
@patch('subprocess.Popen')
def test_run_downloads_output(mock_popen, mock_kaggle_cmd):
    mock_popen.return_value = DummyProcess()

    with tempfile.TemporaryDirectory() as td:
        dest = Path(td)
        (dest / 'output_images').mkdir(parents=True)
        img = dest / 'output_images' / 'image_001.png'
        img.write_text('fake')

        with patch.object(download, 'QUIET_PERIOD', 0.01), \
             patch.object(download, 'CHECK_INTERVAL', 0.001), \
             patch('imggenhub.kaggle.core.download.time.sleep', return_value=None):
            download.run(dest=str(dest), kernel_id='foo/bar')

        final_image = dest / 'images' / 'image_001.png'
        assert final_image.exists()


@patch('imggenhub.kaggle.core.download._get_kaggle_command', return_value=['kaggle'])
@patch('subprocess.Popen')
def test_run_non_zero_returncode_success(mock_popen, mock_kaggle_cmd):
    mock_popen.return_value = DummyProcess(poll_results=[1], stdout='Output file downloaded to /path')

    with tempfile.TemporaryDirectory() as td:
        dest = Path(td)
        (dest / 'output_images').mkdir(parents=True)
        img = dest / 'output_images' / 'image_001.png'
        img.write_text('fake')

        with patch.object(download, 'QUIET_PERIOD', 0.01), \
             patch.object(download, 'CHECK_INTERVAL', 0.001), \
             patch('imggenhub.kaggle.core.download.time.sleep', return_value=None):
            download.run(dest=str(dest), kernel_id='foo/bar')

        final_image = dest / 'images' / 'image_001.png'
        assert final_image.exists()


@patch('imggenhub.kaggle.core.download._get_kaggle_command', return_value=['kaggle'])
@patch('subprocess.Popen')
@patch('imggenhub.kaggle.core.download.logging.warning')
def test_run_non_zero_returncode_failure(mock_warning, mock_popen, mock_kaggle_cmd):
    mock_popen.return_value = DummyProcess(poll_results=[1], stdout='some output', stderr='error')

    with tempfile.TemporaryDirectory() as td:
        dest = Path(td)

        with patch.object(download, 'QUIET_PERIOD', 0.01), \
             patch.object(download, 'CHECK_INTERVAL', 0.001), \
             patch('imggenhub.kaggle.core.download.time.sleep', return_value=None):
            download.run(dest=str(dest), kernel_id='foo/bar')

        assert mock_warning.called


@patch('shutil.which', return_value=True)
@patch('pathlib.Path.exists', return_value=True)
def test_get_kaggle_command_poetry(mock_exists, mock_which):
    with patch('subprocess.run', return_value=MagicMock(returncode=0)):
        cmd = download._get_kaggle_command()
        assert 'poetry' in cmd[0]


@patch('shutil.which', return_value=True)
@patch('pathlib.Path.exists', return_value=True)
@patch('subprocess.run', return_value=MagicMock(returncode=1))
def test_get_kaggle_command_poetry_fallback(mock_subproc, mock_exists, mock_which):
    cmd = download._get_kaggle_command()
    assert cmd == [sys.executable, "-m", "kaggle.cli"]


@patch('imggenhub.kaggle.core.download._get_kaggle_command', return_value=['kaggle'])
@patch('subprocess.Popen')
def test_flatten_nested_output(mock_popen, mock_kaggle_cmd):
    mock_popen.return_value = DummyProcess(poll_results=[1])

    with tempfile.TemporaryDirectory() as td:
        dest = Path(td)
        nested = dest / 'output' / '20251115_000000'
        nested.mkdir(parents=True, exist_ok=True)
        img_file = nested / 'image_001.png'
        img_file.write_text('fakecontent')

        with patch.object(download, 'QUIET_PERIOD', 0.01), \
             patch.object(download, 'CHECK_INTERVAL', 0.001), \
             patch('imggenhub.kaggle.core.download.time.sleep', return_value=None):
            download.run(dest=str(dest), kernel_id='foo/bar')

        assert (dest / 'images' / 'image_001.png').exists()
