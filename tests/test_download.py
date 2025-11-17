from unittest.mock import patch, MagicMock
from imggenhub.kaggle.core import download
import tempfile
from pathlib import Path

@patch('imggenhub.kaggle.core.download._get_kaggle_command', return_value=['kaggle'])
@patch('subprocess.run')
@patch('pathlib.Path.mkdir')
def test_run_downloads_output(mock_mkdir, mock_subproc, mock_kaggle_cmd):
    mock_result = MagicMock(returncode=0, stdout='Output file downloaded to', stderr='')
    mock_subproc.return_value = mock_result

    # Mock the write_text calls to avoid file system operations
    with patch('pathlib.Path.write_text'):
        download.run(dest='output_images', kernel_id='foo/bar')

    assert mock_mkdir.called
    assert mock_subproc.called

@patch('imggenhub.kaggle.core.download._get_kaggle_command', return_value=['kaggle'])
@patch('subprocess.run')
@patch('pathlib.Path.mkdir')
@patch('pathlib.Path.write_text')
def test_run_non_zero_returncode_success(mock_write, mock_mkdir, mock_subproc, mock_kaggle_cmd):
    # Test when returncode != 0 but download succeeded
    mock_result = MagicMock(returncode=1, stdout='Output file downloaded to /path', stderr='some error')
    mock_subproc.return_value = mock_result

    # Mock read_text as well
    with patch('pathlib.Path.read_text', return_value='Output file downloaded to /path'):
        download.run(dest='output_images', kernel_id='foo/bar')

    # Should not raise, should write logs
    assert mock_write.called

@patch('imggenhub.kaggle.core.download._get_kaggle_command', return_value=['kaggle'])
@patch('subprocess.run')
@patch('pathlib.Path.mkdir')
@patch('pathlib.Path.write_text')
@patch('imggenhub.kaggle.core.download.logging.warning')
def test_run_non_zero_returncode_failure(mock_warning, mock_write, mock_mkdir, mock_subproc, mock_kaggle_cmd):
    # Test when returncode != 0 and no success message
    mock_result = MagicMock(returncode=1, stdout='some output', stderr='error')
    mock_subproc.return_value = mock_result
    # Mock the log file read to return the stdout
    with patch('pathlib.Path.read_text', return_value='some output'):
        download.run(dest='output_images', kernel_id='foo/bar')
    # Should log warning
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
    assert cmd == ["python", "-m", "kaggle.cli"]

@patch('imggenhub.kaggle.core.download._get_kaggle_command', return_value=['kaggle'])
@patch('subprocess.run')
def test_flatten_nested_output(mock_subproc, mock_kaggle_cmd):
        # Create a temporary dest folder and nested structure to simulate kaggle output
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "output_images"
            nested = dest / "output" / "20251115_000000"
            nested.mkdir(parents=True, exist_ok=True)
            # Create a fake image file inside nested folder
            img_file = nested / "image_001.png"
            img_file.write_text('fakecontent')

            mock_result = MagicMock(returncode=0, stdout='Output file downloaded to', stderr='')
            mock_subproc.return_value = mock_result
            # Run download and ensure file is moved up to dest root
            download.run(dest=str(dest), kernel_id='foo/bar')
            # File should exist in dest root now and not under nested path
            assert (dest / 'image_001.png').exists()
            assert not (nested / 'image_001.png').exists()
