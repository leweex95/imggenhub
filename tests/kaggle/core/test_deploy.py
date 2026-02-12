from unittest.mock import MagicMock, mock_open, patch

from imggenhub.kaggle.core import deploy


def test_run_updates_notebook_and_metadata_and_pushes():
    notebook_data = {"cells": [{"cell_type": "code", "source": ["PROMPTS = []\n", "MODEL_ID = \"foo\"\n"]}]}
    metadata = {"enable_gpu": "false", "code_file": "notebook.json"}
    kaggle_result = MagicMock(stdout="Kernel pushed successfully", stderr="")

    with patch("builtins.open", new_callable=mock_open), \
         patch("json.dump") as mock_json_dump, \
         patch("json.load", side_effect=[notebook_data, metadata, metadata]), \
         patch("imggenhub.kaggle.core.deploy._get_kaggle_command", return_value=["kaggle"]), \
         patch("imggenhub.kaggle.core.deploy.load_kaggle_config", return_value={"deployment_timeout_minutes": 30, "retry_interval_seconds": 60}), \
         patch("subprocess.run", return_value=kaggle_result) as mock_subproc, \
         patch("pathlib.Path.exists", return_value=True):
        deploy.run(["a", "b"], "notebook.json", "bar", ".", gpu=True)

    assert mock_subproc.called
    assert mock_json_dump.called


def test_get_kaggle_command_poetry():
    with patch("shutil.which", return_value=True), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("subprocess.run", return_value=MagicMock(returncode=0)):
        cmd = deploy._get_kaggle_command()
    assert "poetry" in cmd[0]
