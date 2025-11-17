from unittest.mock import patch, mock_open, MagicMock
import json
from pathlib import Path
import sys
import os

sys.modules['shutil'] = __import__('shutil')

from unittest.mock import patch, mock_open, MagicMock
import json
from pathlib import Path
import sys
import os

sys.modules['shutil'] = __import__('shutil')

from imggenhub.kaggle.core import deploy

def test_run_updates_notebook_and_metadata_and_pushes():
    with patch('builtins.open', new_callable=mock_open, read_data='{"cells": [{"cell_type": "code", "source": ["PROMPTS = []\n", "MODEL_ID = \"foo\"\n"]}]}') as mock_file, \
         patch('json.dump') as mock_json_dump, \
         patch('json.load', return_value={"cells": [{"cell_type": "code", "source": ["PROMPTS = []\n", "MODEL_ID = \"foo\"\n"]}]}), \
         patch('imggenhub.kaggle.core.deploy._get_kaggle_command', return_value=['kaggle']) as mock_kaggle_cmd, \
         patch('subprocess.run') as mock_subproc:
        
        prompts_list = ["a", "b"]
        notebook = "notebook.json"
        kernel_path = "."
        metadata = {"enable_gpu": "false"}
        # Patch Path.exists and json.load for metadata
        with patch('pathlib.Path.exists', return_value=True), \
             patch('json.load', side_effect=[{"cells": [{"cell_type": "code", "source": ["PROMPTS = []\n", "MODEL_ID = \"foo\"\n"]}]}, metadata]):
            with patch('builtins.open', mock_open()):
                deploy.run(prompts_list, notebook, kernel_path, gpu=True, model_id="bar")
        assert mock_subproc.called
        assert mock_json_dump.called

def test_get_kaggle_command_poetry():
    with patch('shutil.which', return_value=True), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('subprocess.run', return_value=MagicMock(returncode=0)):
        
        cmd = deploy._get_kaggle_command()
        assert 'poetry' in cmd[0]
