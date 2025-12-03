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
    notebook_data = {"cells": [{"cell_type": "code", "source": ["PROMPTS = []\n", "MODEL_ID = \"foo\"\n"]}]}
    metadata = {"enable_gpu": "false", "code_file": "notebook.json"}
    
    with patch('builtins.open', new_callable=mock_open) as mock_file, \
         patch('json.dump') as mock_json_dump, \
         patch('json.load', side_effect=[notebook_data, metadata, metadata]), \
         patch('imggenhub.kaggle.core.deploy._get_kaggle_command', return_value=['kaggle']) as mock_kaggle_cmd, \
         patch('subprocess.run') as mock_subproc, \
         patch('pathlib.Path.exists', return_value=True):
        
        prompts_list = ["a", "b"]
        notebook = "notebook.json"
        kernel_path = "."
        deploy.run(prompts_list, notebook, "bar", kernel_path, gpu=True)
        
        assert mock_subproc.called
        assert mock_json_dump.called

def test_get_kaggle_command_poetry():
    with patch('shutil.which', return_value=True), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('subprocess.run', return_value=MagicMock(returncode=0)):
        
        cmd = deploy._get_kaggle_command()
        assert 'poetry' in cmd[0]
