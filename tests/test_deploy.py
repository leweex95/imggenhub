import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
from pathlib import Path
import sys
import os

sys.modules['shutil'] = __import__('shutil')

from imggenhub.kaggle.core import deploy

class TestDeploy(unittest.TestCase):
    @patch('builtins.open', new_callable=mock_open, read_data='{"cells": [{"cell_type": "code", "source": ["PROMPTS = []\n", "MODEL_ID = \"foo\"\n"]}]}')
    @patch('json.dump')
    @patch('json.load', return_value={"cells": [{"cell_type": "code", "source": ["PROMPTS = []\n", "MODEL_ID = \"foo\"\n"]}]})
    @patch('imggenhub.kaggle.core.deploy._get_kaggle_command', return_value=['kaggle'])
    @patch('subprocess.run')
    def test_run_updates_notebook_and_metadata_and_pushes(self, mock_subproc, mock_kaggle_cmd, mock_json_load, mock_json_dump, mock_file):
        prompts_list = ["a", "b"]
        notebook = "notebook.json"
        kernel_path = "."
        metadata = {"enable_gpu": "false"}
        # Patch Path.exists and json.load for metadata
        with patch('pathlib.Path.exists', return_value=True), \
             patch('json.load', side_effect=[{"cells": [{"cell_type": "code", "source": ["PROMPTS = []\n", "MODEL_ID = \"foo\"\n"]}]}, metadata]):
            with patch('builtins.open', mock_open()):
                deploy.run(prompts_list, notebook, kernel_path, gpu=True, model_id="bar")
        self.assertTrue(mock_subproc.called)
        self.assertTrue(mock_json_dump.called)

    @patch('shutil.which', return_value=True)
    @patch('pathlib.Path.exists', return_value=True)
    def test_get_kaggle_command_poetry(self, mock_exists, mock_which):
        with patch('subprocess.run', return_value=MagicMock(returncode=0)):
            cmd = deploy._get_kaggle_command()
            self.assertIn('poetry', cmd[0])

if __name__ == '__main__':
    unittest.main()
