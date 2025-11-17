from unittest.mock import patch, mock_open
from imggenhub.kaggle.utils import prompts

def test_prompt_only():
    result = prompts.resolve_prompts(prompt="foo")
    assert result == ["foo"]

def test_prompts_only():
    result = prompts.resolve_prompts(prompts=["a", "b"])
    assert result == ["a", "b"]

    def test_prompt_and_prompts_error(self):
        try:
            prompts.resolve_prompts(prompt="foo", prompts=["bar"])
            assert False, "Expected ValueError"
        except ValueError:
            pass  # Expected

    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='["a", "b"]')
    @patch('json.load', return_value=["a", "b"])
    def test_prompts_file(self, mock_json, mock_file, mock_exists):
        result = prompts.resolve_prompts(prompts_file="file.json")
        assert result == ["a", "b"]

    @patch('pathlib.Path.exists', return_value=False)
    def test_prompts_file_not_found(self, mock_exists):
        try:
            prompts.resolve_prompts(prompts_file="file.json")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected

    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    @patch('json.load', return_value={})
    def test_prompts_file_invalid(self, mock_json, mock_file, mock_exists):
        try:
            prompts.resolve_prompts(prompts_file="file.json")
            assert False, "Expected ValueError"
        except ValueError:
            pass  # Expected

    def test_no_input(self):
        try:
            prompts.resolve_prompts()
            assert False, "Expected ValueError"
        except ValueError:
            pass  # Expected
