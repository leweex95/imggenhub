import unittest
from unittest.mock import patch, mock_open
from imggenhub.kaggle.utils import prompts

class TestPrompts(unittest.TestCase):
    def test_prompt_only(self):
        result = prompts.resolve_prompts(prompt="foo")
        self.assertEqual(result, ["foo"])

    def test_prompts_only(self):
        result = prompts.resolve_prompts(prompts=["a", "b"])
        self.assertEqual(result, ["a", "b"])

    def test_prompt_and_prompts_error(self):
        with self.assertRaises(ValueError):
            prompts.resolve_prompts(prompt="foo", prompts=["bar"])

    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='["a", "b"]')
    @patch('json.load', return_value=["a", "b"])
    def test_prompts_file(self, mock_json, mock_file, mock_exists):
        result = prompts.resolve_prompts(prompts_file="file.json")
        self.assertEqual(result, ["a", "b"])

    @patch('pathlib.Path.exists', return_value=False)
    def test_prompts_file_not_found(self, mock_exists):
        with self.assertRaises(FileNotFoundError):
            prompts.resolve_prompts(prompts_file="file.json")

    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    @patch('json.load', return_value={})
    def test_prompts_file_invalid(self, mock_json, mock_file, mock_exists):
        with self.assertRaises(ValueError):
            prompts.resolve_prompts(prompts_file="file.json")

    def test_no_input(self):
        with self.assertRaises(ValueError):
            prompts.resolve_prompts()

if __name__ == '__main__':
    unittest.main()
