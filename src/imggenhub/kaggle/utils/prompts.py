from pathlib import Path
import json

def resolve_prompts(prompts_file=None, prompt=None, prompts=None):
    """Return a list of prompts based on inputs"""
    if prompt:
        # Accept both a single string and a list of strings
        if isinstance(prompt, str):
            return [prompt]
        return prompt

    if prompts_file:
        prompts_path = Path(prompts_file)
        if not prompts_path.is_absolute():
            prompts_path = Path(__file__).parent / prompts_path
        if prompts_path.exists():
            with open(prompts_path, "r", encoding="utf-8") as f:
                prompts_list = json.load(f)
            if not isinstance(prompts_list, list) or not prompts_list:
                raise ValueError(f"Prompts file must contain a non-empty list, got: {prompts_list}")
            return prompts_list
        else:
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    raise ValueError("No prompts provided: specify --prompt or --prompts_file")
