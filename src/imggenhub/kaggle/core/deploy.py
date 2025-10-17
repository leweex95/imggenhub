import os
import re
import json
from pathlib import Path
import subprocess
import shutil


def run(prompts_list, notebook="kaggle-notebook-image-generation.ipynb", kernel_path=".", gpu=None, model_id=None):
    """
    Deploy Kaggle notebook kernel, optionally overriding prompts and model.
    """

    # Load notebook
    nb_path = Path(notebook)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Update PROMPTS and MODEL_ID in notebook
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            # Update PROMPTS
            for i, line in enumerate(cell["source"]):
                if line.strip().startswith("PROMPTS ="):
                    cell["source"][i] = f"PROMPTS = {prompts_list}\n"
                    break
            # Update MODEL_ID if provided
            if model_id:
                for i, line in enumerate(cell["source"]):
                    if line.strip().startswith("MODEL_ID ="):
                        cell["source"][i] = f"MODEL_ID = \"{model_id}\"\n"
                        break

    # Save updated notebook
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    # Update kernel metadata
    metadata_path = Path(kernel_path) / "kernel-metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        kernel_meta = json.load(f)
    if gpu is not None:
        kernel_meta["enable_gpu"] = str(gpu).lower()
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(kernel_meta, f, indent=2)

    # Push via Kaggle CLI - try Poetry first, fallback to direct python
    kaggle_cmd = _get_kaggle_command()
    subprocess.run([*kaggle_cmd, "kernels", "push", "-p", str(kernel_path)], check=True)


def _get_kaggle_command():
    """
    Get the appropriate command to run Kaggle CLI.
    
    Returns:
        list: Command parts to execute kaggle CLI
    """
    # Check if poetry is available and we're in a poetry project
    def find_pyproject_toml():
        current = Path.cwd()
        while current != current.parent:  # Stop at root
            if (current / "pyproject.toml").exists():
                return True
            current = current.parent
        return False
    
    if shutil.which("poetry") and find_pyproject_toml():
        try:
            # Test if poetry can run the kaggle command
            result = subprocess.run(
                ["poetry", "run", "python", "-c", "import kaggle"],
                capture_output=True,
                check=False
            )
            if result.returncode == 0:
                return ["poetry", "run", "python", "-m", "kaggle.cli"]
        except Exception:
            pass
    
    # Fallback to direct python call
    return ["python", "-m", "kaggle.cli"]
