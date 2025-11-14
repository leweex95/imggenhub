import os
import re
import json
from pathlib import Path
import subprocess
import shutil
import logging


def run(prompts_list, notebook="kaggle-notebook-image-generation.ipynb", kernel_path=".", gpu=None, model_id=None, refiner_model_id=None, guidance=None, steps=None, precision="fp16", negative_prompt=None, output_dir=None):
    """
    Deploy Kaggle notebook kernel, optionally overriding prompts and model.
    """

    # Load notebook
    nb_path = Path(notebook)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Update parameters in notebook
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
            # Update REFINER_MODEL_ID if provided
            if refiner_model_id:
                for i, line in enumerate(cell["source"]):
                    if line.strip().startswith("REFINER_MODEL_ID ="):
                        cell["source"][i] = f"REFINER_MODEL_ID = \"{refiner_model_id}\"\n"
                        break
            # Update GUIDANCE if provided
            if guidance is not None:
                for i, line in enumerate(cell["source"]):
                    if line.strip().startswith("GUIDANCE ="):
                        cell["source"][i] = f"GUIDANCE = {guidance}\n"
                        break
            # Update STEPS if provided
            if steps is not None:
                for i, line in enumerate(cell["source"]):
                    if line.strip().startswith("STEPS ="):
                        cell["source"][i] = f"STEPS = {steps}\n"
                        break
            # Update USE_REFINER automatically based on refiner model
            use_refiner = refiner_model_id is not None
            for i, line in enumerate(cell["source"]):
                if line.strip().startswith("USE_REFINER ="):
                    cell["source"][i] = f"USE_REFINER = {use_refiner}\n"
                    break
            # Update PRECISION if provided
            if precision:
                for i, line in enumerate(cell["source"]):
                    if line.strip().startswith("PRECISION ="):
                        cell["source"][i] = f"PRECISION = \"{precision}\"\n"
                        break
            # Update NEGATIVE_PROMPT if provided
            if negative_prompt:
                for i, line in enumerate(cell["source"]):
                    if line.strip().startswith("NEGATIVE_PROMPT ="):
                        cell["source"][i] = f"NEGATIVE_PROMPT = \"{negative_prompt}\"\n"
                        break
            # Update OUTPUT_DIR if provided
            if output_dir:
                for i, line in enumerate(cell["source"]):
                    if line.strip().startswith("OUTPUT_DIR ="):
                        # Use forward slashes (cross-platform compatible for Kaggle Linux)
                        # Extract just the folder name, not the full path
                        output_dir_unix = str(output_dir).replace("\\", "/")
                        cell["source"][i] = f"OUTPUT_DIR = \"{output_dir_unix}\"\n"
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
    result = subprocess.run([*kaggle_cmd, "kernels", "push", "-p", str(kernel_path)], 
                           check=True, capture_output=True, text=True, encoding='utf-8')
    
    # Log output safely
    if result.stdout:
        logging.debug(f"Kaggle push output: {result.stdout}")
    if result.stderr:
        logging.debug(f"Kaggle push stderr: {result.stderr}")


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
