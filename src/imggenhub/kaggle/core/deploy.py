import os
import re
import json
from pathlib import Path
import subprocess
import shutil
import logging


def run(prompts_list, notebook="kaggle-notebook-image-generation.ipynb", kernel_path=".", gpu=None, model_id=None, refiner_model_id=None, guidance=None, steps=None, precision=None, negative_prompt=None, output_dir=None, refiner_guidance=None, refiner_steps=None, refiner_precision=None, refiner_negative_prompt=None, hf_token=None):
    """
    Deploy Kaggle notebook kernel, optionally overriding prompts and model.
    Automatically selects the correct notebook based on model type.
    """
    
    # Auto-select notebook based on model_id
    if model_id:
        # Use FLUX notebook for Kaggle FLUX models
        if "flux" in model_id.lower():
            notebook = "kaggle-notebook-flux-generation.ipynb"
            logging.info(f"Selected FLUX notebook for model: {model_id}")
        # Use standard notebook for Stability AI and HuggingFace models
        elif any(org in model_id.lower() for org in ["stabilityai", "black-forest-labs", "runwayml"]):
            notebook = "kaggle-notebook-image-generation.ipynb"
            logging.info(f"Selected standard notebook for model: {model_id}")
    
    # Resolve notebook path relative to kernel_path
    nb_path = Path(kernel_path) / notebook
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")
    
    # Load notebook
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Update parameters in notebook
    # Skip parameter updates for FLUX notebook to avoid breaking JSON indentation
    if "flux" not in notebook.lower():
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
                        if "MODEL_ID =" in line:
                            cell["source"][i] = f"MODEL_ID = \"{model_id}\"\n"
                            break
                # Update REFINER_MODEL_ID if provided
                if refiner_model_id:
                    for i, line in enumerate(cell["source"]):
                        if "REFINER_MODEL_ID =" in line:
                            cell["source"][i] = f"REFINER_MODEL_ID = \"{refiner_model_id}\"\n"
                            break
                # Update GUIDANCE if provided
                if guidance is not None:
                    for i, line in enumerate(cell["source"]):
                        if "GUIDANCE =" in line:
                            cell["source"][i] = f"GUIDANCE = {guidance}\n"
                            break
                # Update STEPS if provided
                if steps is not None:
                    for i, line in enumerate(cell["source"]):
                        if "STEPS =" in line:
                            cell["source"][i] = f"STEPS = {steps}\n"
                            break
                # Update USE_REFINER based on whether a refiner model is provided
                # (No assumptions about model compatibility; let the pipeline handle it)
                use_refiner = refiner_model_id is not None
                for i, line in enumerate(cell["source"]):
                    if "USE_REFINER =" in line:
                        cell["source"][i] = f"USE_REFINER = {use_refiner}\n"
                        break
                # Update PRECISION if provided
                if precision:
                    for i, line in enumerate(cell["source"]):
                        if "PRECISION =" in line:
                            cell["source"][i] = f"PRECISION = \"{precision}\"\n"
                            break
                # Update NEGATIVE_PROMPT if provided
                if negative_prompt:
                    for i, line in enumerate(cell["source"]):
                        if "NEGATIVE_PROMPT =" in line:
                            cell["source"][i] = f"NEGATIVE_PROMPT = \"{negative_prompt}\"\n"
                            break
                # Update OUTPUT_DIR if provided
                if output_dir:
                    for i, line in enumerate(cell["source"]):
                        if "OUTPUT_DIR =" in line:
                            # Use forward slashes (cross-platform compatible for Kaggle Linux)
                            # Write only the output folder name (basename) so the notebook
                            # can prepend 'output/' if desired and avoid nested paths.
                            output_dir_unix = str(output_dir).replace("\\", "/")
                            output_basename = os.path.basename(output_dir_unix)
                            cell["source"][i] = f"OUTPUT_DIR = \"{output_basename}\"\n"
                            break
                # Update REFINER_GUIDANCE if provided
                if refiner_guidance is not None:
                    for i, line in enumerate(cell["source"]):
                        if "REFINER_GUIDANCE =" in line:
                            cell["source"][i] = f"REFINER_GUIDANCE = {refiner_guidance}\n"
                            break
                # Update REFINER_STEPS if provided
                if refiner_steps is not None:
                    for i, line in enumerate(cell["source"]):
                        if "REFINER_STEPS =" in line:
                            cell["source"][i] = f"REFINER_STEPS = {refiner_steps}\n"
                            break
                # Update REFINER_PRECISION if provided
                if refiner_precision:
                    for i, line in enumerate(cell["source"]):
                        if "REFINER_PRECISION =" in line:
                            cell["source"][i] = f"REFINER_PRECISION = \"{refiner_precision}\"\n"
                            break
                # Update REFINER_NEGATIVE_PROMPT if provided
                if refiner_negative_prompt:
                    for i, line in enumerate(cell["source"]):
                        if "REFINER_NEGATIVE_PROMPT =" in line:
                            cell["source"][i] = f"REFINER_NEGATIVE_PROMPT = \"{refiner_negative_prompt}\"\n"
                            break
                # Update HF_TOKEN if provided
                if hf_token:
                    for i, line in enumerate(cell["source"]):
                        if "HF_TOKEN =" in line:
                            cell["source"][i] = f"HF_TOKEN = \"{hf_token}\"\n"
                            break
    else:
        logging.info(f"Skipping parameter updates for FLUX notebook to prevent indentation issues")

    # Save updated notebook (only if we modified it)
    if "flux" not in notebook.lower():
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)

    # Update kernel metadata
    metadata_path = Path(kernel_path) / "kernel-metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        kernel_meta = json.load(f)
    
    # Store original code_file for verification
    original_code_file = kernel_meta.get("code_file", "UNKNOWN")
    
    if gpu is not None:
        kernel_meta["enable_gpu"] = str(gpu).lower()
    # Update code_file to point to the correct notebook
    kernel_meta["code_file"] = notebook
    logging.info(f"Updated kernel metadata to use notebook: {notebook}")
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(kernel_meta, f, indent=2)
    logging.info(f"✓ Updated metadata from '{original_code_file}' to '{notebook}'")
    
    # Push via Kaggle CLI - try Poetry first, fallback to direct python
    kaggle_cmd = _get_kaggle_command()
    logging.info(f"Deploying to Kaggle with command: {' '.join(kaggle_cmd)}")
    result = subprocess.run([*kaggle_cmd, "kernels", "push", "-p", str(kernel_path)], 
                           check=True, capture_output=True, text=True, encoding='utf-8')
    
    # Log output safely
    if result.stdout:
        logging.info(f"Kaggle push output: {result.stdout}")
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
