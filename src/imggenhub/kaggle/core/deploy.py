import os
import re
import json
from pathlib import Path
import subprocess
import shutil
import logging


def run(prompts_list, notebook, model_id, kernel_path=".", gpu=None, refiner_model_id=None, guidance=None, steps=None, precision="fp16", negative_prompt=None, output_dir=None, two_stage_refiner=False, refiner_guidance=None, refiner_steps=None, refiner_precision=None, refiner_negative_prompt=None, hf_token=None):
    """
    Deploy Kaggle notebook kernel, optionally overriding prompts and model.
    Uses the specified notebook; user is responsible for matching notebook to model.
    """
    
    # Resolve notebook path relative to kernel_path
    nb_path = Path(kernel_path) / notebook
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")
    
    # Load notebook
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Update parameters in notebook
    # Skip parameter updates for FLUX notebook to avoid breaking JSON indentation
    if "flux" not in str(notebook).lower():
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
                # Update TWO_STAGE_REFINER if provided
                for i, line in enumerate(cell["source"]):
                    if "TWO_STAGE_REFINER =" in line:
                        cell["source"][i] = f"TWO_STAGE_REFINER = {two_stage_refiner}\n"
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
    if "flux" not in str(notebook).lower():
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)

    # Update kernel metadata
    metadata_path = Path(kernel_path) / "kernel-metadata.json"
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            kernel_meta = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # Create default metadata if file is corrupted or missing
        kernel_meta = {
            "id": "leventecsibi/stable-diffusion-batch-generator",
            "title": "Stable Diffusion Batch Generator",
            "code_file": "",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "model_sources": [],
            "metadata": {
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3",
                    "language": "python"
                }
            }
        }
        logging.warning(f"Kernel metadata file corrupted or missing, using defaults: {metadata_path}")
    
    # Store original code_file for verification
    original_code_file = kernel_meta.get("code_file", "UNKNOWN")
    
    if gpu is not None:
        kernel_meta["enable_gpu"] = str(gpu).lower()
    # Update code_file to point to the correct notebook
    kernel_meta["code_file"] = notebook.name
    logging.info(f"Updated kernel metadata to use notebook: {notebook}")
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(kernel_meta, f, indent=2)
    
    # VERIFY that metadata was updated correctly
    with open(metadata_path, "r", encoding="utf-8") as f:
        verify_meta = json.load(f)
    
    if verify_meta.get("code_file") != notebook.name:
        raise RuntimeError(
            f"ERROR: Metadata update verification failed!\n"
            f"  Expected code_file: {notebook.name}\n"
            f"  Got code_file: {verify_meta.get('code_file')}\n"
            f"  Metadata file: {metadata_path}"
        )
    
    logging.info(f"✓ VERIFIED: Metadata correctly updated from '{original_code_file}' to '{notebook.name}'")
    
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
                check=False,
                timeout=10
            )
            if result.returncode == 0:
                return ["poetry", "run", "python", "-m", "kaggle.cli"]
        except Exception:
            pass
    
    # Fallback to direct python call
    return ["python", "-m", "kaggle.cli"]
