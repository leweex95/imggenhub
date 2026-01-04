import os
import re
import json
import time
from pathlib import Path
import subprocess
import shutil
import logging
from imggenhub.kaggle.utils.config_loader import load_kaggle_config


def _update_param(source_lines, param_name, value, is_list=False):
    """
    Update a parameter in notebook source code lines.
    
    Args:
        source_lines: List of source code lines
        param_name: Name of parameter (e.g., "PROMPTS", "MODEL_ID")
        value: New value
        is_list: True if value should be formatted as Python list
    
    Returns:
        Updated source lines
    """
    for i, line in enumerate(source_lines):
        if line.strip().startswith(f"{param_name} ="):
            if is_list:
                source_lines[i] = f"{param_name} = {value}\n"
            elif isinstance(value, str):
                source_lines[i] = f"{param_name} = \"{value}\"\n"
            elif isinstance(value, bool):
                source_lines[i] = f"{param_name} = {value}\n"
            else:
                source_lines[i] = f"{param_name} = {value}\n"
            break
    return source_lines


def run(prompts_list, notebook, model_id, kernel_path=".", gpu=None, refiner_model_id=None, guidance=None, steps=None, precision="fp16", negative_prompt=None, output_dir=None, refiner_guidance=None, refiner_steps=None, refiner_precision=None, refiner_negative_prompt=None, img_size=None, model_filename=None, vae_repo_id=None, vae_filename=None, clip_l_repo_id=None, clip_l_filename=None, t5xxl_repo_id=None, t5xxl_filename=None, wait_timeout=None, retry_interval=None):
    """
    Deploy Kaggle notebook kernel, optionally overriding prompts and model.
    Uses the specified notebook; user is responsible for matching notebook to model.
    
    The HF_TOKEN is read from the Kaggle dataset 'leventecsibi/imggenhub-hf-token'
    which is automatically attached to all kernels. Use sync_hf_token() before
    deployment to ensure the token is up-to-date.
    
    FLUX GGUF model parameters:
    - model_id/model_filename: Main diffusion model (e.g., city96/FLUX.1-schnell-gguf, flux1-schnell-Q4_0.gguf)
    - vae_repo_id/vae_filename: VAE model
    - clip_l_repo_id/clip_l_filename: CLIP-L text encoder
    - t5xxl_repo_id/t5xxl_filename: T5-XXL text encoder
    
    Args:
        ...
        wait_timeout (int): Maximum wait time in minutes for GPU availability.
        retry_interval (int): Interval in seconds between retries.
    """
    
    # Load config for defaults if not provided
    if wait_timeout is None or retry_interval is None:
        config = load_kaggle_config()
        if wait_timeout is None:
            wait_timeout = config.get("deployment_timeout_minutes", 30)
        if retry_interval is None:
            retry_interval = config.get("retry_interval_seconds", 60)

    
    # Resolve notebook path relative to kernel_path
    nb_path = Path(kernel_path) / notebook
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")
    
    # Load notebook
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Detect notebook type
    is_flux_notebook = "flux" in str(notebook).lower()
    is_flux_gguf_notebook = "flux-gguf" in str(notebook).lower()
    is_flux_bf16_notebook = "flux-schnell-bf16" in str(notebook).lower()

    # For FLUX bf16 notebooks, do NOT inject dataset download
    # They download models directly from HuggingFace Hub using HF_TOKEN

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"] if isinstance(cell["source"], list) else [cell["source"]]

            # Update parameters only if provided by user (no silent overrides)
            if prompts_list:
                source = _update_param(source, "PROMPTS", prompts_list, is_list=True)
            if model_id:
                source = _update_param(source, "MODEL_ID", model_id)
            if guidance is not None:
                source = _update_param(source, "GUIDANCE", guidance)
            if steps is not None:
                source = _update_param(source, "STEPS", steps)
            if precision:
                source = _update_param(source, "PRECISION", precision)
            # Override OUTPUT_DIR to "." so notebook writes to download root, not nested "images/" folder
            # This prevents images/images/ nesting when download path already ends in "images"
            source = _update_param(source, "OUTPUT_DIR", ".")
            if img_size:
                source = _update_param(source, "IMG_SIZE", img_size)

            # FLUX GGUF model configuration parameters
            if model_id:
                source = _update_param(source, "MODEL_ID", model_id)
            if model_filename:
                source = _update_param(source, "MODEL_FILENAME", model_filename)
            if vae_repo_id:
                source = _update_param(source, "VAE_REPO_ID", vae_repo_id)
            if vae_filename:
                source = _update_param(source, "VAE_FILENAME", vae_filename)
            if clip_l_repo_id:
                source = _update_param(source, "CLIP_L_REPO_ID", clip_l_repo_id)
            if clip_l_filename:
                source = _update_param(source, "CLIP_L_FILENAME", clip_l_filename)
            if t5xxl_repo_id:
                source = _update_param(source, "T5XXL_REPO_ID", t5xxl_repo_id)
            if t5xxl_filename:
                source = _update_param(source, "T5XXL_FILENAME", t5xxl_filename)

            # For FLUX bf16 notebooks, do NOT modify model loading
            # They load models directly from HuggingFace Hub using HF_TOKEN

            # Non-Flux specific parameters (refiner, negative prompts, etc.)
            if not is_flux_notebook and not is_flux_gguf_notebook and not is_flux_bf16_notebook:
                if refiner_model_id:
                    source = _update_param(source, "REFINER_MODEL_ID", refiner_model_id)
                if negative_prompt:
                    source = _update_param(source, "NEGATIVE_PROMPT", negative_prompt)
                if refiner_guidance is not None:
                    source = _update_param(source, "REFINER_GUIDANCE", refiner_guidance)
                if refiner_steps is not None:
                    source = _update_param(source, "REFINER_STEPS", refiner_steps)
                if refiner_precision:
                    source = _update_param(source, "REFINER_PRECISION", refiner_precision)
                if refiner_negative_prompt:
                    source = _update_param(source, "REFINER_NEGATIVE_PROMPT", refiner_negative_prompt)

            cell["source"] = source

    # Save updated notebook
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
    notebook_path = Path(notebook) if not isinstance(notebook, Path) else notebook
    kernel_meta["code_file"] = notebook_path.name
    
    # HF token dataset - always required for HuggingFace downloads
    HF_TOKEN_DATASET = "leventecsibi/imggenhub-hf-token"
    
    # Dynamic dataset_sources based on notebook type
    model_source = os.environ.get("MODEL_SOURCE", "dataset").lower()
    if is_flux_gguf_notebook:
        if model_source == "dataset":
            # Attach FLUX GGUF model datasets + HF token dataset
            kernel_meta["dataset_sources"] = [
                HF_TOKEN_DATASET,
                "leventecsibi/flux1-schnell-q4-zip",
                "leventecsibi/vae-zip",
                "leventecsibi/clip-l-zip",
                "leventecsibi/t5xxl-zip",
                "leventecsibi/sd-build-zip"
            ]
            print("[INFO] Using dataset source for FLUX GGUF model - datasets attached")
        else:
            # HuggingFace source: only need HF token dataset and sd-build for executable
            kernel_meta["dataset_sources"] = [
                HF_TOKEN_DATASET,
                "leventecsibi/sd-build-zip"
            ]
            print("[INFO] Using HuggingFace source - HF token dataset attached")
    elif is_flux_bf16_notebook:
        # bf16 notebooks need HF token for model download
        kernel_meta["dataset_sources"] = [HF_TOKEN_DATASET]
        print(f"[INFO] Attached HF token dataset for {notebook_path.name}")
    else:
        # Stable diffusion notebooks now read HF token from dataset as well
        kernel_meta["dataset_sources"] = [HF_TOKEN_DATASET]
        print(f"[INFO] Attached HF token dataset for {notebook_path.name}")
    logging.info(f"Updated kernel metadata to use notebook: {notebook}")
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(kernel_meta, f, indent=2)
    
    # VERIFY that metadata was updated correctly
    with open(metadata_path, "r", encoding="utf-8") as f:
        verify_meta = json.load(f)
    
    if verify_meta.get("code_file") != notebook_path.name:
        raise RuntimeError(
            f"ERROR: Metadata update verification failed!\n"
            f"  Expected code_file: {notebook_path.name}\n"
            f"  Got code_file: {verify_meta.get('code_file')}\n"
            f"  Metadata file: {metadata_path}"
        )
    
    logging.info(f" VERIFIED: Metadata correctly updated from '{original_code_file}' to '{notebook_path.name}'")
    
    # Push via Kaggle CLI - try Poetry first, fallback to direct python
    kaggle_cmd = _get_kaggle_command()
    logging.info(f"Deploying to Kaggle with command: {' '.join(kaggle_cmd)}")
    
    start_time = time.time()
    timeout_seconds = wait_timeout * 60
    
    while True:
        try:
            result = subprocess.run(
                [*kaggle_cmd, "kernels", "push", "-p", str(kernel_path)],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )

            if result.stdout:
                logging.info(f"Kaggle push output: {result.stdout}")
                # Check for errors in stdout (Kaggle CLI sometimes returns 0 even on errors)
                if "error" in result.stdout.lower() or "maximum" in result.stdout.lower():
                    raise RuntimeError(f"Kaggle push failed: {result.stdout}")
            if result.stderr:
                logging.debug(f"Kaggle push stderr: {result.stderr}")
            
            # If we reached here, push was successful
            break

        except (subprocess.CalledProcessError, RuntimeError) as exc:
            error_msg = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                error_msg = f"{exc.stdout}\n{exc.stderr}"
            
            error_lower = error_msg.lower()
            
            # Detect GPU limit or session limit errors
            is_gpu_limit = any(msg in error_lower for msg in ["maximum", "session", "limit", "gpu", "409", "conflict"])
            
            if is_gpu_limit:
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    logging.error(f"Kaggle GPU deployment limit reached and timeout of {wait_timeout} minutes exceeded.")
                    raise RuntimeError(f"Kaggle GPU deployment timeout: {error_msg}")
                
                remaining = timeout_seconds - elapsed
                print("\n" + "!"*80)
                print("WARNING: KAGGLE GPU DEPLOYMENT LIMIT REACHED!")
                print("!"*80)
                print(f"No GPU slots available. Waiting {retry_interval}s before retry...")
                print(f"Elapsed: {int(elapsed/60)}m | Timeout: {wait_timeout}m | Remaining: {int(remaining/60)}m")
                print("!"*80 + "\n")
                
                time.sleep(retry_interval)
                continue
            else:
                # Non-retryable error
                logging.error(f"Kaggle push FAILED with non-retryable error")
                if isinstance(exc, subprocess.CalledProcessError):
                    logging.error(f"Exit code: {exc.returncode}")
                    logging.error(f"stdout: {exc.stdout}")
                    logging.error(f"stderr: {exc.stderr}")
                else:
                    logging.error(f"Error: {exc}")
                raise


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
