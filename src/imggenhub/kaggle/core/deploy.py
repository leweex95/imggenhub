import os
import re
import json
from pathlib import Path
import subprocess
import shutil
import logging


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


def run(prompts_list, notebook, model_id, kernel_path=".", gpu=None, refiner_model_id=None, guidance=None, steps=None, precision="fp16", negative_prompt=None, output_dir=None, two_stage_refiner=False, refiner_guidance=None, refiner_steps=None, refiner_precision=None, refiner_negative_prompt=None, hf_token=None, img_size=None):
    """
    Deploy Kaggle notebook kernel, optionally overriding prompts and model.
    Uses the specified notebook; user is responsible for matching notebook to model.
    
    NOTE: If hf_token is provided locally, you must also add it as a Kaggle Secret
    named 'HF_TOKEN' via https://www.kaggle.com/settings so the kernel can access it.
    The notebook reads from os.getenv("HF_TOKEN"), NOT from hardcoded values.
    """
    
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

    # For FLUX notebooks, inject dataset download code
    if is_flux_notebook and not is_flux_gguf_notebook:
        # Add dataset download cell after the first cell (imports/installs)
        dataset_download_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Download FLUX text encoder parts from Kaggle dataset\n",
                "import kagglehub\n",
                "print(\"Downloading FLUX text encoder model parts...\")\n",
                "dataset_path = kagglehub.dataset_download(\"leventecsibi/flux-1-schnell-text-encoder-parts\")\n",
                "print(f\"Dataset downloaded to: {dataset_path}\")\n",
                "\n",
                "# Set up model paths for local loading\n",
                "import os\n",
                "text_encoder_2_path = os.path.join(dataset_path, \"model_part_1.safetensors\")\n",
                "text_encoder_2_path_2 = os.path.join(dataset_path, \"model_part_2.safetensors\")\n",
                "print(f\"Text encoder parts: {text_encoder_2_path}, {text_encoder_2_path_2}\")\n"
            ]
        }
        nb["cells"].insert(1, dataset_download_cell)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"] if isinstance(cell["source"], list) else [cell["source"]]

            # Inject HF_TOKEN directly into the notebook if provided
            if hf_token:
                for i, line in enumerate(source):
                    if 'HF_TOKEN = os.getenv("HF_TOKEN"' in line:
                        source[i] = f'HF_TOKEN = "{hf_token}"\n'
                        logging.info("Injected HF_TOKEN directly into notebook")
                        break

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
                source = _update_param(source, "IMG_SIZE", img_size)            # For FLUX notebooks, modify model loading to use local text encoder
            if is_flux_notebook and not is_flux_gguf_notebook and "FluxPipeline.from_pretrained" in "".join(source):
                # Replace the model loading with custom loading using local text encoder
                source = [line.replace(
                    'pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch_dtype)',
                    'pipe = FluxPipeline.from_pretrained(\n'
                    '    MODEL_ID, \n'
                    '    torch_dtype=torch_dtype,\n'
                    '    text_encoder_2_kwargs={"variant": "fp16"},\n'
                    '    vae_kwargs={"variant": "fp16"}\n'
                    ')\n'
                    '# Load text encoder from local files\n'
                    'from safetensors.torch import load_file\n'
                    'print("Loading text encoder from local files...")\n'
                    'text_encoder_state = {}\n'
                    'text_encoder_state.update(load_file(text_encoder_2_path))\n'
                    'text_encoder_state.update(load_file(text_encoder_2_path_2))\n'
                    'pipe.text_encoder_2.load_state_dict(text_encoder_state, strict=False)\n'
                    'print("Text encoder loaded from local files")'
                ) for line in source]

            # Non-Flux specific parameters (refiner, negative prompts, etc.)
            if not is_flux_notebook and not is_flux_gguf_notebook:
                if refiner_model_id:
                    source = _update_param(source, "REFINER_MODEL_ID", refiner_model_id)
                if negative_prompt:
                    source = _update_param(source, "NEGATIVE_PROMPT", negative_prompt)
                if two_stage_refiner:
                    source = _update_param(source, "TWO_STAGE_REFINER", two_stage_refiner)
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

    # Warn if HF_TOKEN is required but user hasn't set it as Kaggle Secret
    model_source = os.environ.get("MODEL_SOURCE", "dataset").lower()
    if hf_token and model_source != "dataset":
        logging.warning("="*80)
        logging.warning("HF_TOKEN provided locally but notebook uses os.getenv('HF_TOKEN')")
        logging.warning("Ensure you've added 'HF_TOKEN' as a Kaggle Secret at:")
        logging.warning("https://www.kaggle.com/settings")
        logging.warning("Otherwise the kernel will fail to download models from HuggingFace")
        logging.warning("="*80)

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
    
    # Dynamic dataset_sources: only attach datasets if MODEL_SOURCE is "dataset"
    # This saves GPU startup time when using HuggingFace downloads
    model_source = os.environ.get("MODEL_SOURCE", "dataset").lower()
    if is_flux_gguf_notebook:
        if model_source == "dataset":
            # Attach FLUX GGUF datasets
            kernel_meta["dataset_sources"] = [
                "leventecsibi/flux1-schnell-q4-zip",
                "leventecsibi/vae-zip",
                "leventecsibi/clip-l-zip",
                "leventecsibi/t5xxl-zip",
                "leventecsibi/sd-build-zip"
            ]
            print("[INFO] Using dataset source for FLUX GGUF model - datasets attached")
        else:
            # Empty dataset sources when using HuggingFace
            kernel_meta["dataset_sources"] = []
            print("="*80)
            print("[WARNING] Using HuggingFace source for FLUX GGUF model")
            print("[WARNING] No datasets attached - models will be downloaded from HuggingFace Hub")
            print("[WARNING] This may increase kernel startup time")
            print("="*80)
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
