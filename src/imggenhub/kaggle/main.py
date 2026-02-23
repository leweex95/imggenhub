# main.py
import argparse
import subprocess
import logging
import os
import tempfile
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict
from kaggle_connector import JobManager, SelectiveDownloader, DatasetManager
from imggenhub.kaggle.core.parallel_deploy import run_parallel_pipeline, should_use_parallel
from imggenhub.kaggle.utils.prompts import resolve_prompts
from imggenhub.kaggle.utils.cli import log_cli_command, setup_output_directory
from imggenhub.kaggle.utils.filesystem import ensure_output_directory
from imggenhub.kaggle.utils.arg_validator import validate_args
from imggenhub.kaggle.utils.config_loader import load_kaggle_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _notebook_to_script(nb_path: Path) -> str:
    """Extract code cells from a notebook and return a Python script string.

    The Kaggle imggenhub-generator kernel is "script" type, so we must push
    a plain .py file. Pushing .ipynb directly causes Kaggle to only run Cell 0.
    """
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    code_blocks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        code = "".join(src) if isinstance(src, list) else src
        if code.strip():
            code_blocks.append(code)
    return "\n\n".join(code_blocks) + "\n"


def run_pipeline(dest_path, prompts_file, notebook, kernel_path, gpu=False, model_id=None, refiner_model_id=None, prompt=None, prompts=None, guidance=None, steps=None, precision=None, negative_prompt=None, refiner_guidance=None, refiner_steps=None, refiner_precision=None, refiner_negative_prompt=None, img_size=None, model_filename=None, vae_repo_id=None, vae_filename=None, clip_l_repo_id=None, clip_l_filename=None, t5xxl_repo_id=None, t5xxl_filename=None, wait_timeout=None, accelerator=None, lora_repo_id=None, lora_filename=None, lora_scale=0.8):
    """Run Kaggle image generation pipeline: sync HF token -> deploy -> poll -> download"""
    print("Initializing run_pipeline in main.py...")
    cwd = Path(__file__).parent

    # Load Kaggle config
    config = load_kaggle_config()
    if wait_timeout is None:
        wait_timeout = config.get("deployment_timeout_minutes", 30)
    
    retry_interval = config.get("retry_interval_seconds", 60)

    # Sync HF token to Kaggle dataset before deployment
    logging.info("Syncing HF token to Kaggle dataset...")
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set")
            
        from kaggle import api
        username = api.config_values.get("username")
        if not username:
            raise ValueError("Kaggle username not found in config")
        hf_token_dataset = f"{username}/imggenhub-hf-token"
            
        with tempfile.TemporaryDirectory() as tmp_dir:
            token_file = Path(tmp_dir) / "hf_token.json"
            with open(token_file, "w") as f:
                json.dump({"HF_TOKEN": hf_token}, f)
            
            dm = DatasetManager()
            # Use same dataset ID as before
            updated = dm.sync_dataset(hf_token_dataset, [token_file], version_notes="Sync from imggenhub core refactor")
            
        if updated:
            logging.info(f"HF token dataset {hf_token_dataset} updated successfully")
        else:
            logging.info(f"HF token dataset {hf_token_dataset} is already up-to-date")
    except Exception as e:
        raise RuntimeError(f"Failed to sync HF token: {e}") from e

    # Resolve paths
    if prompts_file:
        prompts_file = Path(prompts_file)
        if not prompts_file.is_absolute():
            prompts_file = cwd / prompts_file

    kernel_path = Path(kernel_path)
    if not kernel_path.is_absolute():
        kernel_path = cwd / kernel_path  # Respect user-provided kernel path

    notebook = Path(notebook)
    if not notebook.is_absolute():
        # Try local notebooks folder first, then kernel_path
        local_notebook = cwd / "notebooks" / notebook.name
        if local_notebook.exists():
            notebook = local_notebook
        else:
            notebook = kernel_path / notebook.name  # Fallback to kernel path if not in notebooks

    prompts_list = resolve_prompts(prompts_file, prompt)

    logging.debug(f"Resolved paths:\n prompts_file={prompts_file}\n notebook={notebook}\n kernel_path={kernel_path}\n dest={dest_path}")

    # Resolve username and kernel ID
    from kaggle import api
    username = api.config_values.get("username")
    base_kernel_id = f"{username}/imggenhub-generator"

    # Check if parallel deployment should be used (prompts > 4)
    if should_use_parallel(prompts_list):
        logging.info("="*80)
        logging.info(f"PARALLEL MODE: {len(prompts_list)} prompts detected (threshold: 4)")
        logging.info("Splitting prompts across 2 Kaggle kernels for parallel execution")
        logging.info("="*80)
        
        # Build deploy kwargs for parallel pipeline
        deploy_kwargs = {
            "username": username,
            "base_kernel_id": base_kernel_id,
            "gpu": gpu,
            "refiner_model_id": refiner_model_id,
            "guidance": guidance,
            "steps": steps,
            "precision": precision,
            "negative_prompt": negative_prompt,
            "output_dir": dest_path.name,
            "refiner_guidance": refiner_guidance,
            "refiner_steps": refiner_steps,
            "refiner_precision": refiner_precision,
            "refiner_negative_prompt": refiner_negative_prompt,
            "img_size": img_size,
            "model_filename": model_filename,
            "vae_repo_id": vae_repo_id,
            "vae_filename": vae_filename,
            "clip_l_repo_id": clip_l_repo_id,
            "clip_l_filename": clip_l_filename,
            "t5xxl_repo_id": t5xxl_repo_id,
            "t5xxl_filename": t5xxl_filename,
            "lora_repo_id": lora_repo_id,
            "lora_filename": lora_filename,
            "lora_scale": lora_scale,
        }
        
        run_parallel_pipeline(
            dest_path=dest_path,
            prompts_list=prompts_list,
            notebook=notebook,
            kernel_path=kernel_path,
            wait_timeout=wait_timeout,
            retry_interval=retry_interval,
            polling_interval=config.get("polling_interval_seconds", 60),
            accelerator=accelerator,
            **deploy_kwargs
        )
        
        logging.info(f"Pipeline completed! Output saved to: {dest_path}")
        logging.info("Check remaining GPU quota: https://www.kaggle.com/settings#quotas")
        return

    # Sequential deployment for 4 or fewer prompts
    logging.info("Deploying kernel using Kaggle Connector...")

    # Helper: get lastRunTime for this kernel from Kaggle kernels list
    kernel_name = base_kernel_id.split("/")[-1]

    def _get_last_run_time() -> str:
        from kaggle_connector.utils.cli import run_kaggle_cli
        r = run_kaggle_cli(["kernels", "list", "--mine", "--csv", "--search", kernel_name])
        for line in r.stdout.splitlines():
            if f"{base_kernel_id}," in line:
                parts = line.split(",")
                return parts[3].strip() if len(parts) > 3 else ""
        return ""

    # Record lastRunTime BEFORE push to detect when new run actually starts
    pre_push_time = _get_last_run_time()
    logging.info(f"Kernel last run time before push: {pre_push_time!r}")

    # 1. Prepare Notebook (parameters injection) and convert to Python script
    # The Kaggle kernel is "script" type and expects a .py file named
    # "imggenhub-generator.py". We inject params into the notebook, then
    # extract all code cells to produce the final Python script for pushing.
    with tempfile.TemporaryDirectory() as tmp_deploy_dir:
        tmp_dir_path = Path(tmp_deploy_dir)
        tmp_nb_path = tmp_dir_path / Path(notebook).name
        shutil.copy2(notebook, tmp_nb_path)

        # Prepare parameters
        params = {
            "PROMPTS": prompts_list,
            "MODEL_ID": model_id,
            "GUIDANCE": guidance,
            "STEPS": steps,
            "PRECISION": precision,
            "OUTPUT_DIR": ".",
            "IMG_SIZE": img_size,
            "KERNEL_ID": base_kernel_id
        }

        # LoRA params for bf16 and dual-T4 notebooks
        notebook_str = str(notebook).lower()
        if "flux-schnell-bf16" in notebook_str or "flux-dual-t4" in notebook_str:
            if lora_repo_id: params["LORA_REPO_ID"] = lora_repo_id
            if lora_filename: params["LORA_FILENAME"] = lora_filename
            params["LORA_SCALE"] = lora_scale

        # Flux GGUF specific
        if "flux-gguf" in notebook_str:
            if model_filename: params["MODEL_FILENAME"] = model_filename
            if vae_repo_id: params["VAE_REPO_ID"] = vae_repo_id
            if vae_filename: params["VAE_FILENAME"] = vae_filename
            if clip_l_repo_id: params["CLIP_L_REPO_ID"] = clip_l_repo_id
            if clip_l_filename: params["CLIP_L_FILENAME"] = clip_l_filename
            if t5xxl_repo_id: params["T5XXL_REPO_ID"] = t5xxl_repo_id
            if t5xxl_filename: params["T5XXL_FILENAME"] = t5xxl_filename

        manager = JobManager()
        manager.edit_notebook_params(str(tmp_nb_path), params)

        # Convert the params-injected notebook to a Python script. The Kaggle
        # kernel is "script" type so Kaggle expects a plain .py file. Pushing
        # an .ipynb would cause Kaggle to convert it incorrectly (only Cell 0).
        script_content = _notebook_to_script(tmp_nb_path)
        script_path = tmp_dir_path / "imggenhub-generator.py"
        script_path.write_text(script_content, encoding="utf-8")

        dataset_sources = [f"{username}/imggenhub-hf-token"]
        if "flux-gguf" in notebook_str:
             dataset_sources.extend([
                f"{username}/flux1-schnell-q4-zip",
                f"{username}/vae-zip",
                f"{username}/clip-l-zip",
                f"{username}/t5xxl-zip",
                f"{username}/sd-build-zip"
             ])

        # Always "script" — imggenhub-generator kernel was created as script type.
        kernel_type = "script"
        manager.create_metadata(
            str(tmp_dir_path),
            kernel_id=base_kernel_id,
            code_file="imggenhub-generator.py",
            kernel_type=kernel_type,
            enable_gpu=gpu,
            dataset_sources=dataset_sources,
            accelerator=accelerator
        )

        # 3. Deploy
        manager.deploy(str(tmp_dir_path), wait=True)

    logging.debug("Deploy step completed via connector")

    # Step 2: Poll status
    # After push, Kaggle keeps showing 'complete' (old run) until the new run starts.
    # We detect a new run by watching kernel lastRunTime from 'kernels list'.
    # Once lastRunTime advances past pre_push_time, the new run has started.
    logging.info("Waiting for Kaggle to start the new kernel run...")
    manager = JobManager(base_kernel_id)
    max_registration_wait = wait_timeout * 60  # honour user-configured timeout

    wait_start = time.time()
    while time.time() - wait_start < max_registration_wait:
        current_time = _get_last_run_time()
        if current_time and current_time != pre_push_time:
            logging.info(f"New kernel run detected (lastRunTime: {current_time})")
            break
        reg_status = manager.get_status()
        if "complete" not in reg_status and "unknown" not in reg_status:
            logging.info(f"Kernel transitioned to status: {reg_status}")
            break
        elapsed = int(time.time() - wait_start)
        logging.info(f"Waiting for new run to start... ({elapsed}s elapsed, lastRunTime={current_time!r})")
        time.sleep(30)
    else:
        logging.warning(f"Timed out ({int(max_registration_wait)}s) waiting for new run; proceeding.")

    logging.info("Polling kernel status using Kaggle Connector...")
    status = manager.poll_until_complete()
    logging.debug("Poll status completed")

    if "error" in status.lower():
        log_path = dest_path / "stable-diffusion-batch-generator.log"
        logging.error(f"Kernel failed. See log: {log_path}")
        raise RuntimeError(f"Kaggle kernel {base_kernel_id} failed during image generation. Aborting pipeline.")

    # Step 3: Download output (using selective downloader to get only images)
    logging.info("Downloading output artifacts (images only)...")
    downloader = SelectiveDownloader(base_kernel_id)
    downloader.download_images(
        dest_path=str(dest_path),
        expected_image_count=len(prompts_list),
        stable_count_patience=4,
        polling_interval=config.get("polling_interval_seconds", 60)
    )
    logging.debug("Download completed")

    # Validate image count for sequential deployment
    if not should_use_parallel(prompts_list):
        image_extensions = {".png", ".jpg", ".jpeg"}
        actual_images = len([f for f in dest_path.rglob("*") if f.is_file() and f.suffix.lower() in image_extensions and f.name.startswith("gen_")])
        expected_images = len(prompts_list)
        if actual_images != expected_images:
            logging.error(f"Incomplete image generation: expected {expected_images} images but got {actual_images}")
            raise RuntimeError(
                f"Image generation incomplete: expected {expected_images} images "
                f"but only got {actual_images}. Some prompts failed to generate images."
            )

    logging.info(f"Pipeline completed! Output saved to: {dest_path}")
    logging.info("Check remaining GPU quota: https://www.kaggle.com/settings#quotas")


def main():
    """Main entry point - focused on argument parsing and orchestration"""
    # First parser for early args only (no help to avoid conflicts)
    early_parser = argparse.ArgumentParser(add_help=False)
    early_parser.add_argument("--dest", type=str, default=None)
    early_parser.add_argument("--output_base_dir", type=str, default=None)
    
    # Parse only known args to get output_base_dir and dest early
    early_args, remaining = early_parser.parse_known_args()
    
    # Set up output directory and log CLI command early
    dest_path = setup_output_directory(base_name=early_args.dest, base_dir=early_args.output_base_dir)
    log_cli_command(dest_path)

    # Parse full command line arguments
    parser = argparse.ArgumentParser(description="Kaggle image generation pipeline")
    parser.add_argument("--prompts_file", type=str, default=None, help="JSON file containing list of prompts")
    parser.add_argument("--notebook", type=str, default=None, help="Notebook to use (auto-detects based on model if not specified)")
    parser.add_argument("--kernel_path", type=str, default=str(Path(__file__).parent / "notebooks"))
    parser.add_argument("--gpu", action="store_true", help="Enable GPU for the kernel")
    parser.add_argument("--dest", type=str, default=None, help="Optional prefix for output folder (default: timestamp only)")
    parser.add_argument("--output_base_dir", type=str, default=None, help="Base directory for output runs (default: current working directory)")
    parser.add_argument("--model_id", type=str, default=None, help="Model ID (HuggingFace repo ID) for all models. For quantized models, use GGUF repo ID.")
    parser.add_argument("--refiner_model_id", type=str, default=None, help="Refiner model ID (HuggingFace repo ID) for SDXL refiner.")
    parser.add_argument("--prompt", action='append', default=None, help="Prompt(s) for generation. Can be used multiple times for multiple prompts. For many prompts, use --prompts_file.")
    parser.add_argument("--guidance", type=float, required=True, help="Guidance scale (7-12 recommended for photorealism)")
    parser.add_argument("--steps", type=int, required=True, help="Number of inference steps (50-100 for better quality)")
    parser.add_argument("--precision", type=str, required=True, choices=["fp32", "fp16", "bf16", "int8", "int4", "q4", "q5", "q6", "q8"],
                        help="Precision level (REQUIRED): fp32 (highest quality), fp16 (balanced), bf16 (recommended for Flux), q4/q5/q6/q8 (GGUF quantized), int8 (faster), int4 (fastest)")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Custom negative prompt for better quality control")
    parser.add_argument("--refiner_guidance", type=float, default=None, help="STABLE DIFFUSION ONLY: Guidance scale for refiner (REQUIRED when using --refiner_model_id). Ignored for Flux models.")
    parser.add_argument("--refiner_steps", type=int, default=None, help="STABLE DIFFUSION ONLY: Number of inference steps for refiner (REQUIRED when using --refiner_model_id). Ignored for Flux models.")
    parser.add_argument("--refiner_precision", type=str, default=None, choices=["fp32", "fp16", "bf16", "int8", "int4"],
                        help="STABLE DIFFUSION ONLY: Precision level for refiner (REQUIRED when using --refiner_model_id, or inherits from --precision). Ignored for Flux models.")
    parser.add_argument("--refiner_negative_prompt", type=str, default=None, help="STABLE DIFFUSION ONLY: Custom negative prompt for refiner (defaults to same as --negative_prompt). Ignored for Flux models.")
    parser.add_argument("--img_width", type=int, default=None, help="Image width (defaults: 1024 for stable diffusion, 512 for flux gguf)")
    parser.add_argument("--img_height", type=int, default=None, help="Image height (defaults: 1024 for stable diffusion, 512 for flux gguf)")
    parser.add_argument("--wait_timeout", type=int, default=None, help="Maximum wait time in minutes for GPU availability (overrides YAML config)")
    parser.add_argument("--accelerator", type=str, default=None, choices=["nvidia-t4-x2", "nvidia-p100"], help="Kaggle accelerator type (e.g., nvidia-t4-x2, nvidia-p100)")

    # LoRA configuration for photorealism
    parser.add_argument("--lora_repo_id", type=str, default=None, help="HuggingFace repo ID for LoRA weights (e.g. Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur for photorealism)")
    parser.add_argument("--lora_filename", type=str, default=None, help="LoRA weight filename. If not provided, the default file in the repo is used.")
    parser.add_argument("--lora_scale", type=float, default=0.8, help="LoRA adapter scale / strength (default: 0.8)")
    parser.add_argument("--enhance_photorealism", action="store_true", help="Append photography realism terms to all prompts (grain, skin texture, shallow depth of field, etc.)")

    # FLUX GGUF model configuration (quantized models only)
    parser.add_argument("--model_filename", type=str, default=None, help="Model filename for quantized GGUF models (e.g., flux1-schnell-Q4_0.gguf)")
    parser.add_argument("--vae_repo_id", type=str, default=None, help="FLUX GGUF ONLY: HuggingFace repo ID for VAE model (auto-resolved if not provided)")
    parser.add_argument("--vae_filename", type=str, default=None, help="FLUX GGUF ONLY: VAE model filename (auto-resolved if not provided)")
    parser.add_argument("--clip_l_repo_id", type=str, default=None, help="FLUX GGUF ONLY: HuggingFace repo ID for CLIP-L text encoder (auto-resolved if not provided)")
    parser.add_argument("--clip_l_filename", type=str, default=None, help="FLUX GGUF ONLY: CLIP-L model filename (auto-resolved if not provided)")
    parser.add_argument("--t5xxl_repo_id", type=str, default=None, help="FLUX GGUF ONLY: HuggingFace repo ID for T5-XXL text encoder (auto-resolved if not provided)")
    parser.add_argument("--t5xxl_filename", type=str, default=None, help="FLUX GGUF ONLY: T5-XXL model filename (auto-resolved if not provided)")

    args = parser.parse_args()

    # Validate arguments strictly before any auto-detection or notebook selection
    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Check for missing image dimensions before notebook auto-detection
    if args.img_width is None or args.img_height is None:
        print("\n" + "="*80)
        print("VALIDATION ERROR: MISSING IMAGE DIMENSIONS!")
        print("="*80)
        print("Both --img_width and --img_height are REQUIRED.")
        print("Please specify explicit image dimensions.")
        print("Examples:")
        print("  - Stable Diffusion XL: --img_width 1024 --img_height 1024")
        print("  - FLUX GGUF Q4: --img_width 512 --img_height 512")
        print("="*80 + "\n")
        return

    # Auto-detect notebook based on model type if not specified
    if args.notebook is None:
        # Check if FLUX GGUF based on explicit parameters or model_id
        is_gguf = (args.model_filename or (args.model_id and _is_flux_gguf_model(args.model_id)))
        is_bf16 = args.model_id and _is_flux_bf16_model(args.model_id)
        is_dual_t4 = args.accelerator == "nvidia-t4-x2"
        if is_gguf:
            args.notebook = str(Path(__file__).parent / "notebooks/kaggle-flux-gguf.ipynb")
            print(f"Auto-detected FLUX GGUF model, using notebook: {args.notebook}")
            # Enforce GPU for FLUX GGUF models
            if not args.gpu:
                print("\n" + "="*80)
                print("WARNING: FLUX GGUF Q4 MODELS REQUIRE GPU!")
                print("="*80)
                print("You did NOT specify --gpu flag.")
                print("FLUX GGUF Q4 models cannot run on CPU (too slow and memory-intensive).")
                print("Automatically enabling GPU mode...")
                print("="*80 + "\n")
                args.gpu = True
        elif is_bf16 and is_dual_t4:
            # T4×2: use device_map='balanced' notebook — no CPU offload, significantly faster
            # Also supports larger models (FLUX.1-dev) across 2x16GB=32GB VRAM
            args.notebook = str(Path(__file__).parent / "notebooks/kaggle-flux-dual-t4.ipynb")
            print(f"Auto-detected FLUX bf16 + T4×2 accelerator, using optimized dual-GPU notebook: {args.notebook}")
            print("  device_map='balanced' distributes model across both T4 GPUs (32GB total VRAM)")
            print("  No sequential CPU offload → significantly faster inference")
            if not args.gpu:
                args.gpu = True
        elif is_bf16:
            args.notebook = str(Path(__file__).parent / "notebooks/kaggle-flux-schnell-bf16.ipynb")
            print(f"Auto-detected FLUX bf16 model, using notebook: {args.notebook}")
            # Enforce GPU for FLUX bf16 models
            if not args.gpu:
                print("\n" + "="*80)
                print("WARNING: FLUX bf16 MODELS REQUIRE GPU!")
                print("="*80)
                print("You did NOT specify --gpu flag.")
                print("FLUX bf16 models cannot run on CPU (too slow and memory-intensive).")
                print("Automatically enabling GPU mode...")
                print("="*80 + "\n")
                args.gpu = True
        else:
            args.notebook = str(Path(__file__).parent / "notebooks/kaggle-stable-diffusion.ipynb")
            print(f"Using default notebook: {args.notebook}")
    
    # Validate FLUX model dimensions must be multiples of 16
    if args.model_id and _is_flux_gguf_model(args.model_id):
        # Warn if guidance > 1.0 for FLUX GGUF models
        if args.guidance > 1.0:
            print("\n" + "="*80)
            print("WARNING: HIGH GUIDANCE FOR FLUX GGUF MODEL!")
            print("="*80)
            print(f"You specified --guidance {args.guidance}")
            print(f"For quantized FLUX GGUF models (Q4/Q5/Q6/Q8), the recommended guidance is 0.5-1.0")
            print(f"Higher guidance values may lead to:")
            print(f"  - Over-saturated colors")
            print(f"  - Artifacts and distortions")
            print(f"  - Loss of photorealism")
            print(f"Proceeding anyway, but consider using --guidance 0.5 to 1.0 for better results.")
            print("="*80 + "\n")
        
        if args.img_width % 16 != 0 or args.img_height % 16 != 0:
            print("\n" + "="*80)
            print("VALIDATION ERROR: INVALID FLUX IMAGE DIMENSIONS!")
            print("="*80)
            print(f"FLUX models require dimensions to be multiples of 16.")
            print(f"You provided: {args.img_width}x{args.img_height}")
            print(f"Valid dimensions close to your request:")
            # Suggest nearest valid dimensions
            valid_width = (args.img_width // 16) * 16
            valid_height = (args.img_height // 16) * 16
            print(f"  - {valid_width}x{valid_height}")
            print("="*80 + "\n")
            return

    # Warn if guidance > 2.0 for FLUX bf16 models (photorealism best practice)
    if args.model_id and _is_flux_bf16_model(args.model_id) and args.guidance > 2.0:
        print("\n" + "="*80)
        print("WARNING: HIGH GUIDANCE FOR FLUX BF16 MODEL!")
        print("="*80)
        print(f"You specified --guidance {args.guidance}")
        print("For FLUX bf16 models, guidance 1.0-2.0 produces the most photorealistic results.")
        print("Higher guidance often leads to over-saturated, synthetic-looking images.")
        print("Proceeding anyway, but consider using --guidance 1.0 to 2.0 for better realism.")
        print("="*80 + "\n")

    img_size = (args.img_height, args.img_width)

    # Precision is now required; no auto-detection
    logging.info(f"Using explicit precision: {args.precision}")

    # Warn if refiner flags are used with Flux models (they are ignored)
    is_flux_model = args.model_id and (_is_flux_gguf_model(args.model_id) or _is_flux_bf16_model(args.model_id))
    is_flux_gguf = args.model_filename or (args.model_id and _is_flux_gguf_model(args.model_id))
    is_flux_bf16 = args.model_id and _is_flux_bf16_model(args.model_id)
    
    if (is_flux_model or is_flux_gguf or is_flux_bf16) and (args.refiner_model_id or args.refiner_guidance or args.refiner_steps or args.refiner_precision or args.refiner_negative_prompt):
        print("\n" + "="*80)
        print("WARNING: REFINER FLAGS IGNORED FOR FLUX MODELS")
        print("="*80)
        print("You specified refiner-related flags with a Flux model.")
        print("The following flags will be ignored:")
        if args.refiner_model_id:
            print(f"  - --refiner_model_id: {args.refiner_model_id}")
        if args.refiner_guidance:
            print(f"  - --refiner_guidance: {args.refiner_guidance}")
        if args.refiner_steps:
            print(f"  - --refiner_steps: {args.refiner_steps}")
        if args.refiner_precision:
            print(f"  - --refiner_precision: {args.refiner_precision}")
        if args.refiner_negative_prompt:
            print(f"  - --refiner_negative_prompt: {args.refiner_negative_prompt}")
        print("Proceeding with Flux model generation...")
        print("="*80 + "\n")

    # Validate arguments including precision availability
    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Apply photorealism prompt enhancement if requested
    if getattr(args, "enhance_photorealism", False) and args.prompt:
        realism_suffix = (
            "shot on 35mm film, Kodak Portra 400, shallow depth of field, "
            "natural lighting, skin texture, pores, subtle grain, imperfect skin, "
            "photorealistic, hyperrealistic"
        )
        args.prompt = [f"{p}, {realism_suffix}" for p in args.prompt]
        print("Photorealism enhancement applied to all prompts.")

    run_pipeline(
        dest_path=dest_path,
        prompts_file=args.prompts_file,
        notebook=args.notebook,
        kernel_path=args.kernel_path,
        gpu=args.gpu,
        model_id=args.model_id,
        refiner_model_id=args.refiner_model_id,
        prompt=args.prompt,
        guidance=args.guidance,
        steps=args.steps,
        precision=args.precision,
        negative_prompt=args.negative_prompt,
        refiner_guidance=args.refiner_guidance,
        refiner_steps=args.refiner_steps,
        refiner_precision=args.refiner_precision,
        refiner_negative_prompt=args.refiner_negative_prompt,
        img_size=img_size,
        model_filename=args.model_filename,
        vae_repo_id=args.vae_repo_id,
        vae_filename=args.vae_filename,
        clip_l_repo_id=args.clip_l_repo_id,
        clip_l_filename=args.clip_l_filename,
        t5xxl_repo_id=args.t5xxl_repo_id,
        t5xxl_filename=args.t5xxl_filename,
        wait_timeout=args.wait_timeout,
        accelerator=args.accelerator,
        lora_repo_id=args.lora_repo_id,
        lora_filename=args.lora_filename,
        lora_scale=args.lora_scale,
    )


def _is_kaggle_model(model_id: str) -> bool:
    """
    Check if a model ID refers to a Kaggle model rather than a HuggingFace model.
    Kaggle models typically have usernames that are not standard HF organizations.
    """
    if '/' not in model_id:
        return False
    
    owner = model_id.split('/')[0].lower()
    
    # Known HuggingFace organizations that host models
    hf_orgs = {
        'stabilityai', 'black-forest-labs', 'runwayml', 'compvis', 'openai',
        'google', 'microsoft', 'facebook', 'huggingface', 'meta', 'anthropic',
        'eleutherai', 'bigscience', 'bigcode', 'salesforce', 'amazon', 'nvidia',
        'intel', 'apple', 'tencent', 'baidu', 'alibaba', 'bytedance'
    }
    
    # If owner is not a known HF org, assume it's a Kaggle model
    return owner not in hf_orgs


def _is_flux_gguf_model(model_id: str) -> bool:
    """
    Check if a model ID refers to a FLUX GGUF model.
    These are quantized FLUX models in GGUF format.
    """
    if not model_id:
        return False
    model_lower = model_id.lower()
    return ('flux' in model_lower and ('gguf' in model_lower or 'q4' in model_lower or 'q8' in model_lower))


def _is_flux_bf16_model(model_id: str) -> bool:
    """
    Check if a model ID refers to a FLUX bf16 model.
    These are black-forest-labs/FLUX.1-schnell or similar full-precision models.
    """
    if not model_id:
        return False
    model_lower = model_id.lower()
    return ('flux' in model_lower and 'black-forest-labs' in model_lower) or model_id == "black-forest-labs/FLUX.1-schnell"


if __name__ == "__main__":
    main()
