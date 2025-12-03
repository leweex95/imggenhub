# main.py
import argparse
import subprocess
import logging
from pathlib import Path
from imggenhub.kaggle.core import deploy, download
from imggenhub.kaggle.core import download_selective
from imggenhub.kaggle.core.parallel_deploy import run_parallel_pipeline, should_use_parallel
from imggenhub.kaggle.secrets import sync_hf_token
from imggenhub.kaggle.utils import poll_status
from imggenhub.kaggle.utils.prompts import resolve_prompts
from imggenhub.kaggle.utils.cli import log_cli_command, setup_output_directory
from imggenhub.kaggle.utils.filesystem import ensure_output_directory
from imggenhub.kaggle.utils.arg_validator import validate_args

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_pipeline(dest_path, prompts_file, notebook, kernel_path, gpu=False, model_name=None, refiner_model_name=None, prompt=None, prompts=None, guidance=None, steps=None, precision=None, negative_prompt=None, two_stage_refiner=False, refiner_guidance=None, refiner_steps=None, refiner_precision=None, refiner_negative_prompt=None, img_size=None, diffusion_repo_id=None, diffusion_filename=None, vae_repo_id=None, vae_filename=None, clip_l_repo_id=None, clip_l_filename=None, t5xxl_repo_id=None, t5xxl_filename=None):
    """Run Kaggle image generation pipeline: sync HF token -> deploy -> poll -> download"""
    print("Initializing run_pipeline in main.py...")
    cwd = Path(__file__).parent

    # Sync HF token to Kaggle dataset before deployment
    logging.info("Syncing HF token to Kaggle dataset...")
    try:
        updated = sync_hf_token()
        if updated:
            logging.info("HF token dataset updated successfully")
        else:
            logging.info("HF token dataset is already up-to-date")
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
        notebook = kernel_path / notebook.name  # Resolve notebook relative to kernel path

    prompts_list = resolve_prompts(prompts_file, prompt, prompts)

    logging.debug(f"Resolved paths:\n prompts_file={prompts_file}\n notebook={notebook}\n kernel_path={kernel_path}\n dest={dest_path}")

    # Check if parallel deployment should be used (prompts > 4)
    if should_use_parallel(prompts_list):
        logging.info("="*80)
        logging.info(f"PARALLEL MODE: {len(prompts_list)} prompts detected (threshold: 4)")
        logging.info("Splitting prompts across 2 Kaggle kernels for parallel execution")
        logging.info("="*80)
        
        # Build deploy kwargs for parallel pipeline
        deploy_kwargs = {
            "model_id": model_name,
            "gpu": gpu,
            "refiner_model_id": refiner_model_name,
            "guidance": guidance,
            "steps": steps,
            "precision": precision,
            "negative_prompt": negative_prompt,
            "output_dir": dest_path.name,
            "two_stage_refiner": two_stage_refiner,
            "refiner_guidance": refiner_guidance,
            "refiner_steps": refiner_steps,
            "refiner_precision": refiner_precision,
            "refiner_negative_prompt": refiner_negative_prompt,
            "img_size": img_size,
            "diffusion_repo_id": diffusion_repo_id,
            "diffusion_filename": diffusion_filename,
            "vae_repo_id": vae_repo_id,
            "vae_filename": vae_filename,
            "clip_l_repo_id": clip_l_repo_id,
            "clip_l_filename": clip_l_filename,
            "t5xxl_repo_id": t5xxl_repo_id,
            "t5xxl_filename": t5xxl_filename,
        }
        
        run_parallel_pipeline(
            dest_path=dest_path,
            prompts_list=prompts_list,
            notebook=notebook,
            kernel_path=kernel_path,
            **deploy_kwargs
        )
        
        logging.info(f"Pipeline completed! Output saved to: {dest_path}")
        logging.info("Check remaining GPU quota: https://www.kaggle.com/settings#quotas")
        return

    # Sequential deployment for 4 or fewer prompts
    logging.info("Deploying kernel...")
    # Pass only the run name (e.g., '20251115_182905') so the notebook can
    # prepend 'output/' itself and avoid creating nested output paths.
    deploy.run(
        prompts_list=prompts_list,
        notebook=notebook,
        model_id=model_name,
        kernel_path=kernel_path,
        gpu=gpu,
        refiner_model_id=refiner_model_name,
        guidance=guidance,
        steps=steps,
        precision=precision,
        negative_prompt=negative_prompt,
        output_dir=dest_path.name,
        two_stage_refiner=two_stage_refiner,
        refiner_guidance=refiner_guidance,
        refiner_steps=refiner_steps,
        refiner_precision=refiner_precision,
        refiner_negative_prompt=refiner_negative_prompt,
        img_size=img_size,
        diffusion_repo_id=diffusion_repo_id,
        diffusion_filename=diffusion_filename,
        vae_repo_id=vae_repo_id,
        vae_filename=vae_filename,
        clip_l_repo_id=clip_l_repo_id,
        clip_l_filename=clip_l_filename,
        t5xxl_repo_id=t5xxl_repo_id,
        t5xxl_filename=t5xxl_filename,
    )
    logging.debug("Deploy step completed")

    # Step 2: Poll status
    logging.info("Polling kernel status...")
    status = poll_status.run()
    logging.debug("Poll status completed")

    if status == "kernelworkerstatus.error":
        log_path = dest_path / "stable-diffusion-batch-generator.log"
        logging.error(f"Kernel failed. See log: {log_path}")
        raise RuntimeError("Kaggle kernel failed during image generation. Aborting pipeline.")

    # Step 3: Download output (using selective downloader to get only images)
    logging.info("Downloading output artifacts (images only)...")
    kernel_id = "leventecsibi/stable-diffusion-batch-generator"
    success = download_selective.run(kernel_id=kernel_id, dest=str(dest_path))
    logging.debug("Download completed")
    
    if not success:
        logging.warning("Selective download did not complete successfully")

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
    parser.add_argument("--prompts_file", type=str, default="./config/prompts.json")
    parser.add_argument("--notebook", type=str, default=None, help="Notebook to use (auto-detects based on model if not specified)")
    parser.add_argument("--kernel_path", type=str, default="./config")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU for the kernel")
    parser.add_argument("--dest", type=str, default=None, help="Optional prefix for output folder (default: timestamp only)")
    parser.add_argument("--output_base_dir", type=str, default=None, help="Base directory for output runs (default: current working directory)")
    parser.add_argument("--model_name", type=str, default=None, help="Image generation model to use")
    parser.add_argument("--refiner_model_name", type=str, default=None, help="Refiner model to use")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt string")
    parser.add_argument("--prompts", type=str, nargs="+", default=None, help="Multiple prompts")
    parser.add_argument("--guidance", type=float, required=True, help="Guidance scale (7-12 recommended for photorealism)")
    parser.add_argument("--steps", type=int, required=True, help="Number of inference steps (50-100 for better quality)")
    parser.add_argument("--precision", type=str, required=True, choices=["fp32", "fp16", "bf16", "int8", "int4", "q4", "q5", "q6", "q8"],
                        help="Precision level (REQUIRED): fp32 (highest quality), fp16 (balanced), bf16 (recommended for Flux), q4/q5/q6/q8 (GGUF quantized), int8 (faster), int4 (fastest)")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Custom negative prompt for better quality control")
    parser.add_argument("--two_stage_refiner", action="store_true", help="Use two-stage approach: base model Ôćĺ unload VRAM Ôćĺ refiner model (saves VRAM)")
    parser.add_argument("--refiner_guidance", type=float, default=None, help="Guidance scale for refiner (REQUIRED when using --refiner_model_name)")
    parser.add_argument("--refiner_steps", type=int, default=None, help="Number of inference steps for refiner (REQUIRED when using --refiner_model_name)")
    parser.add_argument("--refiner_precision", type=str, default=None, choices=["fp32", "fp16", "bf16", "int8", "int4"],
                        help="Precision level for refiner (REQUIRED when using --refiner_model_name, or inherits from --precision)")
    parser.add_argument("--refiner_negative_prompt", type=str, default=None, help="Custom negative prompt for refiner (defaults to same as --negative_prompt)")
    parser.add_argument("--img_width", type=int, default=None, help="Image width (defaults: 1024 for stable diffusion, 512 for flux gguf)")
    parser.add_argument("--img_height", type=int, default=None, help="Image height (defaults: 1024 for stable diffusion, 512 for flux gguf)")
    
    # FLUX GGUF model configuration
    parser.add_argument("--diffusion_repo_id", type=str, default=None, help="HuggingFace repo ID for diffusion model (e.g., city96/FLUX.1-schnell-gguf)")
    parser.add_argument("--diffusion_filename", type=str, default=None, help="Diffusion model filename (e.g., flux1-schnell-Q5_0.gguf, flux1-schnell-Q6_K.gguf)")
    parser.add_argument("--vae_repo_id", type=str, default=None, help="HuggingFace repo ID for VAE model")
    parser.add_argument("--vae_filename", type=str, default=None, help="VAE model filename")
    parser.add_argument("--clip_l_repo_id", type=str, default=None, help="HuggingFace repo ID for CLIP-L model")
    parser.add_argument("--clip_l_filename", type=str, default=None, help="CLIP-L model filename")
    parser.add_argument("--t5xxl_repo_id", type=str, default=None, help="HuggingFace repo ID for T5-XXL model")
    parser.add_argument("--t5xxl_filename", type=str, default=None, help="T5-XXL model filename")

    args = parser.parse_args()

    # Auto-detect notebook based on model type if not specified
    if args.notebook is None:
        # Check if FLUX GGUF based on explicit parameters or model_name
        is_gguf = (args.diffusion_repo_id or args.diffusion_filename or 
                   (args.model_name and _is_flux_gguf_model(args.model_name)))
        is_bf16 = args.model_name and _is_flux_bf16_model(args.model_name)
        
        if is_gguf:
            args.notebook = "./config/kaggle-flux-gguf.ipynb"
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
        elif is_bf16:
            args.notebook = "./config/kaggle-flux-schnell-bf16.ipynb"
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
            args.notebook = "./config/kaggle-stable-diffusion.ipynb"
            print(f"Using default notebook: {args.notebook}")

    # Validate required image dimensions - NO DEFAULTS ALLOWED
    if not args.img_width or not args.img_height:
        print("\n" + "="*80)
        print("ÔÜá´ŞĆ  VALIDATION ERROR: MISSING IMAGE DIMENSIONS!")
        print("="*80)
        print("Both --img_width and --img_height are REQUIRED.")
        print("Please specify explicit image dimensions.")
        print("Examples:")
        print("  - Stable Diffusion XL: --img_width 1024 --img_height 1024")
        print("  - FLUX GGUF Q4: --img_width 512 --img_height 512")
        print("="*80 + "\n")
        return
    
    # Validate FLUX model dimensions must be multiples of 16
    if args.model_name and _is_flux_gguf_model(args.model_name):
        # Warn if guidance > 1.0 for FLUX GGUF models
        if args.guidance > 1.0:
            print("\n" + "="*80)
            print("⚠️  WARNING: HIGH GUIDANCE FOR FLUX GGUF MODEL!")
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
            print("ÔÜá´ŞĆ  VALIDATION ERROR: INVALID FLUX IMAGE DIMENSIONS!")
            print("="*80)
            print(f"FLUX models require dimensions to be multiples of 16.")
            print(f"You provided: {args.img_width}x{args.img_height}")
            print(f"Valid dimensions close to your request:")
            # Suggest nearest valid dimensions
            valid_width = (args.img_width // 16) * 16
            valid_height = (args.img_height // 16) * 16
            print(f"  - {valid_width}x{valid_height}")
            print(f"Common FLUX dimensions:")
            print(f"  - 512x512 (fast)")
            print(f"  - 768x768 (balanced)")
            print(f"  - 1024x1024 (high quality)")
            print(f"  - 1920x1088 (full HD, multiple of 16)")
            print("="*80 + "\n")
            return
    
    img_size = (args.img_height, args.img_width)

    # Precision is now required; no auto-detection
    logging.info(f"Using explicit precision: {args.precision}")

    # Validate arguments including precision availability
    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}")
        return
    run_pipeline(
        dest_path=dest_path,
        prompts_file=args.prompts_file,
        notebook=args.notebook,
        kernel_path=args.kernel_path,
        gpu=args.gpu,
        model_name=args.model_name,
        refiner_model_name=args.refiner_model_name,
        prompt=args.prompt,
        prompts=args.prompts,
        guidance=args.guidance,
        steps=args.steps,
        precision=args.precision,
        negative_prompt=args.negative_prompt,
        two_stage_refiner=args.two_stage_refiner,
        refiner_guidance=args.refiner_guidance,
        refiner_steps=args.refiner_steps,
        refiner_precision=args.refiner_precision,
        refiner_negative_prompt=args.refiner_negative_prompt,
        img_size=img_size,
        diffusion_repo_id=args.diffusion_repo_id,
        diffusion_filename=args.diffusion_filename,
        vae_repo_id=args.vae_repo_id,
        vae_filename=args.vae_filename,
        clip_l_repo_id=args.clip_l_repo_id,
        clip_l_filename=args.clip_l_filename,
        t5xxl_repo_id=args.t5xxl_repo_id,
        t5xxl_filename=args.t5xxl_filename,
    )


    # ...existing code...


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
