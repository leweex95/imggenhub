# main.py
import argparse
import subprocess
import logging
from pathlib import Path
from imggenhub.kaggle.core import deploy, download
from imggenhub.kaggle.utils import poll_status
from imggenhub.kaggle.utils.prompts import resolve_prompts
from imggenhub.kaggle.utils.cli import log_cli_command, setup_output_directory
from imggenhub.kaggle.utils.filesystem import ensure_output_directory
from imggenhub.kaggle.utils.precision_validator import PrecisionValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_pipeline(dest_path, prompts_file, notebook, kernel_path, gpu=False, model_name=None, refiner_model_name=None, prompt=None, prompts=None, guidance=None, steps=None, precision=None, negative_prompt=None, two_stage_refiner=False, refiner_guidance=None, refiner_steps=None, refiner_precision=None, refiner_negative_prompt=None, hf_token=None, img_size=None):
    """Run Kaggle image generation pipeline: deploy -> poll -> download"""
    print("Initializing run_pipeline in main.py...")
    cwd = Path(__file__).parent

    # Resolve paths
    if prompts_file:
        prompts_file = Path(prompts_file)
        if not prompts_file.is_absolute():
            prompts_file = cwd / prompts_file

    notebook = Path(notebook)
    if not notebook.is_absolute():
        notebook = cwd / "config" / notebook.name  # Always resolve relative to config directory

    kernel_path = Path(kernel_path)
    if not kernel_path.is_absolute():
        kernel_path = cwd / "config"  # Always use config directory

    prompts_list = resolve_prompts(prompts_file, prompt, prompts)

    logging.debug(f"Resolved paths:\n prompts_file={prompts_file}\n notebook={notebook}\n kernel_path={kernel_path}\n dest={dest_path}")

    # Step 1: Deploy
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
        hf_token=hf_token,
        img_size=img_size,
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

    # Step 3: Download output
    logging.info("Downloading output artifacts...")
    download.run(dest_path)
    logging.debug("Download completed")

    logging.info(f"Pipeline completed! Output saved to: {dest_path}")
    logging.info("Check remaining GPU quota: https://www.kaggle.com/settings#quotas")


def main():
    """Main entry point - focused on argument parsing and orchestration"""
    parser = argparse.ArgumentParser(description="Kaggle image generation pipeline")
    parser.add_argument("--dest", type=str, default=None, help="Optional prefix for output folder (default: timestamp only)")
    parser.add_argument("--output_base_dir", type=str, default=None, help="Base directory for output runs (default: current working directory)")
    
    # Parse only known args to get output_base_dir and dest early
    early_args, remaining = parser.parse_known_args()
    
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
    parser.add_argument("--precision", type=str, required=True, choices=["fp32", "fp16", "bf16", "int8", "int4", "q4"],
                        help="Precision level (REQUIRED): fp32 (highest quality), fp16 (balanced), bf16 (recommended for Flux), q4 (GGUF quantized), int8 (faster), int4 (fastest)")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Custom negative prompt for better quality control")
    parser.add_argument("--two_stage_refiner", action="store_true", help="Use two-stage approach: base model → unload VRAM → refiner model (saves VRAM)")
    parser.add_argument("--refiner_guidance", type=float, default=None, help="Guidance scale for refiner (REQUIRED when using --refiner_model_name)")
    parser.add_argument("--refiner_steps", type=int, default=None, help="Number of inference steps for refiner (REQUIRED when using --refiner_model_name)")
    parser.add_argument("--refiner_precision", type=str, default=None, choices=["fp32", "fp16", "bf16", "int8", "int4"],
                        help="Precision level for refiner (REQUIRED when using --refiner_model_name, or inherits from --precision)")
    parser.add_argument("--refiner_negative_prompt", type=str, default=None, help="Custom negative prompt for refiner (defaults to same as --negative_prompt)")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token for accessing gated models (e.g., FLUX.1-schnell)")
    parser.add_argument("--img_width", type=int, default=None, help="Image width (defaults: 1024 for stable diffusion, 512 for flux gguf)")
    parser.add_argument("--img_height", type=int, default=None, help="Image height (defaults: 1024 for stable diffusion, 512 for flux gguf)")

    args = parser.parse_args()

    # Load HF_TOKEN from .env file if not provided via CLI
    if not args.hf_token:
        import os
        from pathlib import Path
        import sys
        
        # Try to load from .env in the project root
        env_path = Path(__file__).parent.parent.parent.parent / ".env"
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)
            args.hf_token = os.getenv("HF_TOKEN")
            if args.hf_token:
                print(f"[INFO] Loaded HF_TOKEN from .env file")
        else:
            # Try to load from current working directory
            try:
                from dotenv import load_dotenv
                load_dotenv()
                args.hf_token = os.getenv("HF_TOKEN")
                if args.hf_token:
                    print(f"[INFO] Loaded HF_TOKEN from .env in current directory")
            except:
                pass

    # Auto-detect notebook based on model type if not specified
    if args.notebook is None:
        if args.model_name and _is_flux_gguf_model(args.model_name):
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
        else:
            args.notebook = "./config/kaggle-stable-diffusion.ipynb"
            print(f"Using default notebook: {args.notebook}")

    # Validate required image dimensions - NO DEFAULTS ALLOWED
    if not args.img_width or not args.img_height:
        print("\n" + "="*80)
        print("⚠️  VALIDATION ERROR: MISSING IMAGE DIMENSIONS!")
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
        if args.img_width % 16 != 0 or args.img_height % 16 != 0:
            print("\n" + "="*80)
            print("⚠️  VALIDATION ERROR: INVALID FLUX IMAGE DIMENSIONS!")
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
        _validate_args(args)
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
        hf_token=args.hf_token,
        img_size=img_size
    )


def _validate_args(args):
    """Validate command line arguments with loud warnings for any fallbacks"""
    
    # Validate model name is provided
    if not args.model_name:
        print("="*80)
        print("[ERROR] --model_name is required")
        print("[ERROR] Example: --model_name flux-gguf-q4")
        print("="*80)
        raise ValueError("--model_name is required")
    
    # Validate prompts are provided
    if not args.prompt and not args.prompts and not args.prompts_file:
        print("="*80)
        print("[ERROR] No prompts provided")
        print("[ERROR] Specify at least one of:")
        print("[ERROR]   --prompt \"your prompt\"")
        print("[ERROR]   --prompts \"prompt1\" \"prompt2\"")
        print("[ERROR]   --prompts_file path/to/prompts.txt")
        print("="*80)
        raise ValueError("No prompts provided")
    
    # Validate compulsory parameters before any deployment
    if args.steps is None:
        print("="*80)
        print("[ERROR] --steps is required")
        print("[ERROR] Example: --steps 4")
        print("="*80)
        raise ValueError("--steps is required")
    
    if args.guidance is None:
        print("="*80)
        print("[ERROR] --guidance is required")
        print("[ERROR] Example: --guidance 1.0")
        print("="*80)
        raise ValueError("--guidance is required")
    
    if args.precision is None:
        print("="*80)
        print("[ERROR] --precision is required")
        print("[ERROR] Example: --precision q4")
        print("[ERROR] Available: fp16, fp32, q4, q8")
        print("="*80)
        raise ValueError("--precision is required")

    # Automatically enable refiner if refiner_model_name is specified
    use_refiner = (args.refiner_model_name is not None)

    # Validate precision availability for base model
    if args.precision != "auto":  # Skip if auto-detected (already validated)
        # Skip validation for Kaggle models (not on HuggingFace)
        if _is_kaggle_model(args.model_name):
            print(f"Skipping precision validation for Kaggle model: {args.model_name}")
        # Skip validation for FLUX models - they support all precisions via torch_dtype
        elif "flux" in args.model_name.lower():
            print(f"Skipping precision validation for FLUX model (supports all precisions): {args.model_name}")
        else:
            print(f"Validating precision '{args.precision}' availability for {args.model_name}...")
            detector = PrecisionValidator(args.hf_token)
            try:
                available_variants = detector.detect_available_variants(args.model_name)
                if args.precision not in available_variants:
                    available_str = ", ".join(available_variants) if available_variants else "none"
                    raise ValueError(f"Precision '{args.precision}' not available for model '{args.model_name}'. Available: {available_str}")
                print(f"[OK] Precision '{args.precision}' is available")
            except Exception as e:
                raise ValueError(f"Failed to validate precision for model '{args.model_name}': {e}")

    # Validate refiner precision if using refiner
    if use_refiner and args.refiner_precision:
        if not args.refiner_model_name:
            raise ValueError("--refiner_model_name is required when specifying --refiner_precision")
        refiner_model = args.refiner_model_name
        # Skip validation for Kaggle models (not on HuggingFace)
        if _is_kaggle_model(refiner_model):
            print(f"Skipping refiner precision validation for Kaggle model: {refiner_model}")
        else:
            print(f"Validating refiner precision '{args.refiner_precision}' availability for {refiner_model}...")
            detector = PrecisionValidator(args.hf_token)
            try:
                available_variants = detector.detect_available_variants(refiner_model)
                if args.refiner_precision not in available_variants:
                    available_str = ", ".join(available_variants) if available_variants else "none"
                    raise ValueError(f"Refiner precision '{args.refiner_precision}' not available for model '{refiner_model}'. Available: {available_str}")
                print(f"[OK] Refiner precision '{args.refiner_precision}' is available")
            except Exception as e:
                raise ValueError(f"Failed to validate refiner precision for model '{refiner_model}': {e}")

    # Validate refiner parameters if using refiner
    if use_refiner:
        if args.refiner_guidance is None:
            raise ValueError("--refiner_guidance is required when using a refiner model")
        if args.refiner_steps is None:
            raise ValueError("--refiner_steps is required when using a refiner model")

    # Validate prompts
    if not args.prompt and not args.prompts and not args.prompts_file:
        raise ValueError("No prompts provided: specify --prompt, --prompts, or --prompts_file")


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


if __name__ == "__main__":
    main()
