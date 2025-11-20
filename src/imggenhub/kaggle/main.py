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
from imggenhub.kaggle.utils.precision_availability import PrecisionAvailabilityChecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_pipeline(dest_path, prompts_file, notebook, kernel_path, gpu=False, model_name=None, refiner_model_name=None, prompt=None, prompts=None, guidance=None, steps=None, precision=None, negative_prompt=None, refiner_guidance=None, refiner_steps=None, refiner_precision=None, refiner_negative_prompt=None, hf_token=None):
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
        prompts_list,
        notebook,
        kernel_path,
        gpu,
        model_id=model_name,
        refiner_model_id=refiner_model_name,
        guidance=guidance,
        steps=steps,
        precision=precision,
        negative_prompt=negative_prompt,
        output_dir=dest_path.name,
        refiner_guidance=refiner_guidance,
        refiner_steps=refiner_steps,
        refiner_precision=refiner_precision,
        refiner_negative_prompt=refiner_negative_prompt,
        hf_token=hf_token,
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
    # Parse command line arguments first to get dest
    parser = argparse.ArgumentParser(description="Kaggle image generation pipeline")
    parser.add_argument("--prompts_file", type=str, default="./config/prompts.json")
    parser.add_argument("--notebook", type=str, default="./config/kaggle-notebook-image-generation.ipynb")
    parser.add_argument("--kernel_path", type=str, default="./config")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU for the kernel")
    parser.add_argument("--dest", type=str, default=None, help="Output directory base name (default: timestamp only)")
    parser.add_argument("--model_name", type=str, default=None, help="Image generation model to use")
    parser.add_argument("--refiner_model_name", type=str, default=None, help="Refiner model to use")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt string")
    parser.add_argument("--prompts", type=str, nargs="+", default=None, help="Multiple prompts")
    parser.add_argument("--guidance", type=float, required=True, help="Guidance scale (7-12 recommended for photorealism)")
    parser.add_argument("--steps", type=int, required=True, help="Number of inference steps (50-100 for better quality)")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8", "int4"],
                        default=None, help="Precision level: fp32 (highest quality), fp16 (balanced), int8 (faster), int4 (fastest)")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Custom negative prompt for better quality control")
    parser.add_argument("--refiner_guidance", type=float, default=None, help="Guidance scale for refiner (REQUIRED when using --refiner_model_name)")
    parser.add_argument("--refiner_steps", type=int, default=None, help="Number of inference steps for refiner (REQUIRED when using --refiner_model_name)")
    parser.add_argument("--refiner_precision", type=str, default=None, choices=["fp32", "fp16", "int8", "int4"],
                        help="Precision level for refiner (REQUIRED when using --refiner_model_name)")
    parser.add_argument("--refiner_negative_prompt", type=str, default=None, help="Custom negative prompt for refiner (defaults to same as --negative_prompt)")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token for accessing gated models (e.g., FLUX.1-schnell)")

    # Parse known args first to get dest for output directory setup
    args, remaining = parser.parse_known_args()
    
    # Set up output directory using dest argument
    dest_path = setup_output_directory(args.dest)
    log_cli_command(dest_path)

    # Re-parse all arguments now that we have dest_path
    args = parser.parse_args()

    # Warn if GPU is not enabled
    if not args.gpu:
        print("\033[91m\033[1mWARNING: --gpu flag not provided! This will run on CPU and may take several hours to complete.\033[0m")
        print("\033[93mConsider adding --gpu to enable GPU acceleration on Kaggle.\033[0m")

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
        refiner_guidance=args.refiner_guidance,
        refiner_steps=args.refiner_steps,
        refiner_precision=args.refiner_precision,
        refiner_negative_prompt=args.refiner_negative_prompt,
        hf_token=args.hf_token
    )


def _validate_args(args):
    """Validate command line arguments"""
    # Automatically enable refiner if refiner_model_name is specified
    use_refiner = (args.refiner_model_name is not None)

    # Validate compulsory parameters before any deployment
    if args.steps is None:
        raise ValueError("--steps is required")
    if args.guidance is None:
        raise ValueError("--guidance is required")
    if args.precision is None:
        raise ValueError("--precision is required (no auto-detection or defaults allowed)")
    if args.precision not in ["fp32", "fp16", "int8", "int4"]:
        raise ValueError(f"--precision must be one of: fp32, fp16, int8, int4. Got: {args.precision}")

    # Validate parameter types and ranges
    if not isinstance(args.steps, int) or args.steps <= 0:
        raise ValueError(f"--steps must be a positive integer. Got: {args.steps}")
    if not isinstance(args.guidance, (int, float)) or args.guidance <= 0:
        raise ValueError(f"--guidance must be a positive number. Got: {args.guidance}")

    # Validate model name is provided
    if not args.model_name:
        raise ValueError("--model_name is required")
    if not isinstance(args.model_name, str) or not args.model_name.strip():
        raise ValueError("--model_name must be a non-empty string")

    # Validate prompts
    if not args.prompt and not args.prompts and not args.prompts_file:
        raise ValueError("No prompts provided: specify --prompt, --prompts, or --prompts_file")
    if args.prompt and not isinstance(args.prompt, str):
        raise ValueError("--prompt must be a string")
    if args.prompts and not isinstance(args.prompts, list):
        raise ValueError("--prompts must be a list of strings")
    if args.prompts_file and not isinstance(args.prompts_file, str):
        raise ValueError("--prompts_file must be a string")

    # Validate refiner parameters if using refiner
    if args.refiner_model_name:
        if not isinstance(args.refiner_model_name, str) or not args.refiner_model_name.strip():
            raise ValueError("--refiner_model_name must be a non-empty string")
        if args.refiner_guidance is None:
            raise ValueError("--refiner_guidance is required when using a refiner model")
        if args.refiner_steps is None:
            raise ValueError("--refiner_steps is required when using a refiner model")
        if not isinstance(args.refiner_guidance, (int, float)) or args.refiner_guidance <= 0:
            raise ValueError(f"--refiner_guidance must be a positive number. Got: {args.refiner_guidance}")
        if not isinstance(args.refiner_steps, int) or args.refiner_steps <= 0:
            raise ValueError(f"--refiner_steps must be a positive integer. Got: {args.refiner_steps}")
        if args.refiner_precision is None:
            raise ValueError("--refiner_precision is required when using a refiner model")
        if args.refiner_precision not in ["fp32", "fp16", "int8", "int4"]:
            raise ValueError(f"--refiner_precision must be one of: fp32, fp16, int8, int4. Got: {args.refiner_precision}")

    # Validate precision availability for base model
    # Skip validation for Kaggle models (not on HuggingFace)
    if _is_kaggle_model(args.model_name):
        print(f"Skipping precision validation for Kaggle model: {args.model_name}")
    else:
        print(f"Validating precision '{args.precision}' availability for {args.model_name}...")
        detector = PrecisionAvailabilityChecker(args.hf_token)
        try:
            available_variants = detector.get_available_precisions(args.model_name)
            if args.precision not in available_variants:
                available_str = ", ".join(available_variants) if available_variants else "none"
                raise ValueError(f"Precision '{args.precision}' not available for model '{args.model_name}'. Available: {available_str}")
            print(f"[OK] Precision '{args.precision}' is available")
        except Exception as e:
            raise ValueError(f"Failed to validate precision for model '{args.model_name}': {e}")

    # Validate refiner precision availability if using refiner
    if use_refiner:
        refiner_model = args.refiner_model_name
        # Skip validation for Kaggle models (not on HuggingFace)
        if _is_kaggle_model(refiner_model):
            print(f"Skipping refiner precision validation for Kaggle model: {refiner_model}")
        else:
            print(f"Validating refiner precision '{args.refiner_precision}' availability for {refiner_model}...")
            detector = PrecisionAvailabilityChecker(args.hf_token)
            try:
                available_variants = detector.get_available_precisions(refiner_model)
                if args.refiner_precision not in available_variants:
                    available_str = ", ".join(available_variants) if available_variants else "none"
                    raise ValueError(f"Refiner precision '{args.refiner_precision}' not available for model '{refiner_model}'. Available: {available_str}")
                print(f"[OK] Refiner precision '{args.refiner_precision}' is available")
            except Exception as e:
                raise ValueError(f"Failed to validate refiner precision for model '{refiner_model}': {e}")


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


if __name__ == "__main__":
    main()
