"""
Argument validation utilities for Kaggle image generation pipeline.
"""
import os
from typing import Any

def is_kaggle_model(model_id: str) -> bool:
    if not model_id or '/' not in model_id:
        return False
    parts = model_id.split('/')
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return False
    owner = parts[0].lower()
    hf_orgs = {
        'stabilityai', 'black-forest-labs', 'runwayml', 'compvis', 'openai',
        'google', 'microsoft', 'facebook', 'huggingface', 'meta', 'meta-llama', 'anthropic',
        'eleutherai', 'bigscience', 'bigcode', 'salesforce', 'amazon', 'nvidia',
        'intel', 'apple', 'tencent', 'baidu', 'alibaba', 'bytedance'
    }
    return owner not in hf_orgs

def is_flux_gguf_model(model_id: str) -> bool:
    if not model_id:
        return False
    model_lower = model_id.lower()
    return ('flux' in model_lower and ('gguf' in model_lower or 'q4' in model_lower or 'q8' in model_lower))

def validate_args(args: Any):
    # Warn if model_filename is provided for non-GGUF models
    is_sd_model = hasattr(args, 'model_id') and args.model_id and ("stabilityai" in args.model_id.lower())
    is_flux_bf16 = hasattr(args, 'model_id') and args.model_id and ('flux' in args.model_id.lower() and 'gguf' not in args.model_id.lower())
    is_flux_gguf = hasattr(args, 'model_filename') and args.model_filename and (args.model_id and 'gguf' in args.model_id.lower())
    if hasattr(args, 'model_filename') and args.model_filename and (is_sd_model or is_flux_bf16):
        print("\n" + "="*80)
        print("WARNING: --model_filename is ignored for this model type")
        print("="*80)
        print(f"You specified --model_filename ({args.model_filename}) for a model type that does not use it.")
        print("Only GGUF quantized models require --model_filename.")
        print("Proceeding with generation...")
        print("="*80 + "\n")

    # Enforce --model_id required if --model_filename is provided
    if hasattr(args, 'model_filename') and args.model_filename:
        if not hasattr(args, 'model_id') or not args.model_id:
            raise ValueError("Error: --model_id is required when --model_filename is provided. Please specify --model_id explicitly.")

    # model_id is always required
    if not getattr(args, 'model_id', None):
        raise ValueError("--model_id is required.")

    if not args.prompt and not args.prompts_file:
        raise ValueError("No prompts provided")
    if args.img_width is None or args.img_height is None:
        raise ValueError("Both --img_width and --img_height are required.")
    if is_flux_gguf_model(args.model_id):
        if args.img_width % 16 != 0 or args.img_height % 16 != 0:
            next_width = ((args.img_width // 16) + 1) * 16 if args.img_width % 16 != 0 else args.img_width
            next_height = ((args.img_height // 16) + 1) * 16 if args.img_height % 16 != 0 else args.img_height
            raise ValueError(f"FLUX models require image dimensions divisible by 16. Got: {args.img_width}x{args.img_height}. Recommendation: {next_width}x{next_height}")
    else:
        if args.img_width % 8 != 0 or args.img_height % 8 != 0:
            next_width = ((args.img_width // 8) + 1) * 8 if args.img_width % 8 != 0 else args.img_width
            next_height = ((args.img_height // 8) + 1) * 8 if args.img_height % 8 != 0 else args.img_height
            raise ValueError(f"Stable Diffusion models require image dimensions divisible by 8. Got: {args.img_width}x{args.img_height}. Recommendation: {next_width}x{next_height}")
    if args.steps is None:
        raise ValueError("--steps is required")
    if args.guidance is None:
        raise ValueError("--guidance is required")
    if args.precision is None:
        raise ValueError("--precision is required")
    
    if hasattr(args, 'wait_timeout') and args.wait_timeout is not None:
        if args.wait_timeout < 0:
            raise ValueError("--wait_timeout must be a non-negative integer.")

    use_refiner = (getattr(args, 'refiner_model_id', None) is not None)
    if args.precision != "auto":
        if is_kaggle_model(args.model_id):
            pass
        elif "flux" in args.model_id.lower():
            pass
        else:
            from imggenhub.kaggle.utils.precision_validator import PrecisionValidator
            hf_token = os.getenv("HF_TOKEN", "")
            detector = PrecisionValidator(hf_token)
            available_variants = detector.detect_available_variants(args.model_id)
            if args.precision not in available_variants:
                available_str = ", ".join(available_variants) if available_variants else "none"
                raise ValueError(f"Precision '{args.precision}' not available for model '{args.model_id}'. Available: {available_str}")
    # Check refiner requirements as soon as refiner_precision is set
    if args.refiner_precision and not getattr(args, 'refiner_model_id', None):
        raise ValueError("--refiner_model_id is required when specifying --refiner_precision")
        refiner_model = args.refiner_model_id
        if is_kaggle_model(refiner_model):
            pass
        else:
            from imggenhub.kaggle.utils.precision_validator import PrecisionValidator
            hf_token = os.getenv("HF_TOKEN", "")
            detector = PrecisionValidator(hf_token)
            available_variants = detector.detect_available_variants(refiner_model)
            if args.refiner_precision not in available_variants:
                available_str = ", ".join(available_variants) if available_variants else "none"
                raise ValueError(f"Refiner precision '{args.refiner_precision}' not available for model '{refiner_model}'. Available: {available_str}")
    if use_refiner:
        if args.refiner_guidance is None:
            raise ValueError("--refiner_guidance is required when using a refiner model")
        if args.refiner_steps is None:
            raise ValueError("--refiner_steps is required when using a refiner model")
    if not args.prompt and not args.prompts_file:
        raise ValueError("No prompts provided: specify --prompt or --prompts_file")
