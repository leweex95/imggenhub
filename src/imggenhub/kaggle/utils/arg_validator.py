"""
Argument validation utilities for Kaggle image generation pipeline.
"""
import os
from typing import Any
from imggenhub.kaggle.utils.model_family import (
    MODEL_FAMILY_FLUX_GGUF,
    MODEL_FAMILY_ILLUSTRIOUS_PONY,
    MODEL_FAMILY_QWEN_IMAGE,
    MODEL_FAMILY_SD35,
    MODEL_FAMILY_WAN21_CHROMA,
    detect_model_family,
    is_flux_gguf_model as _is_flux_gguf_model,
)

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
        'intel', 'apple', 'tencent', 'baidu', 'alibaba', 'bytedance',
        'qwen', 'qwenlm', 'qwen-vl', 'wan-ai', 'wan-video', 'city96',
        'comfyanonymous', 'kandinsky-community', 'lllyasviel'
    }
    return owner not in hf_orgs

def is_flux_gguf_model(model_id: str) -> bool:
    return _is_flux_gguf_model(model_id)

def validate_args(args: Any):
    model_family = detect_model_family(getattr(args, 'model_id', None), getattr(args, 'model_filename', None))

    # Warn if model_filename is provided for non-GGUF models
    has_model_filename = hasattr(args, 'model_filename') and args.model_filename
    if has_model_filename and model_family != MODEL_FAMILY_FLUX_GGUF:
        print("\n" + "="*80)
        print("WARNING: --model-filename is ignored for this model type")
        print("="*80)
        print(f"You specified --model-filename ({args.model_filename}) for a model type that does not use it.")
        print("Only GGUF quantized models require --model-filename.")
        print("Proceeding with generation...")
        print("="*80 + "\n")

    # Enforce --model-id required if --model-filename is provided
    if hasattr(args, 'model_filename') and args.model_filename:
        if not hasattr(args, 'model_id') or not args.model_id:
            raise ValueError("Error: --model-id is required when --model-filename is provided. Please specify --model-id explicitly.")

    # model_id is always required
    if not getattr(args, 'model_id', None):
        raise ValueError("--model-id is required.")

    if not args.prompt and not args.prompts_file:
        raise ValueError("No prompts provided")
    if args.img_width is None or args.img_height is None:
        raise ValueError("Both --img-width and --img-height are required.")
    if model_family == MODEL_FAMILY_FLUX_GGUF:
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
            raise ValueError("--wait-timeout must be a non-negative integer.")

    use_refiner = (getattr(args, 'refiner_model_id', None) is not None)
    if args.precision != "auto":
        should_skip_precision_validation = (
            is_kaggle_model(args.model_id)
            or "flux" in args.model_id.lower()
            or model_family in {
                MODEL_FAMILY_SD35,
                MODEL_FAMILY_WAN21_CHROMA,
                MODEL_FAMILY_QWEN_IMAGE,
                MODEL_FAMILY_ILLUSTRIOUS_PONY,
            }
        )
        if not should_skip_precision_validation:
            from imggenhub.kaggle.utils.precision_validator import PrecisionValidator
            hf_token = os.getenv("HF_TOKEN", "")
            detector = PrecisionValidator(hf_token)
            available_variants = detector.detect_available_variants(args.model_id)
            if args.precision not in available_variants:
                available_str = ", ".join(available_variants) if available_variants else "none"
                raise ValueError(f"Precision '{args.precision}' not available for model '{args.model_id}'. Available: {available_str}")
    # Check refiner requirements as soon as refiner_precision is set
    if args.refiner_precision and not getattr(args, 'refiner_model_id', None):
        raise ValueError("--refiner-model-id is required when specifying --refiner-precision")
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
            raise ValueError("--refiner-guidance is required when using a refiner model")
        if args.refiner_steps is None:
            raise ValueError("--refiner-steps is required when using a refiner model")
    if not args.prompt and not args.prompts_file:
        raise ValueError("No prompts provided: specify --prompt or --prompts-file")
