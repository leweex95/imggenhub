"""
Argument validation utilities for Kaggle image generation pipeline.
"""
from typing import Any

def is_kaggle_model(model_id: str) -> bool:
    if '/' not in model_id:
        return False
    owner = model_id.split('/')[0].lower()
    hf_orgs = {
        'stabilityai', 'black-forest-labs', 'runwayml', 'compvis', 'openai',
        'google', 'microsoft', 'facebook', 'huggingface', 'meta', 'anthropic',
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
    if not args.model_name:
        raise ValueError("--model_name is required")
    if not args.prompt and not args.prompts and not args.prompts_file:
        raise ValueError("No prompts provided")
    if args.img_width is None or args.img_height is None:
        raise ValueError("Both --img_width and --img_height are required.")
    if is_flux_gguf_model(args.model_name):
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
    use_refiner = (args.refiner_model_name is not None)
    if args.precision != "auto":
        if is_kaggle_model(args.model_name):
            pass
        elif "flux" in args.model_name.lower():
            pass
        else:
            from imggenhub.kaggle.utils.precision_validator import PrecisionValidator
            detector = PrecisionValidator(args.hf_token)
            available_variants = detector.detect_available_variants(args.model_name)
            if args.precision not in available_variants:
                available_str = ", ".join(available_variants) if available_variants else "none"
                raise ValueError(f"Precision '{args.precision}' not available for model '{args.model_name}'. Available: {available_str}")
    if use_refiner and args.refiner_precision:
        if not args.refiner_model_name:
            raise ValueError("--refiner_model_name is required when specifying --refiner_precision")
        refiner_model = args.refiner_model_name
        if is_kaggle_model(refiner_model):
            pass
        else:
            from imggenhub.kaggle.utils.precision_validator import PrecisionValidator
            detector = PrecisionValidator(args.hf_token)
            available_variants = detector.detect_available_variants(refiner_model)
            if args.refiner_precision not in available_variants:
                available_str = ", ".join(available_variants) if available_variants else "none"
                raise ValueError(f"Refiner precision '{args.refiner_precision}' not available for model '{refiner_model}'. Available: {available_str}")
    if use_refiner:
        if args.refiner_guidance is None:
            raise ValueError("--refiner_guidance is required when using a refiner model")
        if args.refiner_steps is None:
            raise ValueError("--refiner_steps is required when using a refiner model")
    if not args.prompt and not args.prompts and not args.prompts_file:
        raise ValueError("No prompts provided: specify --prompt, --prompts, or --prompts_file")
