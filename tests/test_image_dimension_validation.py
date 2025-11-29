import pytest
from types import SimpleNamespace
from imggenhub.kaggle.utils.arg_validator import validate_args

def test_invalid_flux_dimensions():
    args = SimpleNamespace(
        model_name="flux-gguf-q4",
        prompt="test",
        prompts=None,
        prompts_file=None,
        img_width=500,
        img_height=250,
        steps=10,
        guidance=7.5,
        precision="fp16",
        refiner_model_name=None,
        refiner_guidance=None,
        refiner_steps=None,
        refiner_precision=None,
        refiner_negative_prompt=None,
        gpu=True,
        notebook=None,
        kernel_path=None,
        dest=None,
        output_base_dir=None,
        negative_prompt=None,
        two_stage_refiner=False,
        hf_token=None,
        refiner_model=None
    )
    with pytest.raises(ValueError, match="FLUX models require image dimensions divisible by 16"):
        validate_args(args)

def test_invalid_sd_dimensions():
    args = SimpleNamespace(
        model_name="stabilityai/stable-diffusion-xl",
        prompt="test",
        prompts=None,
        prompts_file=None,
        img_width=500,
        img_height=250,
        steps=10,
        guidance=7.5,
        precision="fp16",
        refiner_model_name=None,
        refiner_guidance=None,
        refiner_steps=None,
        refiner_precision=None,
        refiner_negative_prompt=None,
        gpu=True,
        notebook=None,
        kernel_path=None,
        dest=None,
        output_base_dir=None,
        negative_prompt=None,
        two_stage_refiner=False,
        hf_token=None,
        refiner_model=None
    )
    with pytest.raises(ValueError, match="Stable Diffusion models require image dimensions divisible by 8"):
        validate_args(args)

def test_valid_flux_dimensions():
    args = SimpleNamespace(
        model_name="flux-gguf-q4",
        prompt="test",
        prompts=None,
        prompts_file=None,
        img_width=512,
        img_height=512,
        steps=10,
        guidance=7.5,
        precision="fp16",
        refiner_model_name=None,
        refiner_guidance=None,
        refiner_steps=None,
        refiner_precision=None,
        refiner_negative_prompt=None,
        gpu=True,
        notebook=None,
        kernel_path=None,
        dest=None,
        output_base_dir=None,
        negative_prompt=None,
        two_stage_refiner=False,
        hf_token=None,
        refiner_model=None
    )
    validate_args(args)  # Should not raise

def test_valid_sd_dimensions():
    args = SimpleNamespace(
        model_name="stabilityai/stable-diffusion-xl",
        prompt="test",
        prompts=None,
        prompts_file=None,
        img_width=512,
        img_height=512,
        steps=10,
        guidance=7.5,
        precision="fp16",
        refiner_model_name=None,
        refiner_guidance=None,
        refiner_steps=None,
        refiner_precision=None,
        refiner_negative_prompt=None,
        gpu=True,
        notebook=None,
        kernel_path=None,
        dest=None,
        output_base_dir=None,
        negative_prompt=None,
        two_stage_refiner=False,
        hf_token=None,
        refiner_model=None
    )
    validate_args(args)  # Should not raise
