from imggenhub.kaggle.utils.model_family import (
    MODEL_FAMILY_FLUX_BF16,
    MODEL_FAMILY_FLUX_GGUF,
    MODEL_FAMILY_ILLUSTRIOUS_PONY,
    MODEL_FAMILY_QWEN_IMAGE,
    MODEL_FAMILY_SD35,
    MODEL_FAMILY_SDXL,
    MODEL_FAMILY_STABLE_DIFFUSION,
    MODEL_FAMILY_WAN21_CHROMA,
    detect_model_family,
    is_flux_bf16_model,
    is_flux_gguf_model,
    supports_sdxl_refiner,
)


def test_detect_model_family_flux_gguf():
    assert detect_model_family("city96/FLUX.1-schnell-gguf") == MODEL_FAMILY_FLUX_GGUF


def test_detect_model_family_flux_bf16():
    assert detect_model_family("black-forest-labs/FLUX.1-schnell") == MODEL_FAMILY_FLUX_BF16


def test_detect_model_family_sdxl():
    assert detect_model_family("stabilityai/stable-diffusion-xl-base-1.0") == MODEL_FAMILY_SDXL
    assert detect_model_family("RunDiffusion/JuggernautXL-v9") == MODEL_FAMILY_SDXL
    assert detect_model_family("RunDiffusion/Juggernaut-XL-v9") == MODEL_FAMILY_SDXL


def test_detect_model_family_sd35():
    assert detect_model_family("stabilityai/stable-diffusion-3.5-large") == MODEL_FAMILY_SD35
    assert detect_model_family("stabilityai/stable-diffusion-3.5-medium") == MODEL_FAMILY_SD35


def test_detect_model_family_wan_chroma():
    assert detect_model_family("Wan-AI/Wan2.1-T2I-14B") == MODEL_FAMILY_WAN21_CHROMA
    assert detect_model_family("chroma-labs/chroma-image-model") == MODEL_FAMILY_WAN21_CHROMA


def test_detect_model_family_qwen():
    assert detect_model_family("Qwen/Qwen-Image") == MODEL_FAMILY_QWEN_IMAGE
    assert detect_model_family("Qwen/Qwen2.5-VL-7B-Instruct") == MODEL_FAMILY_QWEN_IMAGE


def test_detect_model_family_illustrious_pony():
    assert detect_model_family("fancy/checkpoint-illustrious-xl") == MODEL_FAMILY_ILLUSTRIOUS_PONY
    assert detect_model_family("fancy/pony-diffusion-xl") == MODEL_FAMILY_ILLUSTRIOUS_PONY


def test_detect_model_family_fallback():
    assert detect_model_family("runwayml/stable-diffusion-v1-5") == MODEL_FAMILY_STABLE_DIFFUSION


def test_flux_detection_helpers():
    assert is_flux_gguf_model("model/flux-q5_0-gguf")
    assert is_flux_gguf_model("model/flux-q6_0-gguf")
    assert not is_flux_gguf_model("black-forest-labs/FLUX.1-schnell")
    assert is_flux_bf16_model("black-forest-labs/FLUX.1-schnell")


def test_refiner_support_helper():
    assert supports_sdxl_refiner(MODEL_FAMILY_SDXL)
    assert supports_sdxl_refiner(MODEL_FAMILY_ILLUSTRIOUS_PONY)
    assert not supports_sdxl_refiner(MODEL_FAMILY_SD35)
