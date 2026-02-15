"""
Model family detection helpers for Kaggle image generation.

These helpers keep model routing and validation logic consistent across CLI,
argument validation, and notebook execution.
"""
from __future__ import annotations

from typing import Optional


MODEL_FAMILY_FLUX_GGUF = "flux_gguf"
MODEL_FAMILY_FLUX_BF16 = "flux_bf16"
MODEL_FAMILY_SDXL = "sdxl"
MODEL_FAMILY_SD35 = "sd35"
MODEL_FAMILY_WAN21_CHROMA = "wan21_chroma"
MODEL_FAMILY_QWEN_IMAGE = "qwen_image"
MODEL_FAMILY_ILLUSTRIOUS_PONY = "illustrious_pony"
MODEL_FAMILY_STABLE_DIFFUSION = "stable_diffusion"


def _normalize_model_id(model_id: Optional[str]) -> str:
    return (model_id or "").strip().lower()


def is_flux_gguf_model(model_id: Optional[str]) -> bool:
    model_lower = _normalize_model_id(model_id)
    if not model_lower:
        return False
    return "flux" in model_lower and (
        "gguf" in model_lower or "q4" in model_lower or "q5" in model_lower or "q6" in model_lower or "q8" in model_lower
    )


def is_flux_bf16_model(model_id: Optional[str]) -> bool:
    model_lower = _normalize_model_id(model_id)
    if not model_lower:
        return False
    return ("flux" in model_lower and "black-forest-labs" in model_lower) or model_lower == "black-forest-labs/flux.1-schnell"


def detect_model_family(model_id: Optional[str], model_filename: Optional[str] = None) -> str:
    """
    Detect image model family from model_id and optional model filename.
    """
    model_lower = _normalize_model_id(model_id)
    model_filename_lower = (model_filename or "").strip().lower()

    if model_filename_lower and is_flux_gguf_model(model_lower):
        return MODEL_FAMILY_FLUX_GGUF
    if is_flux_gguf_model(model_lower):
        return MODEL_FAMILY_FLUX_GGUF
    if is_flux_bf16_model(model_lower):
        return MODEL_FAMILY_FLUX_BF16

    if any(token in model_lower for token in ("illustrious", "pony")):
        return MODEL_FAMILY_ILLUSTRIOUS_PONY

    if any(
        token in model_lower
        for token in (
            "stable-diffusion-3.5",
            "stable-diffusion-3-5",
            "sd3.5",
            "sd3_5",
            "sd35",
        )
    ):
        return MODEL_FAMILY_SD35

    if any(token in model_lower for token in ("wan2.1", "wan-2.1", "wan2_1", "wan-2", "wan2", "chroma")):
        return MODEL_FAMILY_WAN21_CHROMA

    if any(token in model_lower for token in ("qwen-image", "qwen image", "qwen-vl", "qwen2-vl", "qwen2_5-vl", "qwen")):
        return MODEL_FAMILY_QWEN_IMAGE

    if any(
        token in model_lower
        for token in ("stable-diffusion-xl", "sdxl", "juggernautxl", "juggernaut-xl", "juggernaut_xl", "sd_xl")
    ):
        return MODEL_FAMILY_SDXL

    return MODEL_FAMILY_STABLE_DIFFUSION


def supports_sdxl_refiner(model_family: str) -> bool:
    return model_family in (MODEL_FAMILY_SDXL, MODEL_FAMILY_ILLUSTRIOUS_PONY)
