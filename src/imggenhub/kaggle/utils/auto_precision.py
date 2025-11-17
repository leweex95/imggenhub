#!/usr/bin/env python
"""
Auto-precision detection for HuggingFace models.
Integrates with the existing pipeline to automatically detect optimal precision.
"""

import requests
from typing import Optional, Tuple
import torch

class AutoPrecisionDetector:
    """Automatically detects the best available precision for a model."""

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token
        self.headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    def detect_available_variants(self, model_id: str) -> dict:
        """Get all available variants for a model."""
        try:
            files = self._get_model_files(model_id)
        except Exception:
            return {}

        return self._extract_variants_from_files(files, 'safetensors')

    def detect_best_precision(self, model_id: str, preferred_format: str = 'safetensors') -> Tuple[str, Optional[str]]:
        try:
            files = self._get_model_files(model_id)
        except Exception as e:
            print(f"Warning: Could not detect variants for {model_id}: {e}")
            return 'fp32', None  # Safe fallback

        available_variants = self._extract_variants_from_files(files, preferred_format)

        if not available_variants:
            print(f"Warning: No {preferred_format} variants found for {model_id}, using fp32")
            return 'fp32', None

        # Preference order: fp16 > bf16 > fp32 > int8 > int4
        preference_order = ['fp16', 'bf16', 'fp32', 'int8', 'int4']

        for preferred in preference_order:
            if preferred in available_variants:
                variant = preferred if preferred != 'fp32' else None
                return preferred, variant

        # Fallback to first available
        first_variant = available_variants[0]
        variant = first_variant if first_variant != 'fp32' else None
        return first_variant, variant

    def _get_model_files(self, model_id: str) -> list:
        """Get model files from HuggingFace API."""
        url = f"https://huggingface.co/api/models/{model_id}/tree/main"
        params = {"recursive": "true", "expand": "false"}
        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 403:
            raise ValueError(f"Access denied to {model_id}. Check token permissions.")
        elif response.status_code != 200:
            raise ValueError(f"API error {response.status_code}")

        return response.json()

    def _extract_variants_from_files(self, files: list, format_type: str) -> list:
        """Extract precision variants from file list."""
        extension = f".{format_type}"
        variants = []

        # Model file patterns to look for
        model_patterns = ['model', 'pytorch_model', 'diffusion_pytorch_model']

        for file_info in files:
            if file_info.get('type') != 'file':
                continue

            filename = file_info['path'].split('/')[-1]

            if not filename.endswith(extension):
                continue

            # Check if it's a model file
            is_model_file = any(pattern in filename for pattern in model_patterns)
            if not is_model_file:
                continue

            # Extract precision from filename
            variant = self._extract_precision_from_filename(filename)
            if variant and variant not in variants:
                variants.append(variant)

        return variants

    def _extract_precision_from_filename(self, filename: str) -> Optional[str]:
        """Extract precision indicator from filename."""
        name = filename.rsplit('.', 1)[0].lower()

        precision_map = {
            'fp16': ['fp16', 'half'],
            'fp32': ['fp32', 'float32'],
            'int8': ['int8', '8bit'],
            'int4': ['int4', '4bit'],
            'bf16': ['bf16', 'bfloat16'],
        }

        for precision, indicators in precision_map.items():
            if any(indicator in name for indicator in indicators):
                return precision

        return 'fp32'  # Default if no indicator found


def auto_detect_precision(model_id: str, hf_token: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Convenience function to auto-detect precision for a model.

    Usage:
        precision, variant = auto_detect_precision("black-forest-labs/FLUX.1-schnell", hf_token)
        # Use in your pipeline
    """
    detector = AutoPrecisionDetector(hf_token)
    return detector.detect_best_precision(model_id)
