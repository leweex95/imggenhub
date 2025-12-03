#!/usr/bin/env python
"""
Precision validation for HuggingFace models.
Validates that requested precision variants are available for models.
"""

import requests
from typing import Optional, List


class PrecisionValidator:
    """Validates precision availability for HuggingFace models."""

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token
        self.headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    def detect_available_variants(self, model_id: str) -> List[str]:
        """Get all available precision variants for a model."""
        try:
            files = self._get_model_files(model_id)
        except Exception:
            return []

        return self._extract_variants_from_files(files, 'safetensors')

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

    def _extract_variants_from_files(self, files: list, format_type: str) -> List[str]:
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

        return None  # No default; fail loudly if not found
