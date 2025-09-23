import os
import re
import json
from pathlib import Path
import subprocess


def run(prompts_list, notebook="kaggle-notebook-image-generation.ipynb", kernel_path=".", gpu=None, model_id=None):
    """
    Deploy Kaggle notebook kernel, optionally overriding prompts and model.
    """

    # Load notebook
    nb_path = Path(notebook)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Update PROMPTS and MODEL_ID in notebook
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            # Update PROMPTS
            for i, line in enumerate(cell["source"]):
                if line.strip().startswith("PROMPTS ="):
                    cell["source"][i] = f"PROMPTS = {prompts_list}\n"
                    break
            # Update MODEL_ID if provided
            if model_id:
                for i, line in enumerate(cell["source"]):
                    if line.strip().startswith("MODEL_ID ="):
                        cell["source"][i] = f"MODEL_ID = \"{model_id}\"\n"
                        break

    # Save updated notebook
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    # Update kernel metadata
    metadata_path = Path(kernel_path) / "kernel-metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        kernel_meta = json.load(f)
    if gpu is not None:
        kernel_meta["enable_gpu"] = str(gpu).lower()
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(kernel_meta, f, indent=2)

    # Push via Kaggle CLI
    subprocess.run(["python", "-m", "kaggle.cli", "kernels", "push", "-p", str(kernel_path)], check=True)
